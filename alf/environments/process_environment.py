# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Step a single env in a separate process for lock free paralellism.

Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/parallel_py_environment.py
"""

import atexit
import multiprocessing
import sys
import traceback
import torch
import numpy as np

from absl import logging

import alf.nest as nest
# import torch.multiprocessing as multiprocessing
from torch.multiprocessing.queue import ConnectionWrapper
from alf.data_structures import TimeStep


class ProcessEnvironment(object):

    # Message types for communication via the pipe.
    _READY = 1
    _ACCESS = 2
    _CALL = 3
    _RESULT = 4
    _EXCEPTION = 5
    _CLOSE = 6

    def __init__(self, env_constructor, flatten=False):
        """Step environment in a separate process for lock free paralellism.

    The environment is created in an external process by calling the provided
    callable. This can be an environment class, or a function creating the
    environment and potentially wrapping it. The returned environment should
    not access global variables.

    Args:
      env_constructor: Callable that creates and returns a Python environment.
      flatten: Boolean, whether to assume flattened actions and time_steps
        during communication to avoid overhead.

    Attributes:
      observation_spec: The cached observation spec of the environment.
      action_spec: The cached action spec of the environment.
      time_step_spec: The cached time step spec of the environment.
    """
        self._env_constructor = env_constructor
        self._flatten = flatten
        self._observation_spec = None
        self._action_spec = None
        self._time_step_spec = None

    def start(self, wait_to_start=True):
        """Start the process.

    Args:
      wait_to_start: Whether the call should wait for an env initialization.
    """
        self._conn, conn = multiprocessing.Pipe()
        # self._conn = ConnectionWrapper(self._conn)
        # conn = ConnectionWrapper(conn)
        self._process = multiprocessing.Process(
            target=self._worker,
            args=(conn, self._env_constructor, self._flatten))
        atexit.register(self.close)
        self._process.start()
        if wait_to_start:
            self.wait_start()

    def wait_start(self):
        """Wait for the started process to finish initialization."""
        result = self._conn.recv()
        if isinstance(result, Exception):
            self._conn.close()
            self._process.join(5)
            raise result
        assert result == self._READY, result

    def observation_spec(self):
        if not self._observation_spec:
            self._observation_spec = self.call('observation_spec')()
        return self._observation_spec

    def action_spec(self):
        if not self._action_spec:
            self._action_spec = self.call('action_spec')()
        return self._action_spec

    def time_step_spec(self):
        if not self._time_step_spec:
            self._time_step_spec = self.call('time_step_spec')()
        return self._time_step_spec

    def __getattr__(self, name):
        """Request an attribute from the environment.

    Note that this involves communication with the external process, so it can
    be slow.

    Args:
      name: Attribute to access.

    Returns:
      Value of the attribute.
    """
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.

    Args:
      name: Name of the method to call.
      *args: Positional arguments to forward to the method.
      **kwargs: Keyword arguments to forward to the method.

    Returns:
      Promise object that blocks and provides the return value when called.
    """
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join(5)

    def step(self, action, blocking=True):
        """Step the environment.

    Args:
      action: The action to apply to the environment.
      blocking: Whether to wait for the result.

    Returns:
      time step when blocking, otherwise callable that returns the time step.
    """
        promise = self.call('step', action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=True):
        """Reset the environment.

    Args:
      blocking: Whether to wait for the result.

    Returns:
      New observation when blocking, otherwise callable that returns the new
      observation.
    """
        promise = self.call('reset')
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        """Wait for a message from the worker process and return its payload.

    Raises:
      Exception: An exception was raised inside the worker process.
      KeyError: The reveived message is of an unknown type.

    Returns:
      Payload object of the message.
    """
        message, payload = self._conn.recv()
        # convert received numpy arrays back to tensors
        if all(
                map(lambda x: isinstance(x, np.ndarray),
                    nest.flatten(payload))):
            payload = nest.map_structure(lambda array: torch.from_numpy(array),
                                         payload)
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        self.close()
        raise KeyError(
            'Received message of unexpected type {}'.format(message))

    def _worker(self, conn, env_constructor, flatten=False):
        """The process waits for actions and sends back environment results.

    Args:
      conn: Connection for communication to the main process.
      env_constructor: env_constructor for the OpenAI Gym environment.
      flatten: Boolean, whether to assume flattened actions and time_steps
        during communication to avoid overhead.

    Raises:
      KeyError: When receiving a message of unknown type.
    """
        try:
            env = env_constructor()
            action_spec = env.action_spec()
            conn.send(self._READY)  # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    if flatten and name == 'step':
                        args = [nest.pack_sequence_as(action_spec, args[0])]
                    result = getattr(env, name)(*args, **kwargs)
                    # convert tensors to numpy arrays before sending
                    if all(map(torch.is_tensor, nest.flatten(result))):
                        result = nest.map_structure(
                            lambda tensor: tensor.numpy(), result)
                    if flatten and name in ['step', 'reset']:
                        result = nest.flatten(result)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    env.close()
                    break
                raise KeyError(
                    'Received message of unknown type {}'.format(message))
        except Exception:  # pylint: disable=broad-except
            etype, evalue, tb = sys.exc_info()
            stacktrace = ''.join(traceback.format_exception(etype, evalue, tb))
            message = 'Error in environment process: {}'.format(stacktrace)
            logging.error(message)
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            conn.close()

        def render(self, mode='human'):
            """Render the environment.

        Args:
            mode: One of ['rgb_array', 'human']. Renders to an numpy array, or brings
                up a window where the environment can be visualized.
        Returns:
            An ndarray of shape [width, height, 3] denoting an RGB image if mode is
            `rgb_array`. Otherwise return nothing and render directly to a display
            window.
        Raises:
            NotImplementedError: If the environment does not support rendering.
        """
            return self.call('render', mode)()
