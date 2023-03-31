# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from absl import logging
import atexit
from enum import Enum
from functools import partial
import multiprocessing
import numpy as np
import sys
import torch
import traceback

import alf
from alf.data_structures import TimeStep
import alf.nest as nest
from . import _penv


class _MessageType(Enum):
    """Message types for communication via the pipe.

    The ProcessEnvironment uses pipe to perform IPC, where each of the message
    has a message type. This Enum provides all the available message types.
    """
    READY = 1
    ACCESS = 2
    CALL = 3
    RESULT = 4
    EXCEPTION = 5
    CLOSE = 6


def _worker(conn,
            env_constructor,
            env_id=None,
            flatten=False,
            fast=False,
            num_envs=0,
            name=''):
    """The process waits for actions and sends back environment results.

    Args:
        conn (multiprocessing.connection): Connection for communication to the main process.
        env_constructor (Callable): callable environment creator.
        flatten (bool): whether to assume flattened actions and time_steps
          during communication to avoid overhead.
        fast (bool): whether created by ``FastParallelEnvironment`` or not.
        num_envs (int): number of environments in the ``FastParallelEnvironment``.
            Only used if ``fast`` is True.
        name (str): name of the FastParallelEnvironment. Only used if ``fast``
            is True.

    Raises:
        KeyError: When receiving a message of unknown type.
    """
    try:
        alf.set_default_device("cpu")
        env = env_constructor(env_id=env_id)
        action_spec = env.action_spec()
        if fast:
            penv = _penv.ProcessEnvironment(
                env, partial(process_call, conn, env, flatten,
                             action_spec), env_id, num_envs, env.action_spec(),
                env.time_step_spec()._replace(env_info=env.env_info_spec()),
                name)
            conn.send(_MessageType.READY)  # Ready.
            try:
                penv.worker()
            except KeyboardInterrupt:
                penv.quit()
            except Exception as e:
                print(e)
                traceback.print_exc()
                penv.quit()
        else:
            conn.send(_MessageType.READY)  # Ready.
            while True:
                if not process_call(conn, env, flatten, action_spec):
                    break
    except KeyboardInterrupt:
        # When worker receives interruption from keyboard (i.e. Ctrl-C), notify
        # the parent process to shut down quietly by sending the CLOSE message.
        #
        # This is to avoid sometimes tens of environment processes panicking
        # simultaneously.
        conn.send((_MessageType.CLOSE, None))
    except Exception:  # pylint: disable=broad-except
        etype, evalue, tb = sys.exc_info()
        stacktrace = ''.join(traceback.format_exception(etype, evalue, tb))
        message = 'Error in environment process: {}'.format(stacktrace)
        logging.error(message)
        conn.send((_MessageType.EXCEPTION, stacktrace))
    finally:
        conn.close()


def process_call(conn, env, flatten, action_spec):
    """
    Returns:
        True: continue to work
        False: end the worker
    """
    try:
        # Only block for short times to have keyboard exceptions be raised.
        while True:
            if conn.poll(0.1):
                break
        message, payload = conn.recv()
    except (EOFError, KeyboardInterrupt):
        return False
    if message == _MessageType.ACCESS:
        name = payload
        result = getattr(env, name)
        conn.send((_MessageType.RESULT, result))
    elif message == _MessageType.CALL:
        name, args, kwargs = payload
        if flatten and name == 'step':
            args = [nest.pack_sequence_as(action_spec, args[0])]
        result = getattr(env, name)(*args, **kwargs)
        if flatten and name in ['step', 'reset']:
            result = nest.flatten(result)
            assert all([not isinstance(x, torch.Tensor) for x in result
                        ]), ("Tensor result is not allowed: %s" % name)
        conn.send((_MessageType.RESULT, result))
    elif message == _MessageType.CLOSE:
        assert payload is None
        env.close()
        return False
    else:
        raise KeyError('Received message of unknown type {}'.format(message))
    return True


class ProcessEnvironment(object):

    def __init__(self,
                 env_constructor,
                 env_id=None,
                 flatten=False,
                 fast=False,
                 num_envs=0,
                 name=""):
        """Step environment in a separate process for lock free paralellism.

        The environment is created in an external process by calling the provided
        callable. This can be an environment class, or a function creating the
        environment and potentially wrapping it. The returned environment should
        not access global variables.

        Args:
            env_constructor (Callable): callable environment creator.
            env_id (torch.int32): ID of the the env
            flatten (bool): whether to assume flattened actions and time_steps
                during communication to avoid overhead.
            fast (bool): whether created by ``FastParallelEnvironment`` or not.
            num_envs (int): number of environments in the ``FastParallelEnvironment``.
                Only used if ``fast`` is True.
            name (str): name of the FastParallelEnvironment. Only used if ``fast``
                is True.

        Attributes:
            observation_spec: The cached observation spec of the environment.
            action_spec: The cached action spec of the environment.
            time_step_spec: The cached time step spec of the environment.
        """
        self._env_constructor = env_constructor
        self._flatten = flatten
        self._env_id = env_id
        self._observation_spec = None
        self._action_spec = None
        self._reward_spec = None
        self._time_step_spec = None
        self._env_info_spec = None
        self._conn = None
        self._fast = fast
        self._num_envs = num_envs
        self._name = name
        if fast:
            self._penv = _penv.ProcessEnvironmentCaller(env_id, name)

    def start(self, wait_to_start=True):
        """Start the process.

        Args:
            wait_to_start (bool): Whether the call should wait for an env initialization.
        """
        # The following context made sure that the newly created child process
        # (for environment) is started using the "fork" start method.
        #
        # This is to prevent multiprocessing from accidentally creating the
        # child process with the "spawn" start method. Using "fork" start method
        # is required here because we would like to have the child process
        # inherit the alf configurations from the parent process, so that such
        # configuration are effective for the to-be-created environments in the
        # child process.
        assert not self._conn, "Cannot start() ProcessEnvironment multiple times"
        mp_ctx = multiprocessing.get_context('fork')
        self._conn, conn = mp_ctx.Pipe()
        self._process = mp_ctx.Process(
            target=_worker,
            args=(conn, self._env_constructor, self._env_id, self._flatten,
                  self._fast, self._num_envs, self._name))
        atexit.register(self.close)
        self._process.start()
        if wait_to_start:
            self.wait_start()

    def wait_start(self):
        """Wait for the started process to finish initialization."""
        assert self._conn, "Run ProcessEnvironment.start() first"
        result = self._conn.recv()
        if isinstance(result, Exception):
            self._conn.close()
            self._process.join(5)
            raise result
        assert result == _MessageType.READY, result

    def env_info_spec(self):
        if not self._env_info_spec:
            self._env_info_spec = self.call('env_info_spec')()
        return self._env_info_spec

    def observation_spec(self):
        if not self._observation_spec:
            self._observation_spec = self.call('observation_spec')()
        return self._observation_spec

    def action_spec(self):
        if not self._action_spec:
            self._action_spec = self.call('action_spec')()
        return self._action_spec

    def reward_spec(self):
        if not self._reward_spec:
            self._reward_spec = self.call('reward_spec')()
        return self._reward_spec

    def time_step_spec(self):
        if not self._time_step_spec:
            self._time_step_spec = self.call('time_step_spec')()
        return self._time_step_spec

    def __getattr__(self, name):
        """Request an attribute from the environment.

        Note that this involves communication with the external process, so it can
        be slow.

        Args:
            name (str): Attribute to access.

        Returns:
            Value of the attribute.
        """
        assert self._conn, "Run ProcessEnvironment.start() first"
        if self._fast:
            self._penv.call()
        self._conn.send((_MessageType.ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.

        Args:
            name (str): Name of the method to call.
            *args: Positional arguments to forward to the method.
            **kwargs: Keyword arguments to forward to the method.

        Returns:
            Promise object that blocks and provides the return value when called.
        """
        assert self._conn, "Run ProcessEnvironment.start() first"
        if self._fast:
            self._penv.call()
        payload = name, args, kwargs
        self._conn.send((_MessageType.CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            if self._fast:
                self._penv.close()
            else:
                self._conn.send((_MessageType.CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join()

    def step(self, action, blocking=True):
        """Step the environment.

        Args:
            action (nested tensors): The action to apply to the environment.
            blocking (bool): Whether to wait for the result.

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
            blocking (bool): Whether to wait for the result.

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
        assert self._conn, "Run ProcessEnvironment.start() first"
        message, payload = self._conn.recv()

        # Re-raise exceptions in the main process.
        if message == _MessageType.EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        elif message == _MessageType.RESULT:
            return payload
        elif message == _MessageType.CLOSE:
            # When notified that the child process is going to shut down, do not
            # panic and handle it quietly.
            return None
        self.close()
        raise KeyError(
            'Received message of unexpected type {}'.format(message))

    def render(self, mode='human'):
        """Render the environment.

        Args:
            mode (str): One of ['rgb_array', 'human']. Renders to an numpy array, or brings
                up a window where the environment can be visualized.
        Returns:
            An ndarray of shape [width, height, 3] denoting an RGB image if mode is
            `rgb_array`. Otherwise return nothing and render directly to a display
            window.
        Raises:
            NotImplementedError: If the environment does not support rendering.
        """
        return self.call('render', mode)()
