# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

try:
    import social_bot
    # The following import is to allow gin config of environments take effects
    import social_bot.envs
except ImportError:
    social_bot = None

import sys
import traceback
import contextlib
import socket
import gym
from fasteners.process_lock import InterProcessLock
from tf_agents.environments import suite_gym, wrappers, parallel_py_environment
import gin.tf
import tensorflow as tf
from absl import logging

from alf.environments.utils import UnwrappedEnvChecker

DEFAULT_SOCIALBOT_PORT = 11345

_unwrapped_env_checker_ = UnwrappedEnvChecker()


def is_available():
    return social_bot is not None


class ProcessPyEnvironment(parallel_py_environment.ProcessPyEnvironment):
    """tf_agents ProcessPyEnvironment with render()."""

    def __init__(self, env_constructor, flatten=False):
        super(ProcessPyEnvironment, self).__init__(
            env_constructor, flatten=flatten)

    def _worker(self, conn, env_constructor, flatten=False):
        """It's a little different with `super()._worker`, it closes environment when
        receives _CLOSE.

        Args:
            conn (Pipe): Connection for communication to the main process.
            env_constructor (Callable): env_constructor for the OpenAI Gym environment.
            flatten (bool): whether to assume flattened actions and time_steps
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
                        args = [tf.nest.pack_sequence_as(action_spec, args[0])]
                    result = getattr(env, name)(*args, **kwargs)
                    if flatten and name in ['step', 'reset']:
                        result = tf.nest.flatten(result)
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


@gin.configurable
def load(environment_name,
         port=None,
         wrap_with_process=False,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         env_wrappers=(),
         spec_dtype_map=None):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name: Name for the environment to load.
        port: Port used for the environment
        wrap_with_process: Whether wrap environment in a new process
        discount: Discount to use for the environment.
        max_episode_steps: If None the max_episode_steps will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no timestep_limit set in the environment's spec.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        env_wrappers: Iterable with references to wrapper classes to use on the
            gym_wrapped environment.
        spec_dtype_map: A dict that maps gym specs to tf dtypes to use as the
            default dtype for the tensors. An easy way how to configure a custom
            mapping through Gin is to define a gin-configurable function that returns
            desired mapping and call it in your Gin config file, for example:
            `suite_socialbot.load.spec_dtype_map = @get_custom_mapping()`.

    Returns:
        A PyEnvironmentBase instance.
    """
    global _unwrapped_env_checker_
    _unwrapped_env_checker_.check_and_update(wrap_with_process)

    gym_spec = gym.spec(environment_name)
    if max_episode_steps is None:
        if gym_spec.timestep_limit is not None:
            max_episode_steps = gym_spec.max_episode_steps
        else:
            max_episode_steps = 0

    def env_ctor(port):
        gym_env = gym_spec.make(port=port)
        return suite_gym.wrap_env(
            gym_env,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,
            env_wrappers=env_wrappers,
            spec_dtype_map=spec_dtype_map)

    port_range = [port, port + 1] if port else [DEFAULT_SOCIALBOT_PORT]
    with _get_unused_port(*port_range) as port:
        if wrap_with_process:
            process_env = ProcessPyEnvironment(lambda: env_ctor(port))
            process_env.start()
            py_env = wrappers.PyEnvironmentBaseWrapper(process_env)
        else:
            py_env = env_ctor(port)
    return py_env


@contextlib.contextmanager
def _get_unused_port(start, end=65536):
    """Get an unused port in the range [start, end) .

    Args:
        start (int) : port range start
        end (int): port range end
    """
    process_lock = None
    unused_port = None
    try:
        for port in range(start, end):
            process_lock = InterProcessLock(
                path='/tmp/socialbot/{}.lock'.format(port))
            if not process_lock.acquire(blocking=False):
                process_lock = None
                continue
            try:
                with contextlib.closing(socket.socket()) as sock:
                    sock.bind(('', port))
                    unused_port = port
                    break
            except socket.error:
                process_lock.release()
                process_lock = None
        if unused_port is None:
            raise socket.error("No unused port in [{}, {})".format(start, end))
        yield unused_port
    finally:
        if process_lock is not None:
            process_lock.release()
