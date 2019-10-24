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

# multiprocessing.dummy provides a pure *multithreaded* threadpool that works
# in both python2 and python3 (concurrent.futures isn't available in python2).
#   https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.dummy
from multiprocessing import dummy as mp_threads

import random
import sys
import traceback
import tensorflow as tf
import gin.tf
from absl import logging
import numpy as np

from alf.environments import suite_gym
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment


class ThreadPyEnvironment(py_environment.PyEnvironment):
    """Create, Step a single env in a separate thread
    """

    def __init__(self, env_constructor):
        """Create a ThreadPyEnvironment

        Args:
            env_constructor (Callable): env_constructor for the OpenAI Gym environment
        """
        super().__init__()
        self._pool = mp_threads.Pool(1)
        self._env = self._pool.apply(env_constructor)

    def observation_spec(self):
        return self._apply('observation_spec')

    def action_spec(self):
        return self._apply('action_spec')

    def _step(self, action):
        return self._apply('step', (action, ))

    def _reset(self):
        return self._apply('reset')

    def close(self):
        self._apply('close')
        self._pool.close()
        self._pool.join()

    def render(self, mode='rgb_array'):
        return self._apply('render', (mode, ))

    def seed(self, seed):
        self._apply('seed', (seed, ))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _apply(self, name, args=()):
        func = getattr(self._env, name)
        return self._pool.apply(func, args)


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


parallel_py_environment.ProcessPyEnvironment = ProcessPyEnvironment


class UnwrappedEnvChecker(object):
    """
    A class for checking if there is already an unwrapped env in the current
    process. For some games, if the check is True, then we should stop creating
    more envs (multiple envs cannot coexist in a process).

    See suite_socialbot.py for an example usage of this class.
    """

    def __init__(self):
        self._unwrapped_env_in_process = False

    def check(self):
        assert not self._unwrapped_env_in_process, \
            "You cannot create more envs once there has been an env in the main process!"

    def update(self, wrap_with_process):
        """
        Update the flag.

        Args:
            wrap_with_process (bool): if False, an env is being created without
                being wrapped by a subprocess.
        """
        self._unwrapped_env_in_process |= not wrap_with_process

    def check_and_update(self, wrap_with_process):
        """
        Combine self.check() and self.update()
        """
        self.check()
        self.update(wrap_with_process)


@gin.configurable
def create_environment(env_name='CartPole-v0',
                       env_load_fn=suite_gym.load,
                       num_parallel_environments=30,
                       nonparallel=False):
    """Create environment.

    Args:
        env_name (str): env name
        env_load_fn (Callable) : callable that create an environment
        num_parallel_environments (int): num of parallel environments
        nonparallel (bool): force to create a single env in the current
            process. Used for correctly exposing game gin confs to tensorboard.

    Returns:
        TFPyEnvironment
    """
    if nonparallel:
        # Each time we can only create one unwrapped env at most

        # Create and step the env in a separate thread, env `step` and `reset` must
        #   run in the same thread which the env created in for some simulation
        #   environments such as social_bot(gazebo)
        py_env = ThreadPyEnvironment(lambda: env_load_fn(env_name))
        py_env.seed(np.random.randint(0, np.iinfo(np.int32).max))
    else:
        py_env = parallel_py_environment.ParallelPyEnvironment(
            [lambda: env_load_fn(env_name)] * num_parallel_environments)

        py_env.seed([
            np.random.randint(0,
                              np.iinfo(np.int32).max)
            for i in range(num_parallel_environments)
        ])

    return tf_py_environment.TFPyEnvironment(py_env)


@gin.configurable
def load_with_random_max_episode_steps(env_name,
                                       env_load_fn=suite_gym.load,
                                       min_steps=200,
                                       max_steps=250):
    """Create environment with random max_episode_steps in range [min_steps, max_steps]

    Args:
        env_name (str): env name
        env_load_fn (Callable) : callable that create an environment
        min_steps (int): represent min value of the random range
        max_steps (int): represent max value of the random range
    Returns:
        TFPyEnvironment
    """
    return env_load_fn(
        env_name, max_episode_steps=random.randint(min_steps, max_steps))
