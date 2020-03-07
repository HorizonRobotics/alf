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

import functools
import gin
import numpy as np
import random
import torch

from alf.environments import suite_gym
from alf.environments import thread_torch_environment, parallel_torch_environment
from alf.environments import torch_wrappers


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
                       nonparallel=False,
                       seed=None):
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

        # Create and step the env in a separate thread. env `step` and `reset` must
        #   run in the same thread which the env is created in for some simulation
        #   environments such as social_bot(gazebo)
        torch_env = thread_torch_environment.ThreadTorchEnvironment(
            lambda: env_load_fn(env_name))
        if seed is None:
            torch_env.seed(np.random.randint(0, np.iinfo(np.int32).max))
        else:
            torch_env.seed(seed)
    else:
        # flatten=True will use flattened action and time_step in
        #   process environments to reduce communication overhead.
        torch_env = parallel_torch_environment.ParallelTorchEnvironment(
            [functools.partial(env_load_fn, env_name)] *
            num_parallel_environments,
            flatten=True)

        if seed is None:
            torch_env.seed([
                np.random.randint(0,
                                  np.iinfo(np.int32).max)
                for i in range(num_parallel_environments)
            ])
        else:
            # We want deterministic behaviors for each environment, but different
            # behaviors among different individual environments (to increase the
            # diversity of environment data)!
            torch_env.seed(
                [seed + i for i in range(num_parallel_environments)])

    return torch_env


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
