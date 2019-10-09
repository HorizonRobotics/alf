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
import random

import gin.tf

from tf_agents.environments import suite_gym, parallel_py_environment, tf_py_environment

from alf.environments import suite_mario
from alf.environments import suite_socialbot
from alf.environments import suite_dmlab

parallel_py_environment.ProcessPyEnvironment = suite_socialbot.ProcessPyEnvironment


@gin.configurable
def create_environment(env_name='CartPole-v0',
                       env_load_fn=suite_gym.load,
                       num_parallel_environments=30,
                       force_unwrapped=False):
    """Create environment.

    Args:
        env_name (str): env name
        env_load_fn (Callable) : callable that create an environment
        num_parallel_environments (int): num of parallel environments
        force_unwrapped (bool): force to create a single env in the current
            process; use only when you are certain that multiple envs in the
            current process can co-exist.
    Returns:
        TFPyEnvironment
    """
    wrappable = (env_load_fn in (suite_socialbot.load, suite_mario.load,
                                 suite_dmlab.load))
    xarg = dict()
    if force_unwrapped or num_parallel_environments == 1:
        if wrappable:
            xarg = dict(wrap_with_process=not force_unwrapped)
        py_env = env_load_fn(env_name, **xarg)
    else:
        py_env = parallel_py_environment.ParallelPyEnvironment(
            [lambda: env_load_fn(env_name, **xarg)
             ] * num_parallel_environments)
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
