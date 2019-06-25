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

from absl import logging
from tf_agents.environments import suite_gym, parallel_py_environment, tf_py_environment

from alf.environments import suite_socialbot


@gin.configurable
def create_environment(env_name='CartPole-v0',
                       env_load_fn=suite_gym.load,
                       num_parallel_environments=30):
    """Create environment.

    Args:
        env_name (str): env name
        env_load_fn (Callable) : callable that create an environment
        num_parallel_environments (int): num of parallel environments
    """
    if num_parallel_environments == 1:
        py_env = env_load_fn(env_name)
    else:
        if env_load_fn == suite_socialbot.load:
            logging.info("suite_socialbot environment")
            # No need to wrap with process since ParllelPyEnvironment will do it
            env_load_fn = lambda env_name: suite_socialbot.load(
                env_name, wrap_with_process=False)
        py_env = parallel_py_environment.ParallelPyEnvironment(
            [lambda: env_load_fn(env_name)] * num_parallel_environments)
    return tf_py_environment.TFPyEnvironment(py_env)


@gin.configurable
def load_with_random_max_episode_steps(env_name,
                                       env_load_fn=suite_gym.load,
                                       min_steps=200,
                                       max_steps=250):
    return suite_gym.load(
        env_name, max_episode_steps=random.randint(min_steps, max_steps))
