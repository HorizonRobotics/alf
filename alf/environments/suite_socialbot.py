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
except ImportError:
    social_bot = None

import gym
from tf_agents.environments.suite_gym import wrap_env
import gin.tf


def is_available():
    return social_bot is not None


@gin.configurable
def load(environment_name,
         port=11345,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         env_wrappers=(),
         spec_dtype_map=None):
    gym_spec = gym.spec(environment_name)
    gym_env = gym_spec.make(port=port)

    if max_episode_steps is None and gym_spec.timestep_limit is not None:
        max_episode_steps = gym_spec.max_episode_steps

    return wrap_env(
        gym_env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        env_wrappers=env_wrappers,
        spec_dtype_map=spec_dtype_map)
