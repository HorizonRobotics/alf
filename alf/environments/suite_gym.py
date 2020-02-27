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

import collections
import gym
import gym.spaces
import gin
from alf.environments import torch_wrappers, torch_gym_wrapper


@gin.configurable
def load(environment_name,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         torch_env_wrappers=()):
    """Loads the selected environment and wraps it with the specified wrappers.

  Note that by default a TimeLimit wrapper is used to limit episode lengths
  to the default benchmarks defined by the registered environments.

  Args:
    environment_name: Name for the environment to load.
    discount: Discount to use for the environment.
    max_episode_steps: If None the max_episode_steps will be set to the default
      step limit defined in the environment's spec. No limit is applied if set
      to 0 or if there is no max_episode_steps set in the environment's spec.
    gym_env_wrappers: Iterable with references to wrapper classes to use
      directly on the gym environment.
    torch_env_wrappers: Iterable with references to wrapper classes to use on the
      torch environment.

  Returns:
    A PyEnvironment instance.
  """
    gym_spec = gym.spec(environment_name)
    gym_env = gym_spec.make()

    if max_episode_steps is None and gym_spec.max_episode_steps is not None:
        max_episode_steps = gym_spec.max_episode_steps

    return wrap_env(
        gym_env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        torch_env_wrappers=torch_env_wrappers)


@gin.configurable
def wrap_env(gym_env,
             discount=1.0,
             max_episode_steps=0,
             gym_env_wrappers=(),
             time_limit_wrapper=torch_wrappers.TimeLimit,
             torch_env_wrappers=(),
             auto_reset=True):
    """Wraps given gym environment with TF Agent's GymWrapper.

  Note that by default a TimeLimit wrapper is used to limit episode lengths
  to the default benchmarks defined by the registered environments.

  Args:
    gym_env: An instance of OpenAI gym environment.
    discount: Discount to use for the environment.
    max_episode_steps: Used to create a TimeLimitWrapper. No limit is applied
      if set to 0. Usually set to `gym_spec.max_episode_steps` as done in `load.
    gym_env_wrappers: Iterable with references to wrapper classes to use
      directly on the gym environment.
    time_limit_wrapper: Wrapper that accepts (env, max_episode_steps) params to
      enforce a TimeLimit. Usuaully this should be left as the default,
      torch_wrappers.TimeLimit.
    torch_env_wrappers: Iterable with references to wrapper classes to use on the
      torch environment.
    auto_reset: If True (default), reset the environment automatically after a
      terminal state is reached.

  Returns:
    A TorchEnvironment instance.
  """

    for wrapper in gym_env_wrappers:
        gym_env = wrapper(gym_env)

    env = torch_gym_wrapper.TorchGymWrapper(
        gym_env,
        discount=discount,
        auto_reset=auto_reset,
    )

    if max_episode_steps > 0:
        env = time_limit_wrapper(env, max_episode_steps)

    for wrapper in torch_env_wrappers:
        env = wrapper(env)

    return env
