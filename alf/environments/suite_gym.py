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
import gin
import gym
import gym.spaces

from alf.environments import gym_wrappers, torch_wrappers, torch_gym_wrapper


@gin.configurable
def load(environment_name,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         torch_env_wrappers=(),
         image_channel_first=True):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.
  
    Args:
        environment_name (str): Name for the environment to load.
        env_id (int): (optional) ID of the environment. 
        discount (float): Discount to use for the environment.
        max_episode_steps (int): If None the max_episode_steps will be set to the 
            default step limit defined in the environment's spec. No limit is applied
            if set to 0 or if there is no max_episode_steps set in the environment's
            spec.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers
            classes to use directly on the gym environment.
        torch_env_wrappers (Iterable): Iterable with references to torch_wrappers 
            classes to use on the torch environment.
        image_channel_first (bool): whether transpose image channels to first dimension. 
  
    Returns:
        A TorchEnvironment instance.
    """
    gym_spec = gym.spec(environment_name)
    gym_env = gym_spec.make()

    if max_episode_steps is None and gym_spec.max_episode_steps is not None:
        max_episode_steps = gym_spec.max_episode_steps

    return wrap_env(
        gym_env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        torch_env_wrappers=torch_env_wrappers,
        image_channel_first=image_channel_first)


@gin.configurable
def wrap_env(gym_env,
             env_id=None,
             discount=1.0,
             max_episode_steps=0,
             gym_env_wrappers=(),
             time_limit_wrapper=torch_wrappers.TimeLimit,
             torch_env_wrappers=(),
             image_channel_first=True,
             auto_reset=True):
    """Wraps given gym environment with TorchGymWrapper.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Also note that all gym wrappers assume images are 'channel_last' by default,
    while PyTorch only supports 'channel_first' image inputs. To enable this 
    transpose, 'image_channel_first' is set as True by default. There are two options 
    provided in ALF to handle this transpose: 
        1. Applying the gym_wrappers.ImageChannelFirst after all gym_env_wrappers 
            and before the TorchGymWrapper.
        2. Applying the torch_wrappers.ImageChannelFirst after all torch_gym_wrappers. 
    The first option is used in current function.
  
    Args:
        gym_env (gym.Env): An instance of OpenAI gym environment.
        env_id (int): (optional) ID of the environment.
        discount (float): Discount to use for the environment.
        max_episode_steps (int): Used to create a TimeLimitWrapper. No limit is applied
            if set to 0. Usually set to `gym_spec.max_episode_steps` as done in `load.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers, 
            classes to use directly on the gym environment.
        time_limit_wrapper (TorchEnvironmentBaseWrapper): Wrapper that accepts 
            (env, max_episode_steps) params to enforce a TimeLimit. Usuaully this 
            should be left as the default, torch_wrappers.TimeLimit.
        torch_env_wrappers (Iterable): Iterable with references to torch_wrappers 
            classes to use on the torch environment.
        image_channel_first (bool): whether transpose image channels to first dimension.
            PyTorch only supports channgel_first image inputs.
        auto_reset (bool): If True (default), reset the environment automatically after a
            terminal state is reached.
  
    Returns:
        A TorchEnvironment instance.
    """

    for wrapper in gym_env_wrappers:
        gym_env = wrapper(gym_env)

    # To apply channel_first transpose on gym (py) env
    if image_channel_first:
        gym_env = gym_wrappers.ImageChannelFirst(gym_env)

    env = torch_gym_wrapper.TorchGymWrapper(
        gym_env=gym_env,
        env_id=env_id,
        discount=discount,
        auto_reset=auto_reset,
    )

    if max_episode_steps > 0:
        env = time_limit_wrapper(env, max_episode_steps)

    for wrapper in torch_env_wrappers:
        env = wrapper(env)

    return env
