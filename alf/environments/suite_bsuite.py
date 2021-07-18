# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import alf
from alf.environments import gym_wrappers, alf_wrappers, alf_gym_wrapper
from alf.environments.suite_gym import wrap_env

import bsuite
from bsuite import sweep
from bsuite.utils import gym_wrapper
from gym import spaces
from dm_env import specs
import numpy as np
from typing import Any, Dict, Tuple


def is_available():
    return bsuite is not None


@alf.configurable
def load(environment_name=sweep.CARTPOLE_SWINGUP[0],
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         alf_env_wrappers=()):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used in wrap_env to limit episode 
    lengths to the default benchmarks defined by the registered environments.

    Args:
        environment_name (str): Name for the environment to load.
        env_id (int): (optional) ID of the environment.
        discount (float): Discount to use for the environment.
        max_episode_steps (int): If None the max_episode_steps will be set to zero as not
            all bsuite environments specify max episode lengths. No limit is applied if set 
            to 0.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers
            classes to use directly on the gym environment.
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.

    Returns:
        An AlfEnvironment instance.
    """

    env = bsuite.load_from_id(environment_name)
    gym_env = BSuiteWrapper(env)

    if hasattr(env, '_max_steps'):
        if max_episode_steps is None:
            max_episode_steps = env._max_steps - 1
    elif max_episode_steps is None:
        max_episode_steps = 0

    return wrap_env(
        gym_env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        image_channel_first=False)


class BSuiteWrapper(gym_wrapper.GymFromDMEnv):
    """A wrapper for Bsuite environment.

    The BSuite environment is introduced in
    `Osband et al. Behaviour Suite for Reinforcement Learning  <https://openreview.net/forum?id=rygf-kSYwH>`_.

    It can be accessed on https://github.com/deepmind/bsuite
    """

    _GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]

    def __init__(self, env):
        """
        Args:
            gym_env (gym.Env): An instance of OpenAI gym environment.
        """
        super(BSuiteWrapper, self).__init__(env)

    @property
    def observation_space(self) -> spaces.Box:
        obs_spec = self._env.observation_spec()  # type: specs.Array
        obs_spec = specs.Array(
            shape=(obs_spec.shape[1], ), dtype=np.float32, name='state')
        if isinstance(obs_spec, specs.BoundedArray):
            return spaces.Box(
                low=float(obs_spec.minimum),
                high=float(obs_spec.maximum),
                shape=obs_spec.shape,
                dtype=obs_spec.dtype)
        return spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=obs_spec.shape,
            dtype=obs_spec.dtype)

    def step(self, action: int) -> _GymTimestep:
        timestep = self._env.step(action)
        self._last_observation = timestep.observation
        reward = timestep.reward or 0.
        if timestep.last():
            self.game_over = True
        return np.reshape(
            timestep.observation,
            (timestep.observation.shape[1], )), reward, timestep.last(), {}

    def reset(self) -> np.ndarray:
        self.game_over = False
        timestep = self._env.reset()
        self._last_observation = timestep.observation
        return np.reshape(timestep.observation,
                          (timestep.observation.shape[1], ))
