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
"""Wrap ``dm_control`` environment with a Gym interface.
Adapted and simplified from https://github.com/denisyarats/dmc2gym
"""

from functools import partial

import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
from typing import Dict, Optional, Any

try:
    import dm_control
    from dm_control import suite
    import dm_env
except ImportError:
    dm_control = None


def _dmc_spec_to_box(spec):
    """Convert a dmc spec to a Gym Box.
    Copied from https://github.com/denisyarats/dmc2gym
    """

    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == dm_env.specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == dm_env.specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    """Flatten a dict to a vector.
    Copied from https://github.com/denisyarats/dmc2gym
    """
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCGYMWrapper(gym.core.Env):
    def __init__(self,
                 domain_name: str,
                 task_name: str,
                 visualize_reward: bool = True,
                 from_pixels: bool = False,
                 height: int = 84,
                 width: int = 84,
                 camera_id: int = 0,
                 control_timestep: Optional[float] = None):
        """A Gym env that wraps a ``dm_control`` environment.

        Args:
            domain_name: the domain name corresponds to the physical robot
            task_name: a specific task under a domain, which corresponds to a
                particular MDP structure
            visualize_reward: if True, then the rendered frame will have
                a highlighted color when the agent achieves a reward.
            from_pixels: if True, the observation will be raw pixels; otherwise
                use the interval state vector as the observation.
            height: image observation height
            width: image observation width
            camera_id: which camera to render; a MuJoCo xml file can define
                multiple cameras with different views
            control_timestep: the time duration between two agent actions. If
                this is greater than the agent's primitive physics timestep, then
                multiple physics simulation steps might be performed between two
                actions. If None, the default control timstep defined by DM control
                suite will be used.
        """
        self.metadata.update({'render.modes': ["rgb_array"]})

        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id

        if control_timestep is not None:
            environment_kwargs = {"control_timestep": control_timestep}
        else:
            environment_kwargs = None

        # create task
        self._env_fn = partial(
            suite.load,
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"time_limit": float('inf')},
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward)
        self._env = self._env_fn()

        self._action_space = _dmc_spec_to_box([self._env.action_spec()])

        # create observation space
        if from_pixels:
            shape = [3, height, width]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8)
        else:
            self._observation_space = _dmc_spec_to_box(
                self._env.observation_spec().values())

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            # this returns channels_last images
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id)
            obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed):
        self._action_space.seed(seed)
        self._observation_space.seed(seed)
        # Because dm_control seems not to provide an API for
        # seeding after an env is created, here we need to re-create
        # an env again.
        self._env = self._env_fn(task_kwargs={
            'random': seed,
            "time_limit": float('inf')
        })

    def step(self, action):
        assert self._action_space.contains(action)
        time_step = self._env.step(action)
        reward = time_step.reward or 0
        obs = self._get_obs(time_step)
        return obs, reward, False, {}

    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        """Render an RGB image.
        Copied from https://github.com/denisyarats/dmc2gym
        """
        assert mode == 'rgb_array', (
            'only support rgb_array mode, given %s' % mode)
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id)
