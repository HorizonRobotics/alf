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
import numpy as np


@gin.configurable
class FrameStack(gym.Wrapper):
    """Stack previous `stack_size` frames, applied to Gym env."""

    def __init__(self, env, stack_size=4, channel_order='channels_last'):
        """Create a FrameStack object.

        Args:
            stack_size (int):
            channel_order (str): one of `channels_last` or `channels_first`.
                The ordering of the dimensions in the images.
                `channels_last` corresponds to images with shape
                `(height, width, channels)` while `channels_first` corresponds
                to images with shape `(channels, height, width)`.
        """
        super().__init__(env)
        self._frames = collections.deque(maxlen=stack_size)
        self._channel_order = channel_order
        self._stack_size = stack_size

        space = self.env.observation_space
        assert channel_order in ['channels_last', 'channels_first']

        if channel_order == 'channels_last':
            shape = list(space.shape[0:-1]) + [stack_size * space.shape[-1]]
        else:
            shape = [
                stack_size * space.shape[0],
            ] + list(space.shape[1:])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self.env, name)

    def _generate_observation(self):
        if self._channel_order == 'channels_last':
            return np.concatenate(self._frames, axis=-1)
        else:
            return np.concatenate(self._frames, axis=0)

    def reset(self):
        observation = self.env.reset()
        for _ in range(self._stack_size):
            self._frames.append(observation)
        return self._generate_observation()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._frames.append(observation)
        return self._generate_observation(), reward, done, info


@gin.configurable
class FrameSkip(gym.Wrapper):
    """
    Repeat same action n times and return the last observation
     and accumulated reward
    """

    def __init__(self, env, skip):
        """Create a FrameSkip object

        Args:
            skip (int): skip `skip` frames (skip=1 means no skip)
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        obs = None
        accumulated_reward = 0
        done = False
        info = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            accumulated_reward += reward
            if done: break
        return obs, accumulated_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
