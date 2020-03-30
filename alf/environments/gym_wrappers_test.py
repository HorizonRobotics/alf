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

import gym
from gym import spaces
import numpy as np

import alf
from alf.environments.gym_wrappers import FrameStack


# FakeEnvironments adapted from gym/gym/wrappers/test_pixel_observation.py
class FakeEnvironment(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(
            shape=(1, ), low=-1, high=1, dtype=np.float32)

    def render(self, width=32, height=32, *args, **kwargs):
        del args
        del kwargs
        image_shape = (height, width, 3)
        return np.zeros(image_shape, dtype=np.uint8)

    def reset(self):
        observation = self.observation_space.sample()
        return observation

    def step(self, action):
        del action
        observation = self.observation_space.sample()
        reward, terminal, info = 0.0, False, {}
        return observation, reward, terminal, info


class FakeDictObservationEnvironment(FakeEnvironment):
    def __init__(self, *args, **kwargs):
        self.observation_space = spaces.Dict({
            'image':
                spaces.Box(shape=(2, 2, 3), low=-1, high=1, dtype=np.float32),
            'states':
                spaces.Box(shape=(4, ), low=-1, high=1, dtype=np.float32),
            'language':
                spaces.MultiDiscrete(nvec=[20, 20, 20]),  # dim == 3
            'dict':
                spaces.Dict({
                    'inner_states':
                        spaces.Box(
                            shape=(7, ), low=-1, high=1, dtype=np.float32),
                })
        })
        super().__init__(*args, **kwargs)


class FrameStackTest(alf.test.TestCase):
    def _create_env(self, stack_fields):
        return FrameStack(
            env=FakeDictObservationEnvironment(), fields=stack_fields)

    def test_framestack_all_fields(self):
        env = self._create_env(
            ['image', 'states', 'language', 'dict.inner_states'])
        obs = env.reset()
        all_shapes = (
            obs['image'].shape,
            obs['states'].shape,
            obs['language'].shape,
            obs['dict']['inner_states'].shape,
        )
        expected = (
            (2, 2, 3 * 4),  # 3 channels * 4
            (4 * 4, ),
            (3 * 4, ),
            (7 * 4, ),
        )
        assert all_shapes == expected, "Result " + str(
            all_shapes) + " doesn't match exptected " + str(expected)

    def test_framestack_partial_fields(self):
        env = self._create_env(['image', 'dict.inner_states'])
        obs = env.reset()
        all_shapes = (
            obs['image'].shape,
            obs['states'].shape,
            obs['language'].shape,
            obs['dict']['inner_states'].shape,
        )
        expected = (
            (2, 2, 3 * 4),  # 3 channels * 4
            (4, ),  # not stacking
            (3, ),  # not stacking
            (7 * 4, ),  # stacking nested field referred by path
        )
        assert all_shapes == expected, "Result " + str(
            all_shapes) + " doesn't match exptected " + str(expected)


if __name__ == '__main__':
    alf.test.main()
