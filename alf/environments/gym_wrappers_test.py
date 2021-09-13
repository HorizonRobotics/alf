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
from absl.testing import parameterized

import alf
from alf.environments.gym_wrappers import (FrameStack, ContinuousActionClip,
                                           ContinuousActionMapping)


# FakeEnvironments adapted from gym/gym/wrappers/test_pixel_observation.py
class FakeEnvironment(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(
            shape=(1, ), low=-2, high=3, dtype=np.float32)
        self.observation_space = spaces.Box(
            shape=(10, ), low=-1, high=1, dtype=np.float32)
        self.action = None

    def render(self, width=32, height=32, *args, **kwargs):
        image_shape = (height, width, 3)
        return np.zeros(image_shape, dtype=np.uint8)

    def reset(self):
        observation = self.observation_space.sample()
        return observation

    def step(self, action):
        self.action = action
        observation = self.observation_space.sample()
        reward, terminal, info = 0.0, False, {}
        return observation, reward, terminal, info


class FakeDictObservationEnvironment(FakeEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


class ActionWrappersTest(parameterized.TestCase, alf.test.TestCase):
    def test_continuous_action_clipping(self):
        env = ContinuousActionClip(FakeEnvironment())
        action = env.action_space.high + 0.1
        env.step(action)
        self.assertTrue(np.all(env.unwrapped.action == env.action_space.high))

        action = env.action_space.low - 0.1
        env.step(action)
        self.assertTrue(np.all(env.unwrapped.action == env.action_space.low))

    def test_continuous_action_mapping(self):
        unwrapped = FakeEnvironment()
        env = ContinuousActionClip(
            ContinuousActionMapping(unwrapped, low=-1., high=2.))
        action = np.array([-1.1])
        env.step(action)
        self.assertTrue(np.all(unwrapped.action == unwrapped.action_space.low))

        action = np.array([2.1])
        env.step(action)
        self.assertTrue(
            np.all(unwrapped.action == unwrapped.action_space.high))

        action = np.array([0.])  # from [-1, 2] to [-2, 3]
        env.step(action)
        self.assertAlmostEqual(float(unwrapped.action), -1. / 3)


if __name__ == '__main__':
    alf.test.main()
