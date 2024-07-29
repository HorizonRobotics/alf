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

from absl.testing import parameterized
import gym
from gym import spaces
import numpy as np

import alf
from alf.environments.gym_wrappers import (FrameStack, FrameResize, FrameCrop,
                                           ContinuousActionClip,
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
            all_shapes) + " doesn't match expected " + str(expected)

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
            all_shapes) + " doesn't match expected " + str(expected)


class FrameResizeTest(parameterized.TestCase, alf.test.TestCase):
    def _create_env(self, width, height):
        return FrameResize(
            env=FakeDictObservationEnvironment(),
            width=width,
            height=height,
            fields=["image"])

    @parameterized.parameters((10, 11), (12, 11))
    def test_frame_resize(self, width, height):
        env = self._create_env(width=width, height=height)
        obs = env.reset()

        all_shapes = (
            obs['image'].shape,
            obs['states'].shape,
            obs['language'].shape,
            obs['dict']['inner_states'].shape,
        )
        expected = (
            (height, width, 3),
            (4, ),
            (3, ),
            (7, ),
        )
        assert all_shapes == expected, "Result " + str(
            all_shapes) + " doesn't match expected " + str(expected)

        # test observation space
        observation_space = env.observation_space
        all_shapes_from_obs_space = (
            observation_space['image'].shape,
            observation_space['states'].shape,
            observation_space['language'].shape,
            observation_space['dict']['inner_states'].shape)
        assert all_shapes_from_obs_space == expected, (
            "Observation space " + str(all_shapes_from_obs_space) +
            " doesn't match expected " + str(expected))


class FrameCropTest(parameterized.TestCase, alf.test.TestCase):
    def _create_env(self, sx, sy, width, height, channel_order):
        return FrameCrop(
            env=FakeDictObservationEnvironment(),
            sx=sx,
            sy=sy,
            width=width,
            height=height,
            channel_order=channel_order,
            fields=["image"])

    @parameterized.parameters(
        (0, 0, 1, 1, 'channels_last'), (0, 0, 1, 2, 'channels_last'),
        (0, 0, 2, 1, 'channels_last'), (1, 0, 1, 1, 'channels_last'),
        (0, 1, 1, 1, 'channels_last'), (0, 0, 2, 2, 'channels_last'),
        (0, 0, 1, 1, 'channels_first'), (0, 0, 1, 2, 'channels_first'),
        (0, 0, 2, 1, 'channels_first'), (1, 0, 1, 1, 'channels_first'),
        (0, 1, 1, 1, 'channels_first'), (0, 0, 2, 2, 'channels_first'))
    def test_frame_crop(self, sx, sy, width, height, channel_order):
        env = self._create_env(
            sx=sx,
            sy=sy,
            width=width,
            height=height,
            channel_order=channel_order)
        obs = env.reset()
        all_shapes = (
            obs['image'].shape,
            obs['states'].shape,
            obs['language'].shape,
            obs['dict']['inner_states'].shape,
        )
        expected = (
            (height, width, 3) if channel_order == 'channels_last' else
            (2, height, width),
            (4, ),
            (3, ),
            (7, ),
        )
        assert all_shapes == expected, "Result " + str(
            all_shapes) + " doesn't match expected " + str(expected)

        # test observation space
        observation_space = env.observation_space
        all_shapes_from_obs_space = (
            observation_space['image'].shape,
            observation_space['states'].shape,
            observation_space['language'].shape,
            observation_space['dict']['inner_states'].shape)
        assert all_shapes_from_obs_space == expected, (
            "Observation space " + str(all_shapes_from_obs_space) +
            " doesn't match expected " + str(expected))


class ActionWrappersTest(alf.test.TestCase):
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
