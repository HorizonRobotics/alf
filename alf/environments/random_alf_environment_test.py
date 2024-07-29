# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Test cases adapted from tf_agents' random_py_environment_test.py."""

import alf
from absl.testing import parameterized
import torch
import numpy as np

from alf.environments.random_alf_environment import RandomAlfEnvironment
from alf.tensor_specs import BoundedTensorSpec


class RandomAlfEnvironmentTest(parameterized.TestCase, alf.test.TestCase):
    def testEnvResetAutomatically(self):
        obs_spec = BoundedTensorSpec((2, 3), torch.int32, -10, 10)
        action_spec = BoundedTensorSpec([], torch.int32)
        env = RandomAlfEnvironment(obs_spec, action_spec)

        action = np.array(0, dtype=np.int64)
        time_step = env.step(action)
        self.assertTrue(np.all(time_step.observation >= -10))
        self.assertTrue(np.all(time_step.observation <= 10))
        self.assertTrue(time_step.is_first())

        while not time_step.is_last():
            time_step = env.step(action)
            self.assertTrue(np.all(time_step.observation >= -10))
            self.assertTrue(np.all(time_step.observation <= 10))

        time_step = env.step(action)
        self.assertTrue(np.all(time_step.observation >= -10))
        self.assertTrue(np.all(time_step.observation <= 10))
        self.assertTrue(time_step.is_first())

    @parameterized.named_parameters([
        ('OneStep', 1),
        ('FiveSteps', 5),
    ])
    def testEnvMinDuration(self, min_duration):
        obs_spec = BoundedTensorSpec((2, 3), torch.int32, -10, 10)
        action_spec = BoundedTensorSpec([], torch.int32)
        env = RandomAlfEnvironment(
            obs_spec,
            action_spec,
            episode_end_probability=0.9,
            min_duration=min_duration)
        num_episodes = 100

        action = np.array(0, dtype=np.int64)
        for _ in range(num_episodes):
            time_step = env.step(action)
            self.assertTrue(time_step.is_first())
            num_steps = 0
            while not time_step.is_last():
                time_step = env.step(action)
                num_steps += 1
            self.assertGreaterEqual(num_steps, min_duration)

    @parameterized.named_parameters([
        ('OneStep', 1),
        ('FiveSteps', 5),
    ])
    def testEnvMaxDuration(self, max_duration):
        obs_spec = BoundedTensorSpec((2, 3), torch.int32, -10, 10)
        action_spec = BoundedTensorSpec([], torch.int32)
        env = RandomAlfEnvironment(
            obs_spec,
            action_spec,
            episode_end_probability=0.1,
            max_duration=max_duration)
        num_episodes = 100

        action = np.array(0, dtype=np.int64)
        for _ in range(num_episodes):
            time_step = env.step(action)
            self.assertTrue(time_step.is_first())
            num_steps = 0
            while not time_step.is_last():
                time_step = env.step(action)
                num_steps += 1
            self.assertLessEqual(num_steps, max_duration)

    def testRewardFnCalled(self):
        def reward_fn(unused_step_type, action, unused_observation):
            return action

        action_spec = BoundedTensorSpec((1, ), torch.int64, -10, 10)
        observation_spec = BoundedTensorSpec((1, ), torch.int32, -10, 10)
        env = RandomAlfEnvironment(
            observation_spec, action_spec, reward_fn=reward_fn)

        action = np.array(1, dtype=np.int64)
        time_step = env.step(action)  # No reward in first time_step
        self.assertEqual(np.zeros((), dtype=np.float32), time_step.reward)
        time_step = env.step(action)
        self.assertEqual(np.ones((), dtype=np.float32), time_step.reward)

    def testRendersImage(self):
        action_spec = BoundedTensorSpec((1, ), torch.int64, -10, 10)
        observation_spec = BoundedTensorSpec((1, ), torch.int32, -10, 10)
        env = RandomAlfEnvironment(
            observation_spec, action_spec, render_size=(4, 4, 3))

        env.reset()
        img = env.render()

        self.assertTrue(np.all(img < 256))
        self.assertTrue(np.all(img >= 0))
        self.assertEqual((4, 4, 3), img.shape)
        self.assertEqual(np.uint8, img.dtype)

    def testBatchSize(self):
        batch_size = 3
        obs_spec = BoundedTensorSpec((2, 3), torch.int32, -10, 10)
        action_spec = BoundedTensorSpec((1, ), torch.int64)
        env = RandomAlfEnvironment(
            obs_spec, action_spec, batch_size=batch_size)
        time_step = env.step(np.array(0, dtype=np.int64))
        self.assertEqual(time_step.observation.shape, (3, 2, 3))
        self.assertEqual(time_step.reward.shape[0], batch_size)
        self.assertEqual(time_step.discount.shape[0], batch_size)

    def testCustomRewardFn(self):
        obs_spec = BoundedTensorSpec((2, 3), torch.int32, -10, 10)
        action_spec = BoundedTensorSpec((1, ), torch.int64)
        batch_size = 3
        env = RandomAlfEnvironment(
            obs_spec,
            action_spec,
            reward_fn=lambda *_: np.ones(batch_size),
            batch_size=batch_size)
        env._done = False
        env.reset()
        action = np.ones(batch_size, dtype=np.int64)
        time_step = env.step(action)
        self.assertSequenceAlmostEqual([1.0] * 3, time_step.reward)

    def testRewardCheckerBatchSizeOne(self):
        # Ensure batch size 1 with scalar reward works
        obs_spec = BoundedTensorSpec((2, 3), torch.int32, -10, 10)
        action_spec = BoundedTensorSpec((1, ), torch.int64)
        env = RandomAlfEnvironment(
            obs_spec,
            action_spec,
            reward_fn=lambda *_: np.array([1.0]),
            batch_size=1)
        env._done = False
        env.reset()
        action = np.array([0], dtype=np.int64)
        time_step = env.step(action)
        self.assertEqual(time_step.reward, 1.0)

    def testRewardCheckerSizeMismatch(self):
        # Ensure custom scalar reward with batch_size greater than 1 raises
        # ValueError
        obs_spec = BoundedTensorSpec((2, 3), torch.int32, -10, 10)
        action_spec = BoundedTensorSpec((1, ), torch.int64)
        env = RandomAlfEnvironment(
            obs_spec,
            action_spec,
            reward_fn=lambda *_: np.array([1.0]),
            batch_size=5)
        env.reset()
        env._done = False
        action = np.array(0, dtype=np.int64)
        with self.assertRaises(ValueError):
            env.step(action)


if __name__ == '__main__':
    alf.test.main()
