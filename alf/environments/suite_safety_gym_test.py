# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import functools
import torch

import alf
from alf.environments import suite_safety_gym, alf_environment
from alf.environments import thread_environment, parallel_environment
import alf.nest as nest


class SuiteSafetyGymTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_safety_gym.is_available():
            self.skipTest('suite_safety_gym is not available.')

    def tearDown(self):
        super().tearDown()
        self._env.close()

    def test_unwrapped_env(self):
        self._env = suite_safety_gym.load(
            environment_name='Safexp-PointGoal1-v0')

        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self.assertEqual(torch.float32, self._env.observation_spec().dtype)
        self.assertEqual((suite_safety_gym.VectorReward.REWARD_DIMENSION, ),
                         self._env.reward_spec().shape)

        actions = self._env.action_spec().sample()
        for _ in range(10):
            # unwrapped env (not thread_env or parallel_env) needs to convert
            # from tensor to array
            time_step = self._env.step(actions.cpu().numpy())

    def test_thread_env(self):
        self._env = thread_environment.ThreadEnvironment(
            lambda: suite_safety_gym.load(environment_name=
                                          'Safexp-PointGoal1-v0'))
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self.assertEqual(torch.float32, self._env.observation_spec().dtype)
        self.assertEqual((suite_safety_gym.VectorReward.REWARD_DIMENSION, ),
                         self._env.reward_spec().shape)

        actions = self._env.action_spec().sample()
        for _ in range(10):
            time_step = self._env.step(actions)

    def test_parallel_env(self):
        env_num = 8

        def ctor(env_id=None):
            return suite_safety_gym.load(
                environment_name='Safexp-PointGoal1-v0')

        constructor = functools.partial(ctor)

        self._env = parallel_environment.ParallelAlfEnvironment(
            [constructor] * env_num)
        self.assertTrue(self._env.batched)
        self.assertEqual(self._env.batch_size, env_num)
        self.assertEqual(torch.float32, self._env.observation_spec().dtype)
        self.assertEqual((suite_safety_gym.VectorReward.REWARD_DIMENSION, ),
                         self._env.reward_spec().shape)

        actions = self._env.action_spec().sample(outer_dims=(env_num, ))
        for _ in range(10):
            time_step = self._env.step(actions)

    def test_env_info(self):
        # test creating multiple envs in the same process
        l0_env = suite_safety_gym.load(environment_name="Safexp-PointGoal0-v0")
        l1_env = suite_safety_gym.load(environment_name='Safexp-PointGoal1-v0')

        # level 0 envs don't have costs
        actions = l0_env.action_spec().sample()
        time_step = l0_env.step(actions.cpu().numpy())
        # ['cost_exception', 'goal_met', 'cost']
        self.assertEqual(len(time_step.env_info.keys()), 3)
        self.assertFalse('cost_hazards' in time_step.env_info.keys())
        l0_env.close()

        actions = l1_env.action_spec().sample()
        time_step = l1_env.step(actions.cpu().numpy())
        self.assertGreater(len(time_step.env_info.keys()), 3)
        self.assertTrue('cost_hazards' in time_step.env_info.keys())
        self._env = l1_env


if __name__ == '__main__':
    alf.test.main()
