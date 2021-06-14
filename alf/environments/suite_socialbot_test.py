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
"""Test for alf.environments.suite_socialbot."""

import functools
import torch

import alf
from alf.environments import suite_socialbot, alf_environment
from alf.environments import thread_environment, parallel_environment


class SuiteSocialbotTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_socialbot.is_available():
            self.skipTest('suite_socialbot is not available.')

    def tearDown(self):
        super().tearDown()
        self._env.close()

    def test_socialbot_env_registered(self):
        self._env = suite_socialbot.load(
            'SocialBot-CartPole-v0', wrap_with_process=True)
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)

    def test_observation_spec(self):
        self._env = suite_socialbot.load(
            'SocialBot-CartPole-v0', wrap_with_process=True)
        self.assertEqual(torch.float32, self._env.observation_spec().dtype)
        self.assertEqual((4, ), self._env.observation_spec().shape)

    def test_action_spec(self):
        self._env = suite_socialbot.load(
            'SocialBot-CartPole-v0', wrap_with_process=True)
        self.assertEqual(torch.float32, self._env.action_spec().dtype)
        self.assertEqual((1, ), self._env.action_spec().shape)

    def test_thread_env(self):
        env_name = 'SocialBot-CartPole-v0'
        self._env = thread_environment.ThreadEnvironment(
            lambda: suite_socialbot.load(
                environment_name=env_name, wrap_with_process=False))
        self.assertEqual(torch.float32, self._env.observation_spec().dtype)
        self.assertEqual((4, ), self._env.observation_spec().shape)
        self.assertEqual(torch.float32, self._env.action_spec().dtype)
        self.assertEqual((1, ), self._env.action_spec().shape)

        actions = self._env.action_spec().sample()
        for _ in range(10):
            time_step = self._env.step(actions)

    def test_parallel_envs(self):
        env_num = 5
        env_name = 'SocialBot-CartPole-v0'

        def ctor(env_name, env_id=None):
            return suite_socialbot.load(
                environment_name=env_name, wrap_with_process=False)

        constructor = functools.partial(ctor, env_name)

        self._env = parallel_environment.ParallelAlfEnvironment(
            [constructor] * env_num, start_serially=False)

        self.assertTrue(self._env.batched)
        self.assertEqual(self._env.batch_size, env_num)

        actions = self._env.action_spec().sample(outer_dims=(env_num, ))
        for _ in range(10):
            time_step = self._env.step(actions)


if __name__ == '__main__':
    alf.test.main()
