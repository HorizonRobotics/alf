# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for suite_gym. Adapted from tf_agent suite_gym_test. """

import functools

import alf
from alf.environments import suite_gym
from alf.environments import alf_wrappers
from alf.environments.gym_wrappers import DMAtariPreprocessing, FrameStack
from alf.environments.alf_environment import AlfEnvironment


class SuiteGymTest(alf.test.TestCase):
    def tearDown(self):
        super(SuiteGymTest, self).tearDown()

    def test_load_adds_time_limit_steps(self):
        env = suite_gym.load('CartPole-v1')
        self.assertIsInstance(env, AlfEnvironment)
        self.assertIsInstance(env, alf_wrappers.TimeLimit)

    def test_load_disable_step_limit(self):
        env = suite_gym.load('CartPole-v1', max_episode_steps=0)
        self.assertIsInstance(env, AlfEnvironment)
        self.assertNotIsInstance(env, alf_wrappers.TimeLimit)

    def test_load_disable_alf_wrappers_applied(self):
        duration_wrapper = functools.partial(
            alf_wrappers.TimeLimit, duration=10)
        env = suite_gym.load(
            'CartPole-v1',
            max_episode_steps=0,
            alf_env_wrappers=(duration_wrapper, ))
        self.assertIsInstance(env, AlfEnvironment)
        self.assertIsInstance(env, alf_wrappers.TimeLimit)

    def test_custom_max_steps(self):
        env = suite_gym.load('CartPole-v1', max_episode_steps=5)
        self.assertIsInstance(env, AlfEnvironment)
        self.assertIsInstance(env, alf_wrappers.TimeLimit)
        self.assertEqual(5, env._duration)

    def test_load_atari(self):
        env = suite_gym.load(
            'BreakoutNoFrameskip-v4',
            gym_env_wrappers=(DMAtariPreprocessing, FrameStack))
        self.assertIsInstance(env, AlfEnvironment)

    # # TODO: unittest does not have test_src_dir_path
    # def testGinConfig(self):
    #     gin.parse_config_file(
    #         unittest.test_src_dir_path('environments/configs/suite_gym.gin')
    #     )
    #     env = suite_gym.load()
    #     self.assertIsInstance(env, AlfEnvironment)
    #     self.assertIsInstance(env, alf_wrappers.TimeLimit)


if __name__ == '__main__':
    alf.test.main()
