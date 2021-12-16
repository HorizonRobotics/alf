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
"""Test for suite_dmc. Adapted from alf suite_gym_test. """

import functools

import alf
from alf.environments import suite_dmc
from alf.environments import alf_wrappers
from alf.environments.gym_wrappers import DMAtariPreprocessing, FrameStack
from alf.environments.alf_environment import AlfEnvironment


class SuiteDmcTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_dmc.is_available():
            self.skipTest('suite_dmc is not available.')

    def tearDown(self):
        super(SuiteDmcTest, self).tearDown()

    def test_load_adds_time_limit_steps(self):
        env = suite_dmc.load('fish:swim', from_pixels=False)
        self.assertIsInstance(env, AlfEnvironment)
        self.assertIsInstance(env, alf_wrappers.TimeLimit)

    def test_load_disable_step_limit(self):
        env = suite_dmc.load(
            'fish:swim', max_episode_steps=0, from_pixels=False)
        self.assertIsInstance(env, AlfEnvironment)
        self.assertNotIsInstance(env, alf_wrappers.TimeLimit)

    def test_load_disable_alf_wrappers_applied(self):
        duration_wrapper = functools.partial(
            alf_wrappers.TimeLimit, duration=10)
        env = suite_dmc.load(
            'fish:swim',
            max_episode_steps=0,
            alf_env_wrappers=(duration_wrapper, ),
            from_pixels=False)
        self.assertIsInstance(env, AlfEnvironment)
        self.assertIsInstance(env, alf_wrappers.TimeLimit)

    def test_custom_max_steps(self):
        env = suite_dmc.load(
            'fish:swim', max_episode_steps=5, from_pixels=False)
        self.assertIsInstance(env, AlfEnvironment)
        self.assertIsInstance(env, alf_wrappers.TimeLimit)
        self.assertEqual(5, env._duration)


if __name__ == '__main__':
    alf.test.main()
