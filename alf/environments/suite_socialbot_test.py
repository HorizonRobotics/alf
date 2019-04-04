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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np

from tf_agents.environments import py_environment
from alf.environments import suite_socialbot

import gin.tf


class SuiteSocialbotTest(absltest.TestCase):
    def setUP(self):
        super(SuiteSocialbotTest, self).setUp()
        if not suite_socialbot.is_available():
            self.skipTest('suite_socialbot is not available.')
        else:
            gin.clear_config()

    def testSocialbotEnvRegistered(self):
        env = suite_socialbot.load('SocialBot-Pr2Gripper-v0')
        self.assertIsInstance(env, py_environment.PyEnvironment)

    def testObservationSpec(self):
        env = suite_socialbot.load('SocialBot-Pr2Gripper-v0')
        self.assertEqual(np.float32, env.observation_spec().dtype)
        self.assertEqual((84, 84, 1), env.observation_spec().shape)

    def testActionSpec(self):
        env = suite_socialbot.load('SocialBot-Pr2Gripper-v0')
        self.assertEqual(np.float32, env.action_spec().dtype)
        self.assertEqual((20,), env.action_spec().shape)


if __name__ == '__main__':
    absltest.main()
