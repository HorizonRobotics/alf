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

import numpy as np
import gin.tf
import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics.tf_metrics import \
    AverageEpisodeLengthMetric, AverageReturnMetric, \
    EnvironmentSteps, NumberOfEpisodes
from alf.environments import suite_mario


class SuiteMarioTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_mario.is_available():
            self.skipTest('suite_mario is not available.')
        else:
            gin.clear_config()

    def tearDown(self):
        super().tearDown()
        self._env.close()

    def test_mario_env(self):
        ctor = lambda: suite_mario.load(
            'SuperMarioBros-Nes', 'Level1-1', wrap_with_process=False)

        self._env = parallel_py_environment.ParallelPyEnvironment([ctor] * 4)
        env = tf_py_environment.TFPyEnvironment(self._env)
        self.assertEqual(np.uint8, env.observation_spec().dtype)
        self.assertEqual((84, 84, 4), env.observation_spec().shape)

        random_policy = random_tf_policy.RandomTFPolicy(
            env.time_step_spec(), env.action_spec())

        metrics = [
            AverageReturnMetric(batch_size=4),
            AverageEpisodeLengthMetric(batch_size=4),
            EnvironmentSteps(),
            NumberOfEpisodes()
        ]
        driver = dynamic_step_driver.DynamicStepDriver(
            env, random_policy, observers=metrics, num_steps=100)
        driver.run(maximum_iterations=10000)


if __name__ == '__main__':
    tf.test.main()
