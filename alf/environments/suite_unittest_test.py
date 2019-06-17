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

from absl import logging
import numpy as np

import unittest
from absl.testing import parameterized
import tensorflow as tf

from alf.environments.suite_unittest import ActionType
from alf.environments.suite_unittest import ValueUnittestEnv
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import RNNPolicyUnittestEnv
from tf_agents.trajectories.time_step import TimeStep, StepType


class SuiteUnittestEnvTest(parameterized.TestCase, unittest.TestCase):
    def assertArrayEqual(self, x, y):
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(float(tf.reduce_max(abs(x - y))), 0.)

    @parameterized.parameters(ActionType.Discrete, ActionType.Continuous)
    def test_value_unittest_env(self, action_type):
        batch_size = 1
        steps_per_episode = 13

        env = ValueUnittestEnv(
            batch_size, steps_per_episode, action_type=action_type)

        time_step = env.reset()
        for _ in range(10):
            for s in range(steps_per_episode):
                if s == 0:
                    step_type = StepType.FIRST
                    discount = 1.0
                elif s == steps_per_episode - 1:
                    step_type = StepType.LAST
                    discount = 0.0
                else:
                    step_type = StepType.MID
                    discount = 1.0

                self.assertArrayEqual(time_step.step_type,
                                      tf.constant([step_type] * batch_size))
                self.assertArrayEqual(time_step.reward,
                                      tf.constant([1.0] * batch_size))
                self.assertArrayEqual(time_step.discount,
                                      tf.constant([discount] * batch_size))

                action = tf.random.uniform((batch_size, 1),
                                           minval=0,
                                           maxval=2,
                                           dtype=tf.int64)
                time_step = env.step(action)

    @parameterized.parameters(ActionType.Discrete, ActionType.Continuous)
    def test_policy_unittest_env(self, action_type):
        batch_size = 100
        steps_per_episode = 13

        env = PolicyUnittestEnv(
            batch_size, steps_per_episode, action_type=action_type)

        time_step = env.reset()
        for _ in range(10):
            for s in range(steps_per_episode):
                if s == 0:
                    step_type = StepType.FIRST
                    discount = 1.0
                elif s == steps_per_episode - 1:
                    step_type = StepType.LAST
                    discount = 0.0
                else:
                    step_type = StepType.MID
                    discount = 1.0

                if s == 0:
                    reward = tf.constant([0.] * batch_size)
                else:
                    reward = tf.cast(
                        tf.equal(action, tf.cast(prev_observation, tf.int64)),
                        tf.float32)
                    reward = tf.reshape(reward, shape=(batch_size, ))

                self.assertArrayEqual(time_step.step_type,
                                      tf.constant([step_type] * batch_size))
                self.assertArrayEqual(time_step.reward, reward)
                self.assertArrayEqual(time_step.discount,
                                      tf.constant([discount] * batch_size))

                action = tf.random.uniform((batch_size, 1),
                                           minval=0,
                                           maxval=2,
                                           dtype=tf.int64)
                prev_observation = time_step.observation

                time_step = env.step(action)

    def test_rnn_policy_unittest_env(self):
        batch_size = 100
        steps_per_episode = 5
        gap = 3
        env = RNNPolicyUnittestEnv(batch_size, steps_per_episode, gap)

        time_step = env.reset()
        for _ in range(10):
            for s in range(steps_per_episode):
                if s == 0:
                    observation0 = time_step.observation

                if s == 0:
                    step_type = StepType.FIRST
                    discount = 1.0
                elif s == steps_per_episode - 1:
                    step_type = StepType.LAST
                    discount = 0.0
                else:
                    step_type = StepType.MID
                    discount = 1.0

                if s <= gap:
                    reward = tf.zeros((batch_size, ), dtype=tf.float32)
                else:
                    reward = tf.cast(
                        tf.equal(action * 2 - 1, tf.cast(
                            observation0, tf.int64)), tf.float32)
                    reward = tf.reshape(reward, shape=(batch_size, ))

                self.assertArrayEqual(time_step.step_type,
                                      tf.constant([step_type] * batch_size))
                self.assertArrayEqual(time_step.reward, reward)
                self.assertArrayEqual(time_step.discount,
                                      tf.constant([discount] * batch_size))

                action = tf.random.uniform((batch_size, 1),
                                           minval=0,
                                           maxval=2,
                                           dtype=tf.int64)
                time_step = env.step(action)


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    unittest.main()
