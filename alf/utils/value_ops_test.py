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

import tensorflow as tf

from tf_agents.trajectories.time_step import TimeStep, StepType

from alf.utils import value_ops


class DiscountedReturnTest(tf.test.TestCase):
    """Tests for alf.utils.value_ops.discounted_return
    """

    def test_discounted_return(self):
        values = tf.constant([[1.] * 4], tf.float32)
        step_types = tf.constant([[StepType.MID] * 4], tf.int64)
        rewards = tf.constant([[2.] * 4], tf.float32)
        discounts = tf.constant([[0.9] * 4], tf.float32)
        final_value = tf.constant([1.0], tf.float32)
        final_time_step = TimeStep(
            step_type=tf.constant([StepType.MID], tf.int32),
            reward=tf.constant([2.], tf.float32),
            discount=tf.constant([0.9], tf.float32),
            observation=())

        expected = tf.constant(
            [[(((1 * 0.9 + 2) * 0.9 + 2) * 0.9 + 2) * 0.9 + 2,
              ((1 * 0.9 + 2) * 0.9 + 2) * 0.9 + 2,
              (1 * 0.9 + 2) * 0.9 + 2, 1 * 0.9 + 2]],
            dtype=tf.float32)
        self.assertAllClose(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                final_value=final_value,
                final_time_step=final_time_step,
                time_major=False), expected)

        # two episodes, and exceed by time limit (discount=1)
        step_types = tf.constant(
            [[StepType.MID, StepType.MID, StepType.LAST, StepType.MID]],
            tf.int32)
        expected = tf.constant(
            [[(1 * 0.9 + 2) * 0.9 + 2, 1 * 0.9 + 2, 1, 1 * 0.9 + 2]],
            dtype=tf.float32)
        self.assertAllClose(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                final_value=final_value,
                final_time_step=final_time_step,
                time_major=False), expected)

        # tow episodes, and end normal (discount=0)
        step_types = tf.constant(
            [[StepType.MID, StepType.MID, StepType.LAST, StepType.MID]],
            tf.int32)
        discounts = tf.constant([[0.9, 0.9, 0.0, 0.9]])
        expected = tf.constant([[(0 * 0.9 + 2) * 0.9 + 2, 2, 1, 1 * 0.9 + 2]],
                               dtype=tf.float32)

        self.assertAllClose(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                final_value=final_value,
                final_time_step=final_time_step,
                time_major=False), expected)


class GeneralizedAdvantageTest(tf.test.TestCase):
    """Tests for alf.utils.value_ops.generalized_advantage_estimation
    """

    def test_generalized_advantage_estimation(self):
        values = tf.constant([[2.] * 4], tf.float32)
        step_types = tf.constant([[StepType.MID] * 4], tf.int64)
        rewards = tf.constant([[3.] * 4], tf.float32)
        discounts = tf.constant([[0.9] * 4], tf.float32)
        final_value = tf.constant([2.0], tf.float32)
        td_lambda = 0.6 / 0.9

        final_time_step = TimeStep(
            step_type=tf.constant([StepType.MID], tf.int32),
            reward=tf.constant([3.], tf.float32),
            discount=tf.constant([0.9], tf.float32),
            observation=())

        d = 2 * 0.9 + 1
        expected = tf.constant([[((d * 0.6 + d) * 0.6 + d) * 0.6 + d,
                                 (d * 0.6 + d) * 0.6 + d, d * 0.6 + d, d]],
                               dtype=tf.float32)
        self.assertAllClose(
            value_ops.generalized_advantage_estimation(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                final_value=final_value,
                final_time_step=final_time_step,
                td_lambda=td_lambda,
                time_major=False), expected)

        # two episodes, and exceed by time limit (discount=1)

        step_types = tf.constant(
            [[StepType.MID, StepType.MID, StepType.LAST, StepType.MID]],
            tf.int32)
        expected = tf.constant([[d * 0.6 + d, d, 0, d]], dtype=tf.float32)
        self.assertAllClose(
            value_ops.generalized_advantage_estimation(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                final_value=final_value,
                final_time_step=final_time_step,
                td_lambda=td_lambda,
                time_major=False), expected)

        # tow episodes, and end normal (discount=0)
        step_types = tf.constant(
            [[StepType.MID, StepType.MID, StepType.LAST, StepType.MID]],
            tf.int32)
        discounts = tf.constant([[0.9, 0.9, 0.0, 0.9]])
        expected = tf.constant([[1 * 0.6 + d, 1, 0, d]], dtype=tf.float32)

        self.assertAllClose(
            value_ops.generalized_advantage_estimation(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                final_value=final_value,
                final_time_step=final_time_step,
                td_lambda=td_lambda,
                time_major=False), expected)


if __name__ == '__main__':
    tf.test.main()
