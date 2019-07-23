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
"""Test various functions related to calculating values."""

import tensorflow as tf

from tf_agents.trajectories.time_step import TimeStep, StepType

from alf.utils import common, value_ops


class DiscountedReturnTest(tf.test.TestCase):
    """Tests for alf.utils.value_ops.discounted_return
    """

    def test_discounted_return(self):
        """Test alf.utils.value_ops.discounted_return
        """
        values = tf.constant([[1.] * 5], tf.float32)
        step_types = tf.constant([[StepType.MID] * 5], tf.int64)
        rewards = tf.constant([[2.] * 5], tf.float32)
        discounts = tf.constant([[0.9] * 5], tf.float32)
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
                time_major=False), expected)

        # two episodes, and exceed by time limit (discount=1)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        expected = tf.constant(
            [[(1 * 0.9 + 2) * 0.9 + 2, 1 * 0.9 + 2, 1, 1 * 0.9 + 2]],
            dtype=tf.float32)
        self.assertAllClose(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                time_major=False), expected)

        # tow episodes, and end normal (discount=0)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        discounts = tf.constant([[0.9, 0.9, 0.0, 0.9, 0.9]])
        expected = tf.constant([[(0 * 0.9 + 2) * 0.9 + 2, 2, 1, 1 * 0.9 + 2]],
                               dtype=tf.float32)

        self.assertAllClose(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                time_major=False), expected)


class GeneralizedAdvantageTest(tf.test.TestCase):
    """Tests for alf.utils.value_ops.generalized_advantage_estimation
    """

    def test_generalized_advantage_estimation(self):
        """Test alf.utils.value_ops.generalized_advantage_estimation
        """
        values = tf.constant([[2.] * 5], tf.float32)
        step_types = tf.constant([[StepType.MID] * 5], tf.int64)
        rewards = tf.constant([[3.] * 5], tf.float32)
        discounts = tf.constant([[0.9] * 5], tf.float32)
        td_lambda = 0.6 / 0.9

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
                td_lambda=td_lambda,
                time_major=False), expected)

        # two episodes, and exceed by time limit (discount=1)

        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        expected = tf.constant([[d * 0.6 + d, d, 0, d]], dtype=tf.float32)
        self.assertAllClose(
            value_ops.generalized_advantage_estimation(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                td_lambda=td_lambda,
                time_major=False), expected)

        # tow episodes, and end normal (discount=0)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        discounts = tf.constant([[0.9, 0.9, 0.0, 0.9, 0.9]])
        expected = tf.constant([[1 * 0.6 + d, 1, 0, d]], dtype=tf.float32)

        self.assertAllClose(
            value_ops.generalized_advantage_estimation(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                td_lambda=td_lambda,
                time_major=False), expected)


class VTraceTest(tf.test.TestCase):
    """Tests for alf.utils.value_ops.vtrace_returns_and_advantages_impl
    """

    def test_vtrace_returns_and_advantages_impl_on_policy_no_last_step(self):
        """Test vtrace_returns_and_advantages_impl on policy no last_step
            in the middle of the trajectory.
        """
        importance_ratio_clipped = tf.constant([[1.] * 5], tf.float32)
        values = tf.constant([[2.] * 5], tf.float32)
        step_types = tf.constant([[StepType.MID] * 5], tf.int64)
        rewards = tf.constant([[3.] * 5], tf.float32)
        discounts = tf.constant([[0.9] * 5], tf.float32)
        td_lambda = 1.0

        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            time_major=False)
        expected_advantages = value_ops.generalized_advantage_estimation(
            rewards=rewards,
            values=values,
            step_types=step_types,
            discounts=discounts,
            td_lambda=td_lambda,
            time_major=False)
        expected_advantages = tf.transpose(a=expected_advantages)
        expected_advantages = common.tensor_extend_zero(expected_advantages)
        expected_advantages = tf.transpose(a=expected_advantages)
        self.assertAllClose(
            expected_advantages, advantages, msg='advantages differ')

        expected_returns = value_ops.discounted_return(
            rewards=rewards,
            values=values,
            step_types=step_types,
            discounts=discounts,
            time_major=False)
        expected_returns = tf.transpose(a=expected_returns)
        values = tf.transpose(a=values)
        expected_returns = common.tensor_extend(expected_returns, values[-1])
        expected_returns = tf.transpose(a=expected_returns)
        self.assertAllClose(expected_returns, returns, msg='returns differ')

    def test_vtrace_returns_and_advantages_impl_on_policy_has_last_step(self):
        """Test vtrace_returns_and_advantages_impl on policy has last_step
            in the middle of the trajectory.
        """
        importance_ratio_clipped = tf.constant([[1.] * 5], tf.float32)
        values = tf.constant([[2., 2.1, 2.2, 2.3, 2.4]], tf.float32)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        rewards = tf.constant([[3., 3.1, 3.2, 3.3, 3.4]], tf.float32)
        discounts = tf.constant([[0.9, 0.9, 0.0, 0.9, 0.9]])
        td_lambda = 1.0

        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            time_major=False)
        expected_advantages = value_ops.generalized_advantage_estimation(
            rewards=rewards,
            values=values,
            step_types=step_types,
            discounts=discounts,
            td_lambda=td_lambda,
            time_major=False)
        expected_advantages = tf.transpose(a=expected_advantages)
        expected_advantages = common.tensor_extend_zero(expected_advantages)
        expected_advantages = tf.transpose(a=expected_advantages)
        self.assertAllClose(
            expected_advantages, advantages, msg='advantages differ')

        expected_returns = value_ops.discounted_return(
            rewards=rewards,
            values=values,
            step_types=step_types,
            discounts=discounts,
            time_major=False)
        expected_returns = tf.transpose(a=expected_returns)
        values = tf.transpose(a=values)
        expected_returns = common.tensor_extend(expected_returns, values[-1])
        expected_returns = tf.transpose(a=expected_returns)
        self.assertAllClose(expected_returns, returns, msg='returns differ')

    def test_vtrace_returns_and_advantages_impl_off_policy_has_last_step(self):
        """Test vtrace_returns_and_advantages_impl off policy has last_step
            in the middle of the trajectory.
        """
        r = 0.999
        d = 0.9
        importance_ratio_clipped = tf.constant([[r] * 5], tf.float32)
        values = tf.constant([[2.] * 5], tf.float32)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        rewards = tf.constant([[3.] * 5], tf.float32)
        discounts = tf.constant([[d, d, 0., d, d]])
        td_lambda = 1.0

        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            time_major=False)

        td3 = (3. + 2. * d - 2.) * r
        expected_returns = tf.constant(
            [[td3 + d * r * (3. - 2.) * r, r, 0, td3, 0]], tf.float32) + values
        # 5.695401, 2.999   , 2.      , 4.7972  , 2.
        self.assertAllClose(expected_returns, returns, msg='returns differ')

        is_lasts = tf.cast(
            tf.equal(tf.transpose(a=step_types), StepType.LAST), tf.float32)
        expected_advantages = (1 - is_lasts[:-1]) * r * (
            tf.transpose(a=rewards)[1:] + tf.transpose(a=discounts)[1:] *
            tf.transpose(a=expected_returns)[1:] - tf.transpose(a=values)[:-1])
        expected_advantages = common.tensor_extend_zero(expected_advantages)
        expected_advantages = tf.transpose(a=expected_advantages)
        # 3.695401, 0.999   , 0.      , 2.7972  , 0.
        self.assertAllClose(
            expected_advantages, advantages, msg='advantages differ')

        # a case where values are not uniform over time.
        values = tf.constant([[0., 1., 2., 3., 4.]], tf.float32)
        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            time_major=False)

        td3 = (3. + 4. * d - 3) * r
        td1 = 2 * r
        expected_returns = tf.constant([[(3. + 1. * d - 0) * r + d * r * td1,
                                         td1, 0, td3, 0]], tf.float32) + values
        # 5.692502, 2.998   , 2.      , 6.5964  , 4.
        self.assertAllClose(expected_returns, returns, msg='returns differ')

        is_lasts = tf.cast(
            tf.equal(tf.transpose(a=step_types), StepType.LAST), tf.float32)
        expected_advantages = (1 - is_lasts[:-1]) * r * (
            tf.transpose(a=rewards)[1:] + tf.transpose(a=discounts)[1:] *
            tf.transpose(a=expected_returns)[1:] - tf.transpose(a=values)[:-1])
        expected_advantages = common.tensor_extend_zero(expected_advantages)
        expected_advantages = tf.transpose(a=expected_advantages)
        # 5.692502, 1.998   , 0.      , 3.5964  , 0.
        self.assertAllClose(
            expected_advantages, advantages, msg='advantages differ')

    def test_vtrace_impl_on_policy_has_last_step_with_lambda(self):
        """Test vtrace_returns_and_advantages_impl on policy has last_step
            in the middle of the trajectory, and has td_lambda = 0.95

            Hasn't passed test yet.
        """
        importance_ratio_clipped = tf.constant([[1.] * 5], tf.float32)
        values = tf.constant([[2.] * 5], tf.float32)
        step_types = tf.constant([[
            StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
            StepType.MID
        ]], tf.int32)
        rewards = tf.constant([[3.] * 5], tf.float32)
        discounts = tf.constant([[0.9, 0.9, 0.0, 0.9, 0.9]])
        td_lambda = 0.95

        returns, advantages = value_ops.vtrace_returns_and_advantages_impl(
            importance_ratio_clipped,
            rewards,
            values,
            step_types,
            discounts,
            td_lambda=td_lambda,
            time_major=False)
        expected_advantages = value_ops.generalized_advantage_estimation(
            rewards=rewards,
            values=values,
            step_types=step_types,
            discounts=discounts,
            td_lambda=td_lambda,
            time_major=False)
        expected_advantages = tf.transpose(a=expected_advantages)
        expected_advantages = common.tensor_extend_zero(expected_advantages)
        discounts = tf.transpose(a=discounts)
        values = tf.transpose(a=values)
        step_types = tf.transpose(a=step_types)
        is_lasts = tf.cast(tf.equal(step_types, StepType.LAST), tf.float32)

        # Advantage_s of vtrace + (1 - lambda) * gamma * V_s+1 == Advantage_s of GAE
        expected_advantages -= (
            (1 - td_lambda) * (1 - is_lasts) * common.tensor_extend_zero(
                discounts[1:]) * common.tensor_extend_zero(values[1:]))
        expected_advantages = tf.transpose(a=expected_advantages)
        # self.assertAllClose(
        #     expected_advantages, advantages, msg='advantages differ')
        discounts = tf.transpose(a=discounts)
        values = tf.transpose(a=values)
        step_types = tf.transpose(a=step_types)

        expected_returns = value_ops.discounted_return(
            rewards=rewards,
            values=values,
            step_types=step_types,
            discounts=discounts,
            # TODO(Le): need td_lambda version of discounted_return
            time_major=False)
        expected_returns = tf.transpose(a=expected_returns)
        values = tf.transpose(a=values)
        expected_returns = common.tensor_extend(expected_returns, values[-1])
        expected_returns = tf.transpose(a=expected_returns)
        # self.assertAllClose(expected_returns, returns, msg='returns differ')


if __name__ == '__main__':
    tf.test.main()
