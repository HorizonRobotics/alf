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

import unittest
import torch
from alf.data_structures import TimeStep, StepType
from alf.utils import value_ops
import numpy as np

StepType.MID


class DiscountedReturnTest(unittest.TestCase):
    """Tests for alf.utils.value_ops.discounted_return
    """

    def test_discounted_return(self):
        values = torch.tensor([[1.] * 5], dtype=torch.float32)
        step_types = torch.tensor([[np.asscalar(StepType.MID)] * 5],
                                  dtype=torch.int64)
        rewards = torch.tensor([[2.] * 5], dtype=torch.float32)
        discounts = torch.tensor([[0.9] * 5], dtype=torch.float32)
        expected = torch.tensor(
            [[(((1 * 0.9 + 2) * 0.9 + 2) * 0.9 + 2) * 0.9 + 2,
              ((1 * 0.9 + 2) * 0.9 + 2) * 0.9 + 2,
              (1 * 0.9 + 2) * 0.9 + 2, 1 * 0.9 + 2]],
            dtype=torch.float32)
        np.testing.assert_array_almost_equal(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                time_major=False), expected)

        # two episodes, and exceed by time limit (discount=1)
        step_types = torch.tensor([[
            np.asscalar(StepType.MID),
            np.asscalar(StepType.MID),
            np.asscalar(StepType.LAST),
            np.asscalar(StepType.MID),
            np.asscalar(StepType.MID)
        ]],
                                  dtype=torch.int32)
        expected = torch.tensor(
            [[(1 * 0.9 + 2) * 0.9 + 2, 1 * 0.9 + 2, 1, 1 * 0.9 + 2]],
            dtype=torch.float32)
        np.testing.assert_array_almost_equal(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                time_major=False), expected)

        # tow episodes, and end normal (discount=0)
        step_types = torch.tensor([[
            np.asscalar(StepType.MID),
            np.asscalar(StepType.MID),
            np.asscalar(StepType.LAST), StepType.MID,
            np.asscalar(StepType.MID)
        ]],
                                  dtype=torch.int32)
        discounts = torch.tensor([[0.9, 0.9, 0.0, 0.9, 0.9]])
        expected = torch.tensor([[(0 * 0.9 + 2) * 0.9 + 2, 2, 1, 1 * 0.9 + 2]],
                                dtype=torch.float32)

        np.testing.assert_array_almost_equal(
            value_ops.discounted_return(
                rewards=rewards,
                values=values,
                step_types=step_types,
                discounts=discounts,
                time_major=False), expected)


# class GeneralizedAdvantageTest(tf.test.TestCase):
#     """Tests for alf.utils.value_ops.generalized_advantage_estimation
#     """

#     def test_generalized_advantage_estimation(self):
#         values = tf.constant([[2.] * 5], tf.float32)
#         step_types = tf.constant([[StepType.MID] * 5], tf.int64)
#         rewards = tf.constant([[3.] * 5], tf.float32)
#         discounts = tf.constant([[0.9] * 5], tf.float32)
#         td_lambda = 0.6 / 0.9

#         d = 2 * 0.9 + 1
#         expected = tf.constant([[((d * 0.6 + d) * 0.6 + d) * 0.6 + d,
#                                  (d * 0.6 + d) * 0.6 + d, d * 0.6 + d, d]],
#                                dtype=tf.float32)
#         self.assertAllClose(
#             value_ops.generalized_advantage_estimation(
#                 rewards=rewards,
#                 values=values,
#                 step_types=step_types,
#                 discounts=discounts,
#                 td_lambda=td_lambda,
#                 time_major=False), expected)

#         # two episodes, and exceed by time limit (discount=1)

#         step_types = tf.constant([[
#             StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
#             StepType.MID
#         ]], tf.int32)
#         expected = tf.constant([[d * 0.6 + d, d, 0, d]], dtype=tf.float32)
#         self.assertAllClose(
#             value_ops.generalized_advantage_estimation(
#                 rewards=rewards,
#                 values=values,
#                 step_types=step_types,
#                 discounts=discounts,
#                 td_lambda=td_lambda,
#                 time_major=False), expected)

#         # tow episodes, and end normal (discount=0)
#         step_types = tf.constant([[
#             StepType.MID, StepType.MID, StepType.LAST, StepType.MID,
#             StepType.MID
#         ]], tf.int32)
#         discounts = tf.constant([[0.9, 0.9, 0.0, 0.9, 0.9]])
#         expected = tf.constant([[1 * 0.6 + d, 1, 0, d]], dtype=tf.float32)

#         self.assertAllClose(
#             value_ops.generalized_advantage_estimation(
#                 rewards=rewards,
#                 values=values,
#                 step_types=step_types,
#                 discounts=discounts,
#                 td_lambda=td_lambda,
#                 time_major=False), expected)
