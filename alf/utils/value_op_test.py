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

"""Tests for alf.utils.discounted_return"""

from absl.testing import parameterized
import tensorflow as tf
from alf.utils import value_ops


class DiscountedReturnTest(tf.test.TestCase, parameterized.TestCase):
    def test_discounted_return(self):
        values = tf.constant([[1.] * 5], tf.float32)
        rewards = tf.constant([[2.] * 5], tf.float32)
        is_lasts = tf.constant([[False] * 5], tf.bool)
        discounts = tf.constant([[0.9] * 5], tf.float32)

        expected = tf.constant([[
            (((1 * 0.9 + 2) * 0.9 + 2) * 0.9 + 2) * 0.9 + 2,
            ((1 * 0.9 + 2) * 0.9 + 2) * 0.9 + 2,
            (1 * 0.9 + 2) * 0.9 + 2,
            1 * 0.9 + 2,
            1]], dtype=tf.float32)
        self.assertAllClose(value_ops.discounted_return(
            rewards, values, is_lasts, discounts,
            time_major=False), expected)

        # two episodes, and exceed by time limit
        is_lasts = tf.constant([[False, False, True, False, False]], tf.bool)
        expected = tf.constant([[
            (1 * 0.9 + 2) * 0.9 + 2,
            1 * 0.9 + 2,
            1,
            1 * 0.9 + 2,
            1]], dtype=tf.float32)
        self.assertAllClose(value_ops.discounted_return(
            rewards, values, is_lasts, discounts,
            time_major=False), expected)

        # tow episodes, and end normal
        discounts = tf.constant([[0.9, 0.0, 0.9, 0.9, 0.9]])
        expected = tf.constant([[
            (0 * 0.9 + 2) * 0.9 + 2,
            0 * 0.9 + 2,
            1,
            1 * 0.9 + 2,
            1]], dtype=tf.float32)

        self.assertAllClose(value_ops.discounted_return(
            rewards, values, is_lasts, discounts,
            time_major=False), expected)


if __name__ == '__main__':
    tf.test.main()
