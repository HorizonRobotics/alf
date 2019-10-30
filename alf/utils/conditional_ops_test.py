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
"""Unittests for conditional_ops.py."""

import tensorflow as tf

from alf.utils.conditional_ops import conditional_update, select_from_mask


class ConditionalOpsTest(tf.test.TestCase):
    def test_conditional_update(self):
        def _func(x, y):
            return x + 1, y - 1

        batch_size = 256
        target = (tf.random.uniform([batch_size, 3]),
                  tf.random.uniform([batch_size]))
        x = tf.random.uniform([batch_size, 3])
        y = tf.random.uniform([batch_size])

        cond = tf.constant([False] * batch_size)
        updated_target = conditional_update(target, cond, _func, x, y)
        self.assertAllEqual(updated_target[0], target[0])
        self.assertAllEqual(updated_target[1], target[1])

        cond = tf.constant([True] * batch_size)
        updated_target = conditional_update(target, cond, _func, x, y)
        self.assertAllEqual(updated_target[0], x + 1)
        self.assertAllEqual(updated_target[1], y - 1)

        cond = tf.random.uniform((batch_size, )) < 0.5
        updated_target = conditional_update(target, cond, _func, x, y)
        self.assertAllEqual(
            select_from_mask(updated_target[0], cond),
            select_from_mask(x + 1, cond))
        self.assertAllEqual(
            select_from_mask(updated_target[1], cond),
            select_from_mask(y - 1, cond))

        vx = tf.Variable(initial_value=0.)
        vy = tf.Variable(initial_value=0.)

        def _func1(x, y):
            vx.assign(tf.reduce_sum(x))
            vy.assign(tf.reduce_sum(y))
            return ()

        # test empty return
        conditional_update((), cond, _func1, x, y)
        self.assertEqual(vx, tf.reduce_sum(select_from_mask(x, cond)))
        self.assertEqual(vy, tf.reduce_sum(select_from_mask(y, cond)))

    def test_select_from_mask(self):
        data = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [10, 11]])
        cond = tf.constant([False, True, True, False, False, True])
        result = select_from_mask(data, cond)
        self.assertAllEqual(result, tf.constant([[3, 4], [5, 6], [10, 11]]))


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
