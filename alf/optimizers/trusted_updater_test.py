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
from alf.optimizers.trusted_updater import TrustedUpdater


class TrustedUpdaterTest(tf.test.TestCase):
    def test_trusted_updater(self):
        v1 = tf.Variable(initial_value=[1.0, 2.0], trainable=True)
        v2 = tf.Variable(initial_value=tf.ones((8, )), trainable=True)
        updater = TrustedUpdater([v1, v2])

        old_sum_v1 = tf.reduce_sum(v1)
        old_sum_v2 = tf.reduce_sum(v2)
        v1.assign_add(tf.ones((2, )))
        v2.assign_add(tf.ones((8, )))

        def _change_f1():
            tf.print(v1, v2)
            sum_v1 = tf.reduce_sum(v1)
            sum_v2 = tf.reduce_sum(v2)
            return sum_v1 - old_sum_v1, sum_v2 - old_sum_v2

        # Test for correctly adjusting the variables
        changes, steps = updater.adjust_step(_change_f1, (1., 2.))
        self.assertLess(changes[0].numpy(), 1.)
        self.assertLess(changes[1].numpy(), 2.)
        self.assertEqual(2, steps.numpy())

        def _change_f2():
            return (8., 8.)

        # Test for detecting that change cannot be reduced
        with self.assertRaises(tf.errors.InvalidArgumentError):
            updater.adjust_step(_change_f2, (1., 2.))


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    tf.test.main()
