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
import tensorflow as tf
import alf.algorithms.memory as memory


class TestMemory(unittest.TestCase):
    def assertArrayEqual(self, x, y, epsilon=1e-6):
        self.assertEqual(x.shape, y.shape)
        self.assertLess(tf.reduce_max(abs(x - y)), epsilon)

    def test_memory(self):
        mem = memory.MemoryWithUsage(2, 3, scale=20)
        self.assertEqual(mem.dim, 2)
        self.assertEqual(mem.size, 3)

        v00 = tf.constant([1., 0])
        v10 = tf.constant([1., 2])
        v01 = tf.constant([0., 1])
        v11 = tf.constant([-2., 1])

        w0 = tf.stack([v00, v10])
        mem.write(w0)
        self.assertArrayEqual(mem.usage, tf.constant([[1., 0, 0], [1., 0, 0]]))
        r = mem.read(w0)
        self.assertArrayEqual(r, w0)
        self.assertArrayEqual(mem.usage, tf.constant([[2., 0, 0], [2., 0, 0]]))

        # w1 is othorgonal to w0
        w1 = tf.stack([v01, v11])
        mem.write(w1)
        self.assertArrayEqual(mem.usage, tf.constant([[2., 1, 0], [2., 1, 0]]))
        r = mem.read(w1)
        self.assertArrayEqual(r, w1)
        self.assertArrayEqual(mem.usage, tf.constant([[2., 2, 0], [2., 2, 0]]))
        r = mem.read(w0)
        self.assertArrayEqual(r, w0)
        self.assertArrayEqual(mem.usage, tf.constant([[3., 2, 0], [3., 2, 0]]))
        r = mem.read(tf.constant([[2., 2.], [1, 1]]))
        self.assertArrayEqual(r, tf.constant([[0.5, 0.5], [1, 2]]))
        self.assertArrayEqual(mem.usage,
                              tf.constant([[3.5, 2.5, 0], [4., 2, 0]]))

        mem.write(w0)
        # current memory:  [v00 v01 v00] [v10, v11, v10]
        self.assertArrayEqual(mem.usage,
                              tf.constant([[3.5, 2.5, 1], [4., 2, 1]]))

        rkey = tf.stack([w0[0], w1[1]])
        r = mem.read(rkey)
        self.assertArrayEqual(r, rkey)
        r = mem.read(rkey)
        self.assertArrayEqual(r, rkey)
        r = mem.read(rkey)
        self.assertArrayEqual(r, rkey)
        r = mem.read(rkey)
        self.assertArrayEqual(r, rkey)
        self.assertArrayEqual(mem.usage,
                              tf.constant([[5.5, 2.5, 3], [4., 6, 1]]))

        mem.write(w0)
        # current memory:  [v00 v00 v00] [v10, v11, v10]
        self.assertArrayEqual(mem.usage, tf.constant([[5.5, 1, 3], [4., 6,
                                                                    1]]))
        mem.read(w1)
        self.assertArrayEqual(r, tf.stack([v00, v11]))


if __name__ == '__main__':
    unittest.main()
