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

from absl.testing import parameterized

import tensorflow as tf
from alf.utils.reservoir_sampler import ReservoirSampler


class ReservoirSamplerTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters((1, 40, 100), (2, 30, 100), (3, 45, 200))
    def test_data_buffer(self, s, K, T):
        dim = 1
        batch_size = T
        data_spec = [
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(dim, ), dtype=tf.float32)
        ]

        @tf.function
        def _get_batch(i, batch_size):
            x = tf.random.normal(shape=(batch_size, dim))
            x = [tf.range(i, i + batch_size, dtype=tf.int32), x]
            return x

        sampler = ReservoirSampler(data_spec=data_spec, capacity=K, speed=s)
        sampler.add_batch = tf.function(sampler.add_batch)
        selected = dict()
        repeats = 10000
        for _ in range(repeats):
            sampler.clear()
            for i in range(0, T, batch_size):
                batch = _get_batch(i, batch_size)
                sampler.add_batch(batch)
            kept = sampler.get_all()[0].numpy()
            for x in kept:
                selected[x] = selected.get(x, 0) + 1

        total = 0
        for t1, c1 in selected.items():
            for t2, c2 in selected.items():
                # t1 and t2 should be large enough for accurate c1 and c2
                if t1 >= K * (s + 1) and t2 >= K * (s + 1):
                    total += (t1 != t2)
                    q1, q2 = 1, 1
                    for j in range(s - 1):
                        q1 *= (t1 - j) / (T - j - 1)  # theoretical probability
                        q2 *= (t2 - j) / (T - j - 1)
                    self.assertAlmostEqual(c1 / c2, q1 / q2, places=1)
        print("%d time step pairs were tested" % total)


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    tf.test.main()
