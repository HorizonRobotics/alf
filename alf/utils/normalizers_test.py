# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import math

import tensorflow as tf

from alf.utils.normalizers import ScalarWindowNormalizer
from alf.utils.normalizers import ScalarEMNormalizer
from alf.utils.normalizers import ScalarAdaptiveNormalizer


class NormalizersTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self._batch_size = 5
        self._window_size = 100
        self._tensors = tf.random.uniform(
            shape=(self._window_size, self._batch_size), maxval=1.0)

        def _verify_normalization(weights, normalized_tensor, eps):
            tensors_mean = tf.reduce_sum(weights * self._tensors)
            tensors_var = tf.reduce_sum(
                weights * tf.square(self._tensors - tensors_mean))
            target_normalized_tensor = tf.nn.batch_normalization(
                self._tensors[-1],
                tensors_mean,
                tensors_var,
                offset=None,
                scale=None,
                variance_epsilon=eps)
            self.assertAllClose(normalized_tensor, target_normalized_tensor)

        self._verify_normalization = _verify_normalization

    def test_window_normalizer(self):
        normalizer = ScalarWindowNormalizer(window_size=self._window_size)
        for i in range(self._window_size):
            normalized_tensor = normalizer.normalize(self._tensors[i])
        weights = tf.ones([self._window_size, self._batch_size],
                          dtype=tf.float32)
        weights /= tf.reduce_sum(weights)

        self._verify_normalization(weights, normalized_tensor,
                                   normalizer._variance_epsilon)

    def test_em_normalizer(self):
        update_rate = 0.1
        normalizer = ScalarEMNormalizer(update_rate=update_rate)
        for i in range(self._window_size):
            normalized_tensor = normalizer.normalize(self._tensors[i])

        weights = tf.convert_to_tensor([(
            math.pow(1 - update_rate, self._window_size - 1 - i) * update_rate)
                                        for i in range(self._window_size)],
                                       dtype=tf.float32)
        ones = tf.ones([self._batch_size], dtype=tf.float32)
        weights = tf.tensordot(weights, ones, axes=0)
        weights /= tf.reduce_sum(weights)  # reduce em bias

        self._verify_normalization(weights, normalized_tensor,
                                   normalizer._variance_epsilon)

    def test_adaptive_normalizer(self):
        speed = 8.0
        normalizer = ScalarAdaptiveNormalizer(speed=speed)
        for i in range(self._window_size):
            normalized_tensor = normalizer.normalize(self._tensors[i])

        weights = []
        acc_r = 1.0
        for i in range(self._window_size):
            r = speed / (speed + self._window_size - 1 - i)
            weights.append(r * acc_r)
            acc_r *= 1 - r
        weights = tf.convert_to_tensor(weights[::-1], dtype=tf.float32)
        ones = tf.ones([self._batch_size], dtype=tf.float32)
        weights = tf.tensordot(weights, ones, axes=0)
        weights /= tf.reduce_sum(weights)  # reduce adaptive bias

        self._verify_normalization(weights, normalized_tensor,
                                   normalizer._variance_epsilon)


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    tf.test.main()
