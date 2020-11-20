# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import math

import torch

import alf
from alf.utils import math_ops
from alf.utils.normalizers import ScalarWindowNormalizer
from alf.utils.normalizers import ScalarEMNormalizer
from alf.utils.normalizers import ScalarAdaptiveNormalizer


class NormalizersTest(parameterized.TestCase, alf.test.TestCase):
    def setUp(self):
        super().setUp()
        self._batch_size = 5
        self._window_size = 100
        self._tensors = torch.rand(self._window_size, self._batch_size)

        def _verify_normalization(weights,
                                  normalized_tensor,
                                  eps,
                                  use_var=True):
            tensors_mean = torch.sum(weights * self._tensors)
            if use_var:
                tensors_var = torch.sum(
                    weights * math_ops.square(self._tensors - tensors_mean))
            else:
                tensors_var = torch.ones_like(tensors_mean)
            target_normalized_tensor = alf.layers.normalize_along_batch_dims(
                self._tensors[-1],
                tensors_mean,
                tensors_var,
                variance_epsilon=eps)
            self.assertTensorClose(
                normalized_tensor, target_normalized_tensor, epsilon=1e-4)

        self._verify_normalization = _verify_normalization

    @parameterized.parameters((True, ), (False, ))
    def test_window_normalizer(self, unit_std):
        normalizer = ScalarWindowNormalizer(
            window_size=self._window_size, unit_std=unit_std)
        for i in range(self._window_size):
            normalized_tensor = normalizer.normalize(self._tensors[i])
        weights = torch.ones((self._window_size, self._batch_size),
                             dtype=torch.float32)
        weights /= torch.sum(weights)

        self._verify_normalization(
            weights,
            normalized_tensor,
            normalizer._variance_epsilon,
            use_var=not unit_std)

    @parameterized.parameters((True, ), (False, ))
    def test_em_normalizer(self, unit_std):
        update_rate = 0.1
        normalizer = ScalarEMNormalizer(
            update_rate=update_rate, unit_std=unit_std)
        for i in range(self._window_size):
            normalized_tensor = normalizer.normalize(self._tensors[i])

        weights = torch.as_tensor([(
            math.pow(1 - update_rate, self._window_size - 1 - i) * update_rate)
                                   for i in range(self._window_size)],
                                  dtype=torch.float32)
        ones = torch.ones((self._batch_size, ), dtype=torch.float32)
        weights = torch.ger(weights, ones)
        weights /= torch.sum(weights)  # reduce em bias

        self._verify_normalization(
            weights,
            normalized_tensor,
            normalizer._variance_epsilon,
            use_var=not unit_std)

    @parameterized.parameters((True, ), (False, ))
    def test_adaptive_normalizer(self, unit_std):
        speed = 8.0
        normalizer = ScalarAdaptiveNormalizer(speed=speed, unit_std=unit_std)
        for i in range(self._window_size):
            normalized_tensor = normalizer.normalize(self._tensors[i])

        weights = []
        acc_r = 1.0
        for i in range(self._window_size):
            r = speed / (speed + self._window_size - 1 - i)
            weights.append(r * acc_r)
            acc_r *= 1 - r
        weights = torch.as_tensor(weights[::-1], dtype=torch.float32)
        ones = torch.ones((self._batch_size, ), dtype=torch.float32)
        weights = torch.ger(weights, ones)
        weights /= torch.sum(weights)  # reduce adaptive bias

        self._verify_normalization(
            weights,
            normalized_tensor,
            normalizer._variance_epsilon,
            use_var=not unit_std)


if __name__ == '__main__':
    alf.test.main()
