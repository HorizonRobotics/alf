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
"""Test cases adapted from tf_agents' tensor_spec_test.py."""

import numpy as np
import unittest
import alf
from absl.testing import parameterized

from alf.tensor_specs import TensorSpec, BoundedTensorSpec, concat_specs
from alf.tensor_specs import torch_dtype_to_str

import torch

TYPE_PARAMETERS = ((torch.int32, ), (torch.int64, ), (torch.float32, ),
                   (torch.float64, ), (torch.uint8, ))


class TensorSpecTest(parameterized.TestCase, alf.test.TestCase):
    def setUp(self):
        super().setUp()
        self._shape = (20, 30)

    @parameterized.parameters(*TYPE_PARAMETERS)
    def testIntegerSamplesIncludeUpperBound(self, dtype):
        if dtype.is_floating_point:  # Only test on integer dtypes.
            return
        spec = BoundedTensorSpec(self._shape, dtype, 3, 3)
        sample = spec.sample()
        self.assertEqual(sample.shape, self._shape)
        self.assertTrue(torch.all(sample == 3))

    @parameterized.parameters(*TYPE_PARAMETERS)
    def testIntegerSamplesExcludeMaxOfDtype(self, dtype):
        # Exclude non integer types and uint8 (has special sampling logic).
        if dtype.is_floating_point or dtype == torch.uint8:
            return
        info = np.iinfo(torch_dtype_to_str(dtype))
        spec = BoundedTensorSpec(self._shape, dtype, info.max - 1,
                                 info.max - 1)
        sample = spec.sample(outer_dims=(1, ))
        self.assertEqual(sample.shape, (1, ) + self._shape)
        self.assertTrue(torch.all(sample == info.max - 1))

    @parameterized.parameters(*TYPE_PARAMETERS)
    def testBoundedTensorSpecSample(self, dtype):
        if not dtype.is_floating_point:
            return
        # minimum and maximum shape broadcasting
        spec = BoundedTensorSpec(self._shape, dtype, (0, ) * 30, 3)
        sample = spec.sample()
        self.assertEqual(self._shape, sample.shape)
        self.assertTrue(torch.all(sample <= 3))
        self.assertTrue(torch.all(0 <= sample))

        # last minimum is greater than last maximum
        self.assertRaises(AssertionError, BoundedTensorSpec, self._shape,
                          dtype, (0, ) * 29 + (2, ), (1, ) * 30)

    @parameterized.parameters(*TYPE_PARAMETERS)
    def testTensorSpecZero(self, dtype):
        spec = TensorSpec(self._shape, dtype)
        sample = spec.zeros(outer_dims=(3, 10))
        self.assertEqual(sample.shape, (3, 10) + self._shape)
        self.assertTrue(torch.all(sample == 0))

    def testTensorSpecReplace(self):
        spec = TensorSpec(shape=(3, 4), dtype=torch.int32)
        spec1 = spec.replace(shape=(4, 5))
        spec2 = spec.replace(dtype=torch.float32)
        spec3 = spec.replace(shape=(4, 5), dtype=torch.float32)
        self.assertEqual(TensorSpec(shape=(4, 5), dtype=torch.int32), spec1)
        self.assertEqual(TensorSpec(shape=(3, 4), dtype=torch.float32), spec2)
        self.assertEqual(TensorSpec(shape=(4, 5), dtype=torch.float32), spec3)

    def testBoundedTensorSpecReplace(self):
        spec = BoundedTensorSpec(
            shape=(3, 4),
            dtype=torch.int32,
            minimum=np.zeros(4),
            maximum=np.ones(4))
        new_spec = spec.replace(shape=(8, 4), minimum=np.full((4, ), -1))
        self.assertEqual(
            BoundedTensorSpec(
                shape=(8, 4),
                dtype=torch.int32,
                minimum=np.array([-1, -1, -1, -1]),
                maximum=np.array([1, 1, 1, 1])), new_spec)

    def test_concat_specs1(self):
        """Concat TensorSpecs"""

        spec1 = TensorSpec(shape=(3, 4), dtype=torch.int32)
        spec2 = TensorSpec(shape=(4, ), dtype=torch.float32)
        spec3 = TensorSpec(shape=(5, 6), dtype=torch.float32)

        self.assertRaises(AssertionError, concat_specs, [spec1, spec2, spec3])
        spec = concat_specs([spec2, spec3])
        self.assertEqual(spec.shape, (34, ))
        self.assertEqual(spec.dtype, torch.float32)

    def test_concat_specs2(self):
        """Concat BoundedTensorSpecs"""
        spec1 = TensorSpec(shape=(2, 3), dtype=torch.int64)
        spec2 = BoundedTensorSpec(
            shape=(2, 3), minimum=1, maximum=3, dtype=torch.int64)
        spec3 = BoundedTensorSpec(
            shape=(4, ),
            minimum=[1, 2, 3, 4],
            maximum=[5, 6, 7, 8],
            dtype=torch.int64)
        self.assertRaises(AssertionError, concat_specs, [spec1, spec2, spec3])
        spec = concat_specs([spec2, spec3])
        self.assertEqual(spec.shape, (10, ))
        self.assertEqual(spec.dtype, torch.int64)
        self.assertArrayEqual(spec.minimum,
                              np.array([1, 1, 1, 1, 1, 1, 1, 2, 3, 4]))
        self.assertArrayEqual(spec.maximum,
                              np.array([3, 3, 3, 3, 3, 3, 5, 6, 7, 8]))


if __name__ == "__main__":
    unittest.main()
