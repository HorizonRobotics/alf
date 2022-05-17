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

import numpy as np
import torch

import alf
from alf.algorithms.td_loss import LowerBoundedTDLoss
from alf.data_structures import TimeStep, StepType, namedtuple

DataItem = namedtuple(
    "DataItem", ["reward", "step_type", "discount"], default_value=())


class LowerBoundedTDLossTest(unittest.TestCase):
    """Tests for alf.algorithms.td_loss.LowerBoundedTDLoss
    """

    def _check(self, res, expected):
        np.testing.assert_array_almost_equal(res, expected)

    def test_compute_td_target_nstep_bootstrap_lowerbound(self):
        loss = LowerBoundedTDLoss(
            gamma=1., improve_w_nstep_bootstrap=True, td_lambda=1)
        # Tensors are transposed to be time_major [T, B, ...]
        step_types = torch.tensor([[StepType.MID] * 5],
                                  dtype=torch.int64).transpose(0, 1)
        rewards = torch.tensor([[2.] * 5], dtype=torch.float32).transpose(0, 1)
        discounts = torch.tensor([[0.9] * 5], dtype=torch.float32).transpose(
            0, 1)
        values = torch.tensor([[1.] * 5], dtype=torch.float32).transpose(0, 1)
        info = DataItem(
            reward=rewards, step_type=step_types, discount=discounts)
        returns, value, _ = loss.compute_td_target(info, values, values)
        expected_return = torch.tensor(
            [[2 + 0.9 * (2 + 0.9 * (2 + 0.9 * (2 + 0.9))), 0, 0, 0]],
            dtype=torch.float32).transpose(0, 1)
        self._check(res=returns, expected=expected_return)

        expected_value = torch.tensor([[1, 0, 0, 0, 0]],
                                      dtype=torch.float32).transpose(0, 1)
        self._check(res=value, expected=expected_value)

        # n-step return is below 1-step
        values[2:] = -10
        expected_return[0] = 2 + 0.9
        returns, value, _ = loss.compute_td_target(info, values, values)
        self._check(res=returns, expected=expected_return)


if __name__ == '__main__':
    alf.test.main()
