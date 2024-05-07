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
"""Unittests for actor_critic_loss.py"""

import unittest
import torch
import torch.distributions as td
import numpy as np

from alf.utils.dist_utils import compute_entropy, compute_log_probability


class TestEntropyExpand(unittest.TestCase):
    def test_entropy(self):
        m = td.categorical.Categorical(torch.Tensor([0.25, 0.75]))
        M = m.expand([2, 3])
        expected = torch.Tensor([[0.562335, 0.562335, 0.562335],
                                 [0.562335, 0.562335, 0.562335]])
        obtained = compute_entropy(M)
        np.testing.assert_array_almost_equal(expected, obtained)


class TestEntropy(unittest.TestCase):
    def test_entropy(self):
        M = td.categorical.Categorical(
            torch.Tensor([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]))
        expected = torch.Tensor([0.562335, 0.6931471, 0.562335])
        obtained = compute_entropy(M)
        np.testing.assert_array_almost_equal(expected, obtained)


class TestLogProbabilityExpand(unittest.TestCase):
    def test_log_probability(self):
        m = td.categorical.Categorical(torch.Tensor([0.25, 0.75]))
        M = m.expand([2, 3])
        actions = torch.Tensor([1]).repeat(2, 3)
        expected = torch.Tensor([[-0.287682, -0.287682, -0.287682],
                                 [-0.287682, -0.287682, -0.287682]])
        obtained = compute_log_probability(M, actions)
        np.testing.assert_array_almost_equal(expected, obtained)


class TestLogProbability(unittest.TestCase):
    def test_log_probability_(self):
        M = td.categorical.Categorical(
            torch.Tensor([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]))
        actions = torch.Tensor([0]).repeat(3)
        expected = torch.Tensor([-1.38629436, -0.6931471, -0.287682])
        obtained = compute_log_probability(M, actions)
        np.testing.assert_array_almost_equal(expected, obtained)


if __name__ == '__main__':
    unittest.main()
