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
"""Unittests for action sampling"""

import unittest
import torch
import numpy as np
from alf.utils.common import epsilon_greedy_sample


class TestActionSamplingCategorical(unittest.TestCase):
    def test_action_sampling_categorical(self):
        m = torch.distributions.categorical.Categorical(
            torch.Tensor([0.25, 0.75]))
        M = m.expand([10])
        epsilon = 0.0
        action_expected = torch.Tensor([1]).repeat(10)
        action_obtained = epsilon_greedy_sample(M, epsilon)
        self.assertTrue((action_expected == action_obtained).all())


class TestActionSamplingNormal(unittest.TestCase):
    def test_action_sampling_normal(self):
        m = torch.distributions.normal.Normal(
            torch.Tensor([0.3, 0.7]), torch.Tensor([1.0, 1.0]))
        M = m.expand([10, 2])
        epsilon = 0.0
        action_expected = torch.Tensor([0.3, 0.7]).repeat(10, 1)
        action_obtained = epsilon_greedy_sample(M, epsilon)
        self.assertTrue((action_expected == action_obtained).all())
