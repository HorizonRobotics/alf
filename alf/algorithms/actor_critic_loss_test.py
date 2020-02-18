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
"""Unittests for actor_critic_loss.py"""

import unittest
import torch
import numpy as np
from alf.algorithms.actor_critic_loss import _normalize_advantages


class TestAdvantageNormalization(unittest.TestCase):
    def test_advantage_normalization(self):
        advantages = torch.Tensor([[1, 2], [3, 4.0]])
        # results computed from tf
        normalized_advantages_expected = torch.Tensor(
            [[-1.3416407, -0.4472136], [0.4472136, 1.3416407]])
        normalized_advantages_obtained = _normalize_advantages(advantages)
        np.testing.assert_array_almost_equal(normalized_advantages_obtained,
                                             normalized_advantages_expected)
