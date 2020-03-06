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
"""Tests for alf.networks.network."""
import torch
import torch.nn as nn

import alf
from alf.tensor_specs import TensorSpec


class BaseNetwork(alf.networks.Network):
    def __init__(self, v1, **kwargs):
        super().__init__(v1, **kwargs)


class MockNetwork(BaseNetwork):
    def __init__(self, param1, param2, kwarg1=2, kwarg2=3):
        self.param1 = param1
        self.param2 = param2
        self.kwarg1 = kwarg1
        self.kwarg2 = kwarg2

        super().__init__(param1, name='mock')

        self.var1 = nn.Parameter(torch.tensor(1., requires_grad=False))
        self.var2 = nn.Parameter(torch.tensor(2., requires_grad=True))

    def forward(self, observations, network_state=None):
        return self.var1 + self.var2


class NoInitNetwork(MockNetwork):
    pass


class NetworkTest(alf.test.TestCase):
    def test_copy_works(self):
        # pass a TensorSpec to prevent assertion error in Network
        network1 = MockNetwork(TensorSpec([2]), 1)
        network2 = network1.copy()

        self.assertNotEqual(network1, network2)
        self.assertEqual(TensorSpec([2]), network2.param1)
        self.assertEqual(1, network2.param2)
        self.assertEqual(2, network2.kwarg1)
        self.assertEqual(3, network2.kwarg2)

    def test_noinit_copy_works(self):
        # pass a TensorSpec to prevent assertion error in Network
        network1 = NoInitNetwork(TensorSpec([2]), 1)
        network2 = network1.copy()

        self.assertNotEqual(network1, network2)
        self.assertEqual(TensorSpec([2]), network2.param1)
        self.assertEqual(1, network2.param2)
        self.assertEqual(2, network2.kwarg1)
        self.assertEqual(3, network2.kwarg2)

    def test_too_many_args_raises_appropriate_error(self):
        self.assertRaises(TypeError, MockNetwork, 0, 1, 2, 3, 4, 5, 6)


if __name__ == '__main__':
    alf.test.main()
