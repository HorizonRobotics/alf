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
"""Tests for alf.networks.value_networks."""

from absl.testing import parameterized
import unittest
import functools

import torch

from alf.tensor_specs import TensorSpec
from alf.networks import ValueNetwork
from alf.networks import ValueRNNNetwork


class TestValueNetworks(parameterized.TestCase, unittest.TestCase):
    def _init(self, lstm_hidden_size):
        if lstm_hidden_size is not None:
            network_ctor = functools.partial(
                ValueRNNNetwork,
                lstm_hidden_size=lstm_hidden_size,
                value_fc_layer_params=[64, 32])
            state = (torch.randn((
                1,
                lstm_hidden_size,
            ), dtype=torch.float32), ) * 2
        else:
            network_ctor = ValueNetwork
            state = ()
        return network_ctor, state

    @parameterized.parameters((100, ), (None, ))
    def test_value_distribution(self, lstm_hidden_size):
        input_spec = TensorSpec((3, 20, 20), torch.float32)
        conv_layer_params = [(8, 3, 1), (16, 3, 2, 1)]

        image = input_spec.zeros(outer_dims=(1, ))

        network_ctor, state = self._init(lstm_hidden_size)

        value_net = network_ctor(
            input_spec, conv_layer_params=conv_layer_params)
        value, _ = value_net(image, state)

        # (batch_size,)
        self.assertEqual(value.shape, (1, ))


if __name__ == "__main__":
    unittest.main()
