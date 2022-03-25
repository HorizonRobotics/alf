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
"""Tests for alf.networks.q_networks."""

from absl.testing import parameterized
import unittest
import functools

import torch

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks import QNetwork
from alf.networks import QRNNNetwork
from alf.networks.q_networks import ParallelQNetwork
from alf.utils import common
from alf.nest.utils import NestSum


class TestQNetworks(parameterized.TestCase, unittest.TestCase):
    def _init(self, lstm_hidden_size):
        self._action_spec = BoundedTensorSpec((), torch.int64, 0, 2)
        self._num_actions = self._action_spec.maximum - self._action_spec.minimum + 1

        if lstm_hidden_size is not None:
            network_ctor = functools.partial(
                QRNNNetwork, lstm_hidden_size=lstm_hidden_size)
            if isinstance(lstm_hidden_size, int):
                lstm_hidden_size = [lstm_hidden_size]
            state = [()]
            for size in lstm_hidden_size:
                state.append((torch.randn((
                    1,
                    size,
                ), dtype=torch.float32), ) * 2)
        else:
            network_ctor = QNetwork
            state = ()
        return network_ctor, state

    @parameterized.parameters((100, ), (None, ), ((200, 100), ))
    def test_q_value_network(self, lstm_hidden_size):
        input_spec = [TensorSpec((3, 20, 20), torch.float32)]
        conv_layer_params = ((8, 3, 1), (16, 3, 2, 1))
        image = common.zero_tensor_from_nested_spec(input_spec, batch_size=1)

        network_ctor, state = self._init(lstm_hidden_size)

        q_net = network_ctor(
            input_spec,
            self._action_spec,
            input_preprocessors=[torch.relu],
            preprocessing_combiner=NestSum(),
            conv_layer_params=conv_layer_params)
        q_value, state = q_net(image, state)

        # (batch_size, num_actions)
        self.assertEqual(q_value.shape, (1, self._num_actions))

    def test_parallel_q_network(self):
        input_spec = TensorSpec([10])
        inputs = input_spec.zeros(outer_dims=(1, ))

        network_ctor, state = self._init(None)

        q_net = network_ctor(input_spec, self._action_spec)
        n = 5
        parallel_q_net = q_net.make_parallel(n)

        q_value, _ = parallel_q_net(inputs, state)

        # (batch_size, n, num_actions)
        self.assertEqual(q_value.shape, (1, n, self._num_actions))

    def test_rnn_parallel(self):
        input_spec = TensorSpec([10])
        batch_size = 1
        replicas = 2
        inputs = input_spec.zeros(outer_dims=(batch_size, ))

        network_ctor, state = self._init(100)
        state = alf.layers.make_parallel_input(state, replicas)

        q_net = network_ctor(
            input_spec, self._action_spec, input_preprocessors=torch.relu)
        pnet = q_net.make_parallel(replicas)

        self.assertTrue(isinstance(pnet, ParallelQNetwork))
        self.assertEqual(pnet.name, "parallel_" + q_net.name)
        self.assertEqual(
            pnet.state_spec,
            alf.nest.map_structure(
                functools.partial(TensorSpec.from_tensor, from_dim=1), state))

        q_value, _ = pnet(inputs, state)
        self.assertEqual(q_value.shape,
                         (batch_size, replicas, self._num_actions))


if __name__ == "__main__":
    unittest.main()
