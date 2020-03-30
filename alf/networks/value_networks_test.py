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
import functools

import torch

import alf
from alf.tensor_specs import TensorSpec
from alf.networks import ValueNetwork
from alf.networks import ValueRNNNetwork
from alf.nest.utils import NestConcat
from alf.networks.preprocessors import EmbeddingPreprocessor


class TestValueNetworks(parameterized.TestCase, alf.test.TestCase):
    def _init(self, lstm_hidden_size):
        if lstm_hidden_size is not None:
            network_ctor = functools.partial(
                ValueRNNNetwork, lstm_hidden_size=lstm_hidden_size)
            if isinstance(lstm_hidden_size, int):
                lstm_hidden_size = [lstm_hidden_size]
            state = []
            for size in lstm_hidden_size:
                state.append((torch.randn((
                    1,
                    size,
                ), dtype=torch.float32), ) * 2)
        else:
            network_ctor = ValueNetwork
            state = ()
        return network_ctor, state

    @parameterized.parameters((100, ), (None, ), ((200, 100), ))
    def test_value_distribution(self, lstm_hidden_size):
        input_spec1 = TensorSpec((3, 20, 20))
        input_spec2 = TensorSpec((100, ))
        conv_layer_params = ((8, 3, 1), (16, 3, 2, 1))
        embedding_dim = 100

        image = input_spec1.zeros(outer_dims=(1, ))
        vector = input_spec2.zeros(outer_dims=(1, ))

        network_ctor, state = self._init(lstm_hidden_size)

        value_net = network_ctor(
            input_tensor_spec=[input_spec1, input_spec2],
            input_preprocessors=[
                EmbeddingPreprocessor(
                    input_spec1,
                    embedding_dim=embedding_dim,
                    conv_layer_params=conv_layer_params), None
            ],
            preprocessing_combiner=NestConcat())

        value, state = value_net([image, vector], state)

        self.assertEqual(value_net._processed_input_tensor_spec.shape[0], 200)
        self.assertEqual(value_net.output_spec, TensorSpec(()))
        # (batch_size,)
        self.assertEqual(value.shape, (1, ))


if __name__ == "__main__":
    alf.test.main()
