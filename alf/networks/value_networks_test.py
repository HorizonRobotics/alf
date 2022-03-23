# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from absl import logging
import functools
import time
import torch

import alf
from alf.tensor_specs import TensorSpec
from alf.networks import ValueNetwork
from alf.networks import ValueRNNNetwork
from alf.networks.value_networks import ParallelValueNetwork
from alf.nest.utils import NestConcat
from alf.networks.preprocessors import EmbeddingPreprocessor


class TestValueNetworks(parameterized.TestCase, alf.test.TestCase):
    def _init(self, lstm_hidden_size):
        if lstm_hidden_size is not None:
            network_ctor = functools.partial(
                ValueRNNNetwork, lstm_hidden_size=lstm_hidden_size)
            if isinstance(lstm_hidden_size, int):
                lstm_hidden_size = [lstm_hidden_size]
            state = [()]
            for size in lstm_hidden_size:
                state.append((torch.randn((
                    1,
                    size,
                ), dtype=torch.float32), ) * 2)
            state.append(())
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

        self.assertEqual(value_net.output_spec, TensorSpec(()))
        # (batch_size,)
        self.assertEqual(value.shape, (1, ))

    def test_make_parallel(self):
        obs_spec = TensorSpec((20, ), torch.float32)
        value_net = ValueNetwork(obs_spec, fc_layer_params=(256, ))

        replicas = 4
        batch_size = 128

        def _train(pnet, name):
            t0 = time.time()
            optimizer = alf.optimizers.AdamTF(lr=1e-4)
            optimizer.add_param_group({'params': list(pnet.parameters())})
            for _ in range(100):
                obs = obs_spec.randn((batch_size, ))
                values = pnet(obs)[0]
                target = torch.randn_like(values)
                cost = ((values - target)**2).sum()
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
            logging.info(
                "%s time=%s cost=%s" % (name, time.time() - t0, float(cost)))

        pnet = value_net.make_parallel(replicas)
        _train(pnet, "ParallelValueNetwork")

        pnet = alf.networks.network.NaiveParallelNetwork(value_net, replicas)
        _train(pnet, "NaiveParallelNetwork")

    def test_rnn_make_parallel(self):
        obs_spec = TensorSpec((20, ), torch.float32)
        value_net = ValueRNNNetwork(
            obs_spec, fc_layer_params=(256, ), lstm_hidden_size=100)
        batch_size = 4
        state = [(), (torch.randn(
            (batch_size, 100), dtype=torch.float32), ) * 2, ()]
        replicas = 2

        pnet = value_net.make_parallel(replicas)
        state = alf.layers.make_parallel_input(state, replicas)
        self.assertTrue(isinstance(pnet, ParallelValueNetwork))
        self.assertEqual(pnet.name, "parallel_" + value_net.name)
        self.assertEqual(
            pnet.state_spec,
            alf.nest.map_structure(
                functools.partial(TensorSpec.from_tensor, from_dim=1), state))

        value, _ = pnet(obs_spec.randn((batch_size, replicas)), state)
        self.assertEqual(value.shape, (batch_size, replicas))


if __name__ == "__main__":
    alf.test.main()
