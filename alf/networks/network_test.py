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
"""Tests for alf.networks.network."""

from absl.testing import parameterized
from functools import partial

import torch
import torch.nn as nn

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.initializers import _numerical_calculate_gain
from alf.initializers import _calculate_gain
from alf.networks import (ActorNetwork, ActorRNNNetwork, EncodingNetwork,
                          LSTMEncodingNetwork, PreprocessorNetwork,
                          TransformerNetwork, ValueNetwork, ValueRNNNetwork,
                          BetaProjectionNetwork)
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.networks.network import NaiveParallelNetwork


def test_net_copy(net):
    """Test whether net.copy() is correctly implemented"""
    new_net = net.copy()
    params = dict(net.named_parameters())
    new_params = dict(new_net.named_parameters())
    for n, p in new_params.items():
        assert p.shape == params[n].shape, (
            "The shape of the parameter of the "
            "copied network is different from that of the original network: "
            " %s vs %s" % (p.shape, params[n].shape))
        assert id(p) != id(
            params[n]), ("The parameter of the copied parameter "
                         "is the same parameter of the original network")


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


class InitializerTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((torch.relu), (alf.utils.math_ops.identity, ),
                              (torch.tanh, ), (torch.sigmoid, ),
                              (torch.nn.functional.elu, ),
                              (torch.nn.functional.leaky_relu, ))
    def test_numerical_calculate_gain(self, activation):
        numerical_gain = _numerical_calculate_gain(activation)
        if activation.__name__ == "identity":
            gain = _calculate_gain("linear")
        else:
            gain = _calculate_gain(activation.__name__)
        print(activation.__name__, numerical_gain, gain)
        self.assertLess(abs(numerical_gain - gain), 0.1)


class NaiveParallelNetworkTest(alf.test.TestCase):
    def test_non_rnn(self):
        input_spec = TensorSpec((100, ), torch.float32)
        embedding = input_spec.zeros(outer_dims=(6, ))

        network = EncodingNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=(30, 40, 50),
            activation=torch.tanh)
        replicas = 4
        num_layers = 3

        pnet = NaiveParallelNetwork(network, replicas)

        self.assertEqual(
            len(list(pnet.parameters())), num_layers * 2 * replicas)

        output, _ = pnet(embedding)
        self.assertEqual(output.shape, (6, replicas, 50))
        self.assertEqual(pnet.output_spec.shape, (replicas, 50))

    def test_rnn(self):
        input_spec = TensorSpec((100, ), torch.float32)
        embedding = input_spec.zeros(outer_dims=(6, ))

        network = LSTMEncodingNetwork(
            input_tensor_spec=input_spec, hidden_size=(30, 40))
        replicas = 4
        pnet = NaiveParallelNetwork(network, replicas)

        self.assertEqual(pnet.state_spec,
                         [(TensorSpec((4, 30)), TensorSpec((4, 30))),
                          (TensorSpec((4, 40)), TensorSpec((4, 40)))])
        state = alf.utils.common.zero_tensor_from_nested_spec(
            pnet.state_spec, 6)
        output, state = pnet(embedding, state)
        self.assertEqual(output.shape, (6, replicas, 40))
        self.assertEqual(pnet.output_spec.shape, (replicas, 40))
        self.assertEqual(
            alf.utils.dist_utils.extract_spec(state),
            [(TensorSpec((4, 30)), TensorSpec((4, 30))),
             (TensorSpec((4, 40)), TensorSpec((4, 40)))])

    def test_distribution(self):
        input_size = 100
        action_spec = BoundedTensorSpec((4, ))
        network = BetaProjectionNetwork(
            input_size=input_size, action_spec=action_spec)
        pnet = NaiveParallelNetwork(network, 2)
        x = torch.zeros((
            5,
            input_size,
        ))

        dist, _ = pnet(x)
        self.assertEqual(dist.event_shape, action_spec.shape)
        self.assertEqual(dist.batch_shape, (5, 2))


class NetworkWrapperTest(alf.test.TestCase):
    def test_network_wrapper(self):
        input_spec = TensorSpec((100, ), torch.float32)
        net = alf.networks.NetworkWrapper(
            lambda input: input + 1., input_tensor_spec=input_spec)
        self.assertTensorEqual(net._test_forward()[0], input_spec.ones((2, )))

        net1 = alf.networks.NetworkWrapper(
            lambda input, state: (input + state, state + 1.),
            input_tensor_spec=input_spec,
            state_spec=input_spec)
        output, new_state = net1._test_forward()
        self.assertTensorEqual(output, input_spec.zeros((2, )))
        self.assertTensorEqual(new_state, input_spec.ones((2, )))

        pnet1 = net1.make_parallel(n=4)
        self.assertEqual(pnet1.state_spec,
                         TensorSpec((4, ) + input_spec.shape))


class PreprocessorNetworkTest(alf.test.TestCase):
    def test_stateless_preprocessors(self):
        input_spec = TensorSpec((100, ), torch.float32)
        action_spec = TensorSpec((5, ), torch.float32)
        combiner = alf.nest.utils.NestConcat()

        PreprocessorNetwork(
            input_tensor_spec=input_spec,
            input_preprocessors=EmbeddingPreprocessor(
                input_spec, embedding_dim=10),
            preprocessing_combiner=combiner)
        PreprocessorNetwork(
            input_tensor_spec=input_spec,
            input_preprocessors=alf.layers.Reshape(-1),
            preprocessing_combiner=combiner)
        self.assertRaises(
            AssertionError,
            PreprocessorNetwork,
            input_tensor_spec=input_spec,
            input_preprocessors=LSTMEncodingNetwork(
                input_tensor_spec=input_spec),
            preprocessing_combiner=combiner)

        def _create_transformer_net(preprocessor):
            return TransformerNetwork(
                input_tensor_spec=input_spec,
                num_prememory_layers=2,
                num_attention_heads=5,
                d_ff=100,
                input_preprocessors=preprocessor)

        _create_transformer_net(
            alf.nn.Sequential(
                EmbeddingPreprocessor(input_spec, embedding_dim=100),
                alf.layers.Reshape(10, 10)))
        self.assertRaises(
            AssertionError, _create_transformer_net,
            alf.nn.Sequential(
                LSTMEncodingNetwork(
                    input_tensor_spec=input_spec, hidden_size=(100, )),
                alf.layers.Reshape(10, 10)))


if __name__ == '__main__':
    alf.test.main()
