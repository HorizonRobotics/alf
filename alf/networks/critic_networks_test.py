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
"""Tests for alf.networks.value_networks."""

from absl.testing import parameterized
from absl import logging
import functools
import time
import torch

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks import CriticNetwork, CriticRNNNetwork
from alf.networks.network import NaiveParallelNetwork
from alf.networks.network_test import test_net_copy
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.nest.utils import NestConcat


class CriticNetworksTest(parameterized.TestCase, alf.test.TestCase):
    def _init(self, lstm_hidden_size):
        if lstm_hidden_size is not None:
            post_rnn_fc_layer_params = (6, 4)
            network_ctor = functools.partial(
                CriticRNNNetwork,
                lstm_hidden_size=lstm_hidden_size,
                critic_fc_layer_params=post_rnn_fc_layer_params)
            if isinstance(lstm_hidden_size, int):
                lstm_hidden_size = [lstm_hidden_size]
            state = []
            for size in lstm_hidden_size:
                state.append((torch.randn((
                    1,
                    size,
                ), dtype=torch.float32), ) * 2)
        else:
            network_ctor = CriticNetwork
            state = ()
        return network_ctor, state

    @parameterized.parameters((100, ), (None, ), ((200, 100), ))
    def test_critic(self, lstm_hidden_size):
        obs_spec = TensorSpec((3, 20, 20), torch.float32)
        action_spec = TensorSpec((5, ), torch.float32)
        input_spec = (obs_spec, action_spec)

        observation_conv_layer_params = ((8, 3, 1), (16, 3, 2, 1))
        action_fc_layer_params = (10, 8)
        joint_fc_layer_params = (6, 4)

        image = obs_spec.zeros(outer_dims=(2, ))
        action = action_spec.randn(outer_dims=(2, ))

        network_input = (image, action)

        network_ctor, state = self._init(lstm_hidden_size)

        critic_net = network_ctor(
            input_spec,
            observation_conv_layer_params=observation_conv_layer_params,
            action_fc_layer_params=action_fc_layer_params,
            joint_fc_layer_params=joint_fc_layer_params)
        test_net_copy(critic_net)

        value, state = critic_net._test_forward()
        self.assertEqual(value.shape, (2, ))
        if lstm_hidden_size is None:
            self.assertEqual(state, ())

        value, state = critic_net(network_input, state)

        self.assertEqual(critic_net.output_spec, TensorSpec(()))
        # (batch_size,)
        self.assertEqual(value.shape, (2, ))

        # test make_parallel
        pnet = critic_net.make_parallel(6)
        test_net_copy(pnet)

        if lstm_hidden_size is not None:
            # shape of state should be [B, n, ...]
            self.assertRaises(AssertionError, pnet, network_input, state)

        state = alf.nest.map_structure(
            lambda x: x.unsqueeze(1).expand(x.shape[0], 6, x.shape[1]), state)

        value, state = pnet(network_input, state)
        self.assertEqual(pnet.output_spec, TensorSpec((6, )))
        self.assertEqual(value.shape, (2, 6))

    def test_make_parallel(self):
        obs_spec = TensorSpec((20, ), torch.float32)
        action_spec = TensorSpec((5, ), torch.float32)
        critic_net = CriticNetwork((obs_spec, action_spec),
                                   joint_fc_layer_params=(256, 256))

        replicas = 4
        # ParallelCriticNetwork (PCN) is not always faster than NaiveParallelNetwork (NPN).
        # On my machine, for this particular example, with replicas=2,
        # PCN is faster when batch_size in (128, 256, ..., 2048)
        # NPN is faster when batch_size in (4096, 8192, 16384).
        # For a moderately large replicas (32), and smaller batch_size (128),
        # the speed difference is huge: PCN is 20 times faster than NPN.
        batch_size = 128

        def _train(pnet, name):
            t0 = time.time()
            optimizer = alf.optimizers.AdamTF(lr=1e-4)
            optimizer.add_param_group({'params': list(pnet.parameters())})
            for _ in range(100):
                obs = obs_spec.randn((batch_size, ))
                action = action_spec.randn((batch_size, ))
                values = pnet((obs, action))[0]
                target = torch.randn_like(values)
                cost = ((values - target)**2).sum()
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
            logging.info(
                "%s time=%s cost=%s" % (name, time.time() - t0, float(cost)))

        pnet = critic_net.make_parallel(replicas)
        _train(pnet, "ParallelCriticNetwork")

        pnet = NaiveParallelNetwork(critic_net, replicas)
        _train(pnet, "NaiveParallelNetwork")

    @parameterized.parameters((CriticNetwork, ), (CriticRNNNetwork, ))
    def test_discrete_action(self, net_ctor):
        obs_spec = TensorSpec((20, ))
        action_spec = BoundedTensorSpec((), dtype='int64')

        # doesn't support discrete action spec ...
        self.assertRaises(AssertionError, net_ctor, (obs_spec, action_spec))

        # ... unless an preprocessor is specified
        net_ctor((obs_spec, action_spec),
                 action_input_processors=EmbeddingPreprocessor(
                     action_spec, embedding_dim=10))

    @parameterized.parameters((CriticNetwork, ), (CriticRNNNetwork, ))
    def test_mixed_actions(self, net_ctor):
        obs_spec = TensorSpec((20, ))
        action_spec = dict(
            x=BoundedTensorSpec((), dtype='int64'), y=BoundedTensorSpec((3, )))

        input_preprocessors = dict(
            x=EmbeddingPreprocessor(action_spec['x'], embedding_dim=10),
            y=None)

        net_ctor = functools.partial(
            net_ctor, action_input_processors=input_preprocessors)

        # doesn't support mixed actions
        self.assertRaises(AssertionError, net_ctor, (obs_spec, action_spec))

        # ... unless a combiner is specified
        net_ctor((obs_spec, action_spec),
                 action_preprocessing_combiner=NestConcat())


if __name__ == "__main__":
    alf.test.main()
