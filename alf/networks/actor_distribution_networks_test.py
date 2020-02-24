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
"""Tests for alf.networks.actor_distribution_networks."""

from absl.testing import parameterized
import unittest
import functools

import torch

from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks import ActorDistributionNetwork
from alf.networks import ActorRNNDistributionNetwork
from alf.networks import NormalProjectionNetwork


class TestActorDistributionNetworks(parameterized.TestCase, unittest.TestCase):
    def _init(self, lstm_hidden_size):
        if lstm_hidden_size is not None:
            network_ctor = functools.partial(
                ActorRNNDistributionNetwork,
                lstm_hidden_size=lstm_hidden_size,
                actor_fc_layer_params=[64, 32])
            if isinstance(lstm_hidden_size, int):
                lstm_hidden_size = [lstm_hidden_size]
            state = []
            for size in lstm_hidden_size:
                state.append((torch.randn((
                    1,
                    size,
                ), dtype=torch.float32), ) * 2)
        else:
            network_ctor = ActorDistributionNetwork
            state = ()
        return network_ctor, state

    @parameterized.parameters((100, ), (None, ), ([200, 100], ))
    def test_discrete_actor_distribution(self, lstm_hidden_size):
        input_spec = TensorSpec((3, 20, 20), torch.float32)
        action_spec = TensorSpec((), torch.int32)
        conv_layer_params = [(8, 3, 1), (16, 3, 2, 1)]

        image = input_spec.zeros(outer_dims=(1, ))

        network_ctor, state = self._init(lstm_hidden_size)

        # action_spec is not bounded
        self.assertRaises(
            AssertionError,
            network_ctor,
            input_spec,
            action_spec,
            conv_layer_params=conv_layer_params)

        action_spec = BoundedTensorSpec((), torch.int32)
        actor_dist_net = network_ctor(
            input_spec, action_spec, conv_layer_params=conv_layer_params)
        act_dist, _ = actor_dist_net(image, state)
        actions = act_dist.sample((100, ))

        # (num_samples, batch_size)
        self.assertEqual(actions.shape, (100, 1))

        self.assertTrue(
            torch.all(actions >= torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(actions <= torch.as_tensor(action_spec.maximum)))

    @parameterized.parameters((100, ), (None, ), ([200, 100], ))
    def test_continuous_actor_distribution(self, lstm_hidden_size):
        input_spec = TensorSpec((3, 20, 20), torch.float32)
        action_spec = BoundedTensorSpec((3, ), torch.float32)
        conv_layer_params = [(8, 3, 1), (16, 3, 2, 1)]

        image = input_spec.zeros(outer_dims=(1, ))

        network_ctor, state = self._init(lstm_hidden_size)

        actor_dist_net = network_ctor(
            input_spec,
            action_spec,
            conv_layer_params=conv_layer_params,
            continuous_projection_net_ctor=functools.partial(
                NormalProjectionNetwork, scale_distribution=True))
        act_dist, _ = actor_dist_net(image, state)
        actions = act_dist.sample((100, ))

        # (num_samples, batch_size, action_spec_shape)
        self.assertEqual(actions.shape, (100, 1) + action_spec.shape)

        self.assertTrue(
            torch.all(actions >= torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(actions <= torch.as_tensor(action_spec.maximum)))


if __name__ == "__main__":
    unittest.main()
