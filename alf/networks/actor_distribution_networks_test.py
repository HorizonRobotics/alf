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
import functools

import torch

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks import ActorDistributionNetwork
from alf.networks import ActorDistributionRNNNetwork
from alf.networks import NormalProjectionNetwork
from alf.utils.common import zero_tensor_from_nested_spec
from alf.nest.utils import NestConcat
from alf.utils.dist_utils import DistributionSpec


class TestActorDistributionNetworks(parameterized.TestCase, alf.test.TestCase):
    def setUp(self):
        self._input_spec = [
            TensorSpec((3, 20, 20), torch.float32),
            TensorSpec((1, 20, 20), torch.float32)
        ]
        self._image = zero_tensor_from_nested_spec(
            self._input_spec, batch_size=1)
        self._conv_layer_params = ((8, 3, 1), (16, 3, 2, 1))
        self._fc_layer_params = (100, )
        self._input_preprocessor_ctors = [torch.nn.Tanh, None]
        self._preprocessing_combiner = NestConcat(dim=1)

    def _init(self, lstm_hidden_size):
        if lstm_hidden_size is not None:
            network_ctor = functools.partial(
                ActorDistributionRNNNetwork,
                lstm_hidden_size=lstm_hidden_size,
                actor_fc_layer_params=(64, 32))
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

    @parameterized.parameters((100, ), (None, ), ((200, 100), ))
    def test_discrete_actor_distribution(self, lstm_hidden_size):
        action_spec = TensorSpec((), torch.int32)
        network_ctor, state = self._init(lstm_hidden_size)

        # action_spec is not bounded
        self.assertRaises(
            AssertionError,
            network_ctor,
            self._input_spec,
            action_spec,
            conv_layer_params=self._conv_layer_params)

        action_spec = BoundedTensorSpec((), torch.int32)
        actor_dist_net = network_ctor(
            self._input_spec,
            action_spec,
            input_preprocessor_ctors=self._input_preprocessor_ctors,
            preprocessing_combiner=self._preprocessing_combiner,
            conv_layer_params=self._conv_layer_params)

        act_dist, _ = actor_dist_net(self._image, state)
        actions = act_dist.sample((100, ))

        self.assertTrue(
            isinstance(actor_dist_net.output_spec, DistributionSpec))

        # (num_samples, batch_size)
        self.assertEqual(actions.shape, (100, 1))

        self.assertTrue(
            torch.all(actions >= torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(actions <= torch.as_tensor(action_spec.maximum)))

    @parameterized.parameters((100, ), (None, ), ((200, 100), ))
    def test_continuous_actor_distribution(self, lstm_hidden_size):
        action_spec = BoundedTensorSpec((3, ), torch.float32)

        network_ctor, state = self._init(lstm_hidden_size)

        actor_dist_net = network_ctor(
            self._input_spec,
            action_spec,
            input_preprocessor_ctors=self._input_preprocessor_ctors,
            preprocessing_combiner=self._preprocessing_combiner,
            conv_layer_params=self._conv_layer_params,
            continuous_projection_net_ctor=functools.partial(
                NormalProjectionNetwork, scale_distribution=True))
        act_dist, _ = actor_dist_net(self._image, state)
        actions = act_dist.sample((100, ))

        self.assertTrue(
            isinstance(actor_dist_net.output_spec, DistributionSpec))

        # (num_samples, batch_size, action_spec_shape)
        self.assertEqual(actions.shape, (100, 1) + action_spec.shape)

        self.assertTrue(
            torch.all(actions >= torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(actions <= torch.as_tensor(action_spec.maximum)))


if __name__ == "__main__":
    alf.test.main()
