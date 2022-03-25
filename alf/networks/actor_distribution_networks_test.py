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
"""Tests for alf.networks.actor_distribution_networks."""

from alf.networks.actor_distribution_networks import ParallelActorDistributionNetwork
from absl.testing import parameterized
import functools
from absl import logging
import time
import torch
import torch.distributions as td

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks import ActorDistributionNetwork
from alf.networks import ActorDistributionRNNNetwork
from alf.networks import NormalProjectionNetwork, CategoricalProjectionNetwork
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
        self._input_preprocessors = [torch.tanh, None]
        self._preprocessing_combiner = NestConcat(dim=0)

    def _init(self, lstm_hidden_size):
        if lstm_hidden_size is not None:
            network_ctor = functools.partial(
                ActorDistributionRNNNetwork,
                lstm_hidden_size=lstm_hidden_size,
                actor_fc_layer_params=(64, 32))
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
            input_preprocessors=self._input_preprocessors,
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
            input_preprocessors=self._input_preprocessors,
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

    @parameterized.parameters(((200, 100), ), (None, ))
    def test_mixed_actor_distributions(self, lstm_hidden_size):
        action_spec = dict(
            discrete=BoundedTensorSpec((), dtype="int64"),
            continuous=BoundedTensorSpec((3, )))

        network_ctor, state = self._init(lstm_hidden_size)

        actor_dist_net = network_ctor(
            self._input_spec,
            action_spec,
            input_preprocessors=self._input_preprocessors,
            preprocessing_combiner=self._preprocessing_combiner,
            conv_layer_params=self._conv_layer_params)

        act_dist, state = actor_dist_net(self._image, state)

        self.assertTrue(
            isinstance(actor_dist_net.output_spec["discrete"],
                       DistributionSpec))
        self.assertTrue(
            isinstance(actor_dist_net.output_spec["continuous"],
                       DistributionSpec))

        self.assertTrue(isinstance(act_dist["discrete"], td.Categorical))
        self.assertTrue(
            isinstance(act_dist["continuous"].base_dist, td.Normal))

        if lstm_hidden_size is None:
            self.assertEqual(state, ())
        else:
            self.assertEqual(
                len(alf.nest.flatten(state)), 2 * len(lstm_hidden_size))

    def test_make_parallel(self):
        obs_spec = TensorSpec((3, 20, 20), torch.float32)
        network_ctor, _ = self._init(None)
        replicas = 4
        batch_size = 128

        # test continuous action
        action_spec = BoundedTensorSpec((3, ), torch.float32)
        actor_dist_net = network_ctor(
            obs_spec,
            action_spec,
            conv_layer_params=self._conv_layer_params,
            fc_layer_params=self._fc_layer_params,
            continuous_projection_net_ctor=functools.partial(
                NormalProjectionNetwork, scale_distribution=True))

        pnet = actor_dist_net.make_parallel(replicas)
        self.assertTrue(isinstance(pnet, ParallelActorDistributionNetwork))
        self.assertEqual(pnet.name, "parallel_" + actor_dist_net.name)
        self.assertTrue(
            isinstance(actor_dist_net.output_spec, DistributionSpec))
        act_dist, _ = pnet(obs_spec.randn((batch_size, )))
        actions = act_dist.sample()
        self.assertEqual(actions.shape,
                         (batch_size, replicas) + action_spec.shape)
        self.assertTrue(
            torch.all(actions >= torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(actions <= torch.as_tensor(action_spec.maximum)))

        # test discrete action

        action_spec = TensorSpec((), torch.int32)
        # action_spec is not bounded
        self.assertRaises(
            AttributeError,
            network_ctor,
            obs_spec,
            action_spec,
            conv_layer_params=self._conv_layer_params)

        action_spec = BoundedTensorSpec((), torch.int32)
        actor_dist_net = network_ctor(
            obs_spec, action_spec, conv_layer_params=self._conv_layer_params)

        pnet = actor_dist_net.make_parallel(replicas)
        act_dist, _ = pnet(obs_spec.randn((batch_size, )))
        actions = act_dist.sample()
        self.assertEqual(actions.shape,
                         (batch_size, replicas) + action_spec.shape)
        self.assertTrue(
            torch.all(actions >= torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(actions <= torch.as_tensor(action_spec.maximum)))

    def test_rnn_make_parallel(self):
        obs_spec = TensorSpec((3, 20, 20), torch.float32)
        network_ctor, state = self._init(100)
        replicas = 2

        action_spec = BoundedTensorSpec((3, ), torch.float32)
        actor_dist_net = network_ctor(
            obs_spec,
            action_spec,
            conv_layer_params=self._conv_layer_params,
            fc_layer_params=self._fc_layer_params,
            continuous_projection_net_ctor=functools.partial(
                NormalProjectionNetwork, scale_distribution=True))
        pnet = actor_dist_net.make_parallel(replicas)
        state = alf.layers.make_parallel_input(state, replicas)
        self.assertTrue(isinstance(pnet, ParallelActorDistributionNetwork))
        self.assertEqual(pnet.name, "parallel_" + actor_dist_net.name)
        self.assertEqual(
            pnet.state_spec,
            alf.nest.map_structure(
                functools.partial(TensorSpec.from_tensor, from_dim=1), state))

        act_dist, _ = pnet(obs_spec.randn((1, replicas)), state)
        actions = act_dist.sample()
        self.assertEqual(actions.shape, (1, replicas) + action_spec.shape)


if __name__ == "__main__":
    alf.test.main()
