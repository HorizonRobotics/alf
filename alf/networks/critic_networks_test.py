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
from alf.networks import CriticNetwork
from alf.networks import CriticRNNNetwork


class TestCriticNetworks(parameterized.TestCase, alf.test.TestCase):
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

        image = obs_spec.zeros(outer_dims=(1, ))
        action = action_spec.randn(outer_dims=(1, ))

        network_input = (image, action)

        network_ctor, state = self._init(lstm_hidden_size)

        critic_net = network_ctor(
            input_spec,
            observation_conv_layer_params=observation_conv_layer_params,
            action_fc_layer_params=action_fc_layer_params,
            joint_fc_layer_params=joint_fc_layer_params)

        value, state = critic_net._test_forward()
        self.assertEqual(value.shape, (1, ))
        if lstm_hidden_size is None:
            self.assertEqual(state, ())

        value, state = critic_net(network_input, state)

        self.assertEqual(critic_net.output_spec, TensorSpec(()))
        # (batch_size,)
        self.assertEqual(value.shape, (1, ))


if __name__ == "__main__":
    alf.test.main()
