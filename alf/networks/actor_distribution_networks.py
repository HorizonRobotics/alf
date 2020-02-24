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
"""ActorDistributionNetwork and ActorRNNDistributionNetwork."""

import gin

import torch
import torch.nn as nn

from alf.networks import EncodingNetwork
from alf.networks import NormalProjectionNetwork, CategoricalProjectionNetwork
from alf.layers import _create_lstm_cell_state_spec
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


@gin.configurable
class ActorDistributionNetwork(nn.Module):
    """Outputs temporally correlated actions."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork):
        """Creates an instance of `ActorDistributionNetwork`.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the action spec
            conv_layer_params (list[tuple]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (list[int]): a list of integers representing hidden
                FC layer sizes.
            activation (nn.functional): activation used for hidden layers.
            discrete_projection_net_ctor (ProjectionNetwork): constructor that
                generates a discrete projection network that outputs discrete
                actions.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                continuous actions.
        """
        super(ActorDistributionNetwork, self).__init__()
        self._encoding_net = EncodingNetwork(
            input_tensor_spec, conv_layer_params, fc_layer_params, activation)

        def _create_projection_net(input_size):
            if action_spec.is_discrete:
                assert isinstance(action_spec, BoundedTensorSpec), \
                    "The action spec of discrete actions must be bounded!"
                self._projection_net = discrete_projection_net_ctor(
                    input_size=input_size,
                    num_actions=action_spec.maximum - action_spec.minimum + 1)
            else:
                self._projection_net = continuous_projection_net_ctor(
                    input_size=input_size, action_spec=action_spec)

        self._create_projection_net = _create_projection_net
        self._create_projection_net(self._encoding_net.output_size)

    def forward(self, observation, state=()):
        """Computes an action distribution given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state: empty for API consistent with ActorRNNDistributionNetwork

        Returns:
            act_dist (torch.distributions): action distribution
            state: empty
        """
        act_dist = self._projection_net(self._encoding_net(observation))
        return act_dist, state


@gin.configurable
class ActorRNNDistributionNetwork(ActorDistributionNetwork):
    """Outputs temporally uncorrelated actions."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 lstm_hidden_size,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 actor_fc_layer_params=None,
                 activation=torch.relu,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork):
        """Creates an instance of `ActorRNNDistributionNetwork`.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the action spec
            lstm_hidden_size (int): the hidden size of the LSTM cell
            conv_layer_params (list[tuple]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (list[int]): a list of integers representing hidden
                FC layers for encoding the observation.
            actor_fc_layer_params (list[int]): a list of integers representing hidden
                FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers.
            discrete_projection_net_ctor (ProjectionNetwork): constructor that
                generates a discrete projection network that outputs discrete
                actions.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                continuous actions.
        """
        super(ActorRNNDistributionNetwork, self).__init__(
            input_tensor_spec, action_spec, conv_layer_params, fc_layer_params,
            activation, discrete_projection_net_ctor,
            continuous_projection_net_ctor)

        self._lstm_cell = torch.nn.LSTMCell(
            input_size=self._encoding_net.output_size,
            hidden_size=lstm_hidden_size)
        self._state_spec = _create_lstm_cell_state_spec(
            lstm_hidden_size, input_tensor_spec.dtype)

        input_tensor_spec = TensorSpec((lstm_hidden_size, ),
                                       input_tensor_spec.dtype)
        self._after_lstm_encoding_net = EncodingNetwork(
            input_tensor_spec,
            fc_layer_params=actor_fc_layer_params,
            activation=activation)
        # overwrite the projection net with the second encoding net output size
        self._create_projection_net(self._after_lstm_encoding_net.output_size)

    def forward(self, observation, state):
        """Computes an action distribution given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state (tuple): a state tuple (h, c)

        Returns:
            act_dist (torch.distributions): action distribution
            state (tuple): the updated states
        """
        assert isinstance(state, tuple) and len(state) == 2, \
            "The LSTMCell state should be a tuple of (h,c)!"
        encoding = self._encoding_net(observation)
        h_state, c_state = self._lstm_cell(encoding, state)
        act_dist = self._projection_net(self._after_lstm_encoding_net(h_state))
        return act_dist, (h_state, c_state)

    @property
    def state_spec(self):
        return self._state_spec
