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

from .encoding_networks import EncodingNetwork, LSTMEncodingNetwork
from .projection_networks import NormalProjectionNetwork, CategoricalProjectionNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from .network import Network


@gin.configurable
class ActorDistributionNetwork(Network):
    """Outputs temporally uncorrelated actions."""

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
        super(ActorDistributionNetwork, self).__init__(input_tensor_spec, (),
                                                       "")
        self._encoding_net = EncodingNetwork(
            input_tensor_spec, conv_layer_params, fc_layer_params, activation)

        def _create_projection_net(input_size):
            if action_spec.is_discrete:
                assert isinstance(action_spec, BoundedTensorSpec), \
                    "The action spec of discrete actions must be bounded!"
                self._projection_net = discrete_projection_net_ctor(
                    input_size=input_size, action_spec=action_spec)
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

    @property
    def state_spec(self):
        return ()


@gin.configurable
class ActorDistributionRNNNetwork(ActorDistributionNetwork):
    """Outputs temporally correlated actions."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 actor_fc_layer_params=None,
                 activation=torch.relu,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork):
        """Creates an instance of `ActorRNNDistributionNetwork`.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the action spec
            conv_layer_params (list[tuple]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (list[int]): a list of integers representing hidden
                FC layers for encoding the observation.
            lstm_hidden_size (int or list[int] or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
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
        super(ActorDistributionRNNNetwork, self).__init__(
            input_tensor_spec, action_spec, conv_layer_params, fc_layer_params,
            activation, discrete_projection_net_ctor,
            continuous_projection_net_ctor)

        self._lstm_encoding_net = LSTMEncodingNetwork(
            self._encoding_net.output_size, lstm_hidden_size,
            actor_fc_layer_params, activation)

        # overwrite the projection net with the second encoding net output size
        self._create_projection_net(self._lstm_encoding_net.output_size)

    def forward(self, observation, state):
        """Computes an action distribution given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state (nest[tuple]): a nest structure of state tuples (h, c)

        Returns:
            act_dist (torch.distributions): action distribution
            new_state (nest[tuple]): the updated states
        """
        encoding = self._encoding_net(observation)
        encoding, state = self._lstm_encoding_net(encoding, state)
        act_dist = self._projection_net(encoding)
        return act_dist, state

    @property
    def state_spec(self):
        return self._lstm_encoding_net.state_spec
