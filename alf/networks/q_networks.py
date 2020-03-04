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
"""QNetworks"""

import gin

import torch
import torch.nn as nn

from alf.networks import EncodingNetwork, LSTMEncodingNetwork
import alf.layers as layers
from alf.tensor_specs import TensorSpec, BoundedTensorSpec


@gin.configurable
class QNetwork(nn.Module):
    """Create a q-network."""

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu):
        """Creates a q-network.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the tensor spec of the action
        actions.
            conv_layer_params (list[tuple]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (list[int]): a list of integers representing hidden
                FC layer sizes.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
        """

        num_actions = action_spec.maximum - action_spec.minimum + 1

        super(QNetwork, self).__init__()
        self._encoding_net = EncodingNetwork(
            input_tensor_spec,
            conv_layer_params,
            fc_layer_params,
            activation,
            last_layer_size=num_actions,
            last_activation=layers.identity)

    def forward(self, observation, state=()):
        """Computes a value given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state: empty for API consistent with ValueRNNNetwork

        Returns:
            q_value (torch.Tensor): a tensor of the size (num_actions,)
            state: empty
        """
        value = self._encoding_net(observation)
        return torch.squeeze(value, -1), state

    @property
    def state_spec(self):
        return ()


@gin.configurable
class QRNNNetwork(nn.Module):
    """Create a RNN-based that outputs temporally correlated q-values."""

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 post_rnn_fc_layer_params=None,
                 activation=torch.relu):
        """Creates an instance of `QRNNNetwork`.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the tensor spec of the action
        actions.
            conv_layer_params (list[tuple]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (list[int]): a list of integers representing hidden
                FC layers for encoding the observation.
            lstm_hidden_size (int or list[int] or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            post_rnn_fc_layer_params (list[int]): a list of integers representing hidden
                FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
        """

        num_actions = action_spec.maximum - action_spec.minimum + 1

        super(QRNNNetwork, self).__init__()
        self._encoding_net = EncodingNetwork(
            input_tensor_spec, conv_layer_params, fc_layer_params, activation)
        self._lstm_encoding_net = LSTMEncodingNetwork(
            self._encoding_net.output_size,
            lstm_hidden_size,
            post_rnn_fc_layer_params,
            activation,
            last_layer_size=num_actions,
            last_activation=layers.identity)

    def forward(self, observation, state):
        """Computes a value given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state (nest[tuple]): a nest structure of state tuples (h, c)

        Returns:
            value (torch.Tensor): a 1D tensor
            new_state (nest[tuple]): the updated states
        """
        encoding = self._encoding_net(observation)
        q_value, state = self._lstm_encoding_net(encoding, state)
        return torch.squeeze(q_value, -1), state

    @property
    def state_spec(self):
        return self._lstm_encoding_net.state_spec
