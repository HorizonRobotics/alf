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
"""ValueNetwork and ValueRNNNetwork."""

import gin

import torch
import torch.nn as nn

from alf.networks import EncodingNetwork, LSTMEncodingNetwork
import alf.layers as layers
from alf.tensor_specs import TensorSpec


@gin.configurable
class ValueNetwork(nn.Module):
    """Output temporally uncorrelated values."""

    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu):
        """Creates a value network that estimates the expected return.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (tuple[int]): a list of integers representing hidden
                FC layer sizes.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
        """
        super(ValueNetwork, self).__init__()
        self._encoding_net = EncodingNetwork(
            input_tensor_spec,
            conv_layer_params,
            fc_layer_params,
            activation,
            last_layer_size=1,
            last_activation=layers.identity)

    def forward(self, observation, state=()):
        """Computes a value given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state: empty for API consistent with ValueRNNNetwork

        Returns:
            value (torch.Tensor): a 1D tensor
            state: empty
        """
        value = self._encoding_net(observation)
        return torch.squeeze(value, -1), state

    @property
    def state_spec(self):
        return ()


@gin.configurable
class ValueRNNNetwork(nn.Module):
    """Outputs temporally correlated values."""

    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 value_fc_layer_params=None,
                 activation=torch.relu):
        """Creates an instance of `ValueRNNNetwork`.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (tuple[int]): a list of integers representing hidden
                FC layers for encoding the observation.
            lstm_hidden_size (int or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            value_fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
        """
        super(ValueRNNNetwork, self).__init__()
        self._encoding_net = EncodingNetwork(
            input_tensor_spec, conv_layer_params, fc_layer_params, activation)
        self._lstm_encoding_net = LSTMEncodingNetwork(
            self._encoding_net.output_size,
            lstm_hidden_size,
            value_fc_layer_params,
            activation,
            last_layer_size=1,
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
        value, state = self._lstm_encoding_net(encoding, state)
        return torch.squeeze(value, -1), state

    @property
    def state_spec(self):
        return self._lstm_encoding_net.state_spec
