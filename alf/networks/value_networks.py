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

from alf.networks import EncodingNetwork
import alf.layers as layers
from alf.tensor_specs import TensorSpec


def _create_lstm_cell_state_spec(hidden_size, dtype):
    """Create LSTMCell state specs given the hidden size and dtype. According to
    PyTorch LSTMCell doc:

    https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell

    Each LSTMCell has two states: h and c with the same shape.

    Args:
        hidden_size (int): the number of units in the hidden state
        dtype (torch.dtype): dtype of the specs

    Returns:
        specs (tuple[TensorSpec]):
    """
    state_spec = TensorSpec(shape=(hidden_size, ), dtype=dtype)
    return (state_spec, state_spec)


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
            conv_layer_params (list[tuple]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (list[int]): a list of integers representing hidden
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


@gin.configurable
class ValueRNNNetwork(nn.Module):
    """Outputs temporally correlated values."""

    def __init__(self,
                 input_tensor_spec,
                 lstm_hidden_size,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 value_fc_layer_params=None,
                 activation=torch.relu):
        """Creates an instance of `ValueRNNNetwork`.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            lstm_hidden_size (int): the hidden size of the LSTM cell
            conv_layer_params (list[tuple]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (list[int]): a list of integers representing hidden
                FC layers for encoding the observation.
            value_fc_layer_params (list[int]): a list of integers representing hidden
                FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
        """
        super(ValueRNNNetwork, self).__init__()
        self._before_lstm_encoding_net = EncodingNetwork(
            input_tensor_spec, conv_layer_params, fc_layer_params, activation)

        self._lstm_cell = torch.nn.LSTMCell(
            input_size=self._before_lstm_encoding_net.output_size,
            hidden_size=lstm_hidden_size)
        self._state_spec = _create_lstm_cell_state_spec(
            lstm_hidden_size, input_tensor_spec.dtype)

        input_tensor_spec = TensorSpec((lstm_hidden_size, ),
                                       input_tensor_spec.dtype)
        self._after_lstm_encoding_net = EncodingNetwork(
            input_tensor_spec,
            fc_layer_params=value_fc_layer_params,
            activation=activation,
            last_layer_size=1,
            last_activation=layers.identity)

    def forward(self, observation, state):
        """Computes a value given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state (tuple): a state tuple (h, c)

        Returns:
            value (torch.Tensor): a 1D tensor
            state (tuple): the updated states
        """
        assert isinstance(state, tuple) and len(state) == 2, \
            "The LSTMCell state should be a tuple of (h,c)!"
        encoding = self._before_lstm_encoding_net(observation)
        h_state, c_state = self._lstm_cell(encoding, state)
        value = self._after_lstm_encoding_net(h_state)
        return torch.squeeze(value, -1), (h_state, c_state)

    @property
    def state_spec(self):
        return self._state_spec
