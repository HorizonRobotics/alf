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

import alf.layers as layers
from alf.networks import EncodingNetwork, LSTMEncodingNetwork
from alf.networks import Network
from alf.tensor_specs import TensorSpec, BoundedTensorSpec


@gin.configurable
class QNetwork(Network):
    """Create an instance of QNetwork."""

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu,
                 name="QNetwork"):
        """Creates an instance of `QNetwork` for estimating action-value of
        discrete actions. The action-value is defined as the expected return
        starting from the given input observation and taking the given action.
        It takes observation as input and outputs an action-value tensor with
        the shape of [batch_size, num_of_actions].

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the tensor spec of the action
            input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with `input_tensor_spec` (after reshaping).
                If any element is None, then it will be treated as alf.layers.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layer sizes.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
        """
        super(QNetwork, self).__init__(
            input_tensor_spec,
            input_preprocessors,
            preprocessing_combiner,
            name=name)

        num_actions = action_spec.maximum - action_spec.minimum + 1
        self._output_spec = TensorSpec((num_actions, ))

        self._encoding_net = EncodingNetwork(
            input_tensor_spec=self._processed_input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            last_layer_size=num_actions,
            last_activation=layers.identity)

    def forward(self, observation, state=()):
        """Computes action values given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state: empty for API consistent with QRNNNetwork

        Returns:
            action_value (torch.Tensor): a tensor of the size [batch_size, num_actions]
            state: empty
        """
        observation, state = Network.forward(self, observation, state)
        action_value, _ = self._encoding_net(observation)
        return action_value, state


@gin.configurable
class QRNNNetwork(Network):
    """Create a RNN-based that outputs temporally correlated q-values."""

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 value_fc_layer_params=None,
                 activation=torch.relu,
                 name="QRNNNetwork"):
        """Creates an instance of `QRNNNetwork` for estimating action-value of
        discrete actions. The action-value is defined as the expected return
        starting from the given inputs (observation and state) and taking the
        given action. It takes observation and state as input and outputs an
        action-value tensor with the shape of [batch_size, num_of_actions].
        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the tensor spec of the action
            input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with `input_tensor_spec` (after reshaping).
                If any element is None, then it will be treated as alf.layers.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layers for encoding the observation.
            lstm_hidden_size (int or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            value_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
        """
        super(QRNNNetwork, self).__init__(
            input_tensor_spec,
            input_preprocessors,
            preprocessing_combiner,
            name=name)

        num_actions = action_spec.maximum - action_spec.minimum + 1
        self._output_spec = TensorSpec((num_actions, ))

        self._encoding_net = LSTMEncodingNetwork(
            input_tensor_spec=self._processed_input_tensor_spec,
            conv_layer_params=conv_layer_params,
            pre_fc_layer_params=fc_layer_params,
            hidden_size=lstm_hidden_size,
            post_fc_layer_params=value_fc_layer_params,
            activation=activation,
            last_layer_size=num_actions,
            last_activation=layers.identity)

    def forward(self, observation, state):
        """Computes action values given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state (nest[tuple]): a nest structure of state tuples (h, c)

        Returns:
            action_value (torch.Tensor): a tensor of the size [batch_size, num_actions]
            new_state (nest[tuple]): the updated states
        """
        observation, state = Network.forward(self, observation, state)
        action_value, state = self._encoding_net(observation, state)
        return action_value, state

    @property
    def state_spec(self):
        return self._encoding_net.state_spec
