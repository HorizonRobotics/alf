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

import gin

import torch
import torch.nn as nn

import alf.layers as layers
import alf.nest as nest
from alf.networks import EncodingNetwork, LSTMEncodingNetwork
from alf.networks import Network
from alf.tensor_specs import TensorSpec


@gin.configurable
class CriticNetwork(Network):
    """Create an instance of CriticNetwork."""

    def __init__(self,
                 input_tensor_spec,
                 preprocessing_combiner,
                 input_preprocessors=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu,
                 name="CriticNetwork"):
        """Creates an instance of `CriticNetwork` for estimating action-value of
        continuous actions. The action-value is defined as the expected return
        starting from the given input observation and taking the given action.
        This module takes observation as input and action as input and outputs
        an action-value tensor with the shape of [batch_size].

        Args:
            input_tensor_spec: A tuple of TensorSpecs (observation_spec, action_spec)
                representing the inputs.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code. Required for a critic network
                because the inputs will always be a nest (observations and
                actions).
            input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with `input_tensor_spec` (after reshaping).
                If any element is None, then it will be treated as alf.layers.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            conv_layer_params (list[tuple]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (list[int]): a list of integers representing
                hidden FC layer sizes.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            name (str):
        """
        super(CriticNetwork,
              self).__init__(input_tensor_spec, input_preprocessors,
                             preprocessing_combiner, name)
        self._encoding_net = EncodingNetwork(
            input_tensor_spec=self._input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            last_layer_size=1,
            last_activation=layers.identity)
        self._output_spec = TensorSpec(())

    def forward(self, inputs, state=()):
        """Computes action-value given an observation.

        Args:
            inputs:  A tuple of Tensors consistent with `input_tensor_spec`
            state: empty for API consistent with CriticRNNNetwork

        Returns:
            action_value (torch.Tensor): a tensor of the size [batch_size]
            state: empty
        """
        inputs, state = Network.forward(self, inputs, state)
        # `inputs` should now be a single tensor after this
        action_value, _ = self._encoding_net(inputs)
        return torch.squeeze(action_value, -1), state


@gin.configurable
class CriticRNNNetwork(Network):
    """Creates a critic network with RNN."""

    def __init__(self,
                 input_tensor_spec,
                 preprocessing_combiner,
                 input_preprocessors=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 critic_fc_layer_params=None,
                 activation=torch.relu,
                 name="CriticRNNNetwork"):
        """Creates an instance of `CriticRNNNetwork` for estimating action-value
        of continuous actions. The action-value is defined as the expected return
        starting from the given inputs (observation and state) and taking the
        given action. It takes observation and state as input and outputs an
        action-value tensor with the shape of [batch_size].

        Args:
            input_tensor_spec: A tuple of TensorSpecs (observation_spec, action_spec)
                representing the inputs.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code. Required for a critic network
                because the inputs will always be a nest (observations and
                actions).
            input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with `input_tensor_spec` (after reshaping).
                If any element is None, then it will be treated as alf.layers.identity.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
            fc_layer_params (list[int]): a list of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            lstm_hidden_size (int or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            critic_fc_layer_params (list[int]): a list of integers representing
                hidden FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            name (str):
        """
        super(CriticRNNNetwork,
              self).__init__(input_tensor_spec, input_preprocessors,
                             preprocessing_combiner, name)

        self._encoding_net = LSTMEncodingNetwork(
            input_tensor_spec=self._input_tensor_spec,
            conv_layer_params=conv_layer_params,
            pre_fc_layer_params=fc_layer_params,
            hidden_size=lstm_hidden_size,
            post_fc_layer_params=critic_fc_layer_params,
            activation=activation,
            last_layer_size=1,
            last_activation=layers.identity)
        self._output_spec = TensorSpec(())

    def forward(self, inputs, state):
        """Computes action-value given an observation.

        Args:
            inputs:  A tuple of Tensors consistent with `input_tensor_spec`
            state (nest[tuple]): a nest structure of state tuples (h, c)

        Returns:
            action_value (torch.Tensor): a tensor of the size [batch_size]
            new_state (nest[tuple]): the updated states
        """
        inputs, state = Network.forward(self, inputs, state)
        # `inputs` should now be a single tensor after this
        action_value, state = self._encoding_net(inputs, state)
        return torch.squeeze(action_value, -1), state

    @property
    def state_spec(self):
        return self._encoding_net.state_spec
