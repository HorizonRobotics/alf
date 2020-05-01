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
import functools

import torch
import torch.nn as nn

from .encoding_networks import EncodingNetwork, LSTMEncodingNetwork
from .preprocessors import PreprocessorNetwork
from alf.tensor_specs import TensorSpec
import alf.utils.math_ops as math_ops


@gin.configurable
class ValueNetwork(PreprocessorNetwork):
    """Output temporally uncorrelated values."""

    def __init__(self,
                 input_tensor_spec,
                 input_preprocessor_ctors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 name="ValueNetwork"):
        """Creates a value network that estimates the expected return.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            input_preprocessor_ctors (nested ``InputPreprocessor`` constructors):
                a nest of ``InputPreprocessor`` constructors. They are used to
                create the corresponding ``InputPreprocessor`` instances,  each
                of which will be applied to the corresponding input. If not
                None, then it must have the same structure with
                ``input_tensor_spec`` (after reshaping). If any element is None,
                then ``math_ops.identity`` will be used as its corresponding
                operation applied to the input. This arg is helpful if you want
                to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector.
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
            kernel_initializer (Callable): initializer for all the layers but
            the last layer. If none is provided a default xavier_uniform
            initializer will be used.
            name (str):
        """
        super().__init__(
            input_tensor_spec,
            input_preprocessor_ctors,
            preprocessing_combiner,
            name=name)

        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_

        last_kernel_initializer = functools.partial(torch.nn.init.uniform_, \
                                    a=-0.03, b=0.03)

        self._encoding_net = EncodingNetwork(
            input_tensor_spec=self._processed_input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_size=1,
            last_activation=math_ops.identity,
            last_kernel_initializer=last_kernel_initializer)

        self._output_spec = TensorSpec(())

    def forward(self, observation, state=()):
        """Computes a value given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state: empty for API consistent with ValueRNNNetwork

        Returns:
            value (torch.Tensor): a 1D tensor
            state: empty
        """
        observation, state = super().forward(observation, state)
        value, _ = self._encoding_net(observation)
        return torch.squeeze(value, -1), state


@gin.configurable
class ValueRNNNetwork(PreprocessorNetwork):
    """Outputs temporally correlated values."""

    def __init__(self,
                 input_tensor_spec,
                 input_preprocessor_ctors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 value_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 name="ValueRNNNetwork"):
        """Creates an instance of `ValueRNNNetwork`.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            input_preprocessor_ctors (nested ``InputPreprocessor`` constructors):
                a nest of ``InputPreprocessor`` constructors. They are used to
                create the corresponding ``InputPreprocessor`` instances,  each
                of which will be applied to the corresponding input. If not
                None, then it must have the same structure with
                ``input_tensor_spec`` (after reshaping). If any element is None,
                then ``math_ops.identity`` will be used as its corresponding
                operation applied to the input. This arg is helpful if you want
                to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector.
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
            value_fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a default xavier_uniform
                initializer will be used.
            name (str):
        """
        super().__init__(
            input_tensor_spec,
            input_preprocessor_ctors,
            preprocessing_combiner,
            name=name)

        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_

        last_kernel_initializer = functools.partial(torch.nn.init.uniform_, \
                                    a=-0.03, b=0.03)

        self._encoding_net = LSTMEncodingNetwork(
            input_tensor_spec=self._processed_input_tensor_spec,
            conv_layer_params=conv_layer_params,
            pre_fc_layer_params=fc_layer_params,
            hidden_size=lstm_hidden_size,
            post_fc_layer_params=value_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_size=1,
            last_activation=math_ops.identity,
            last_kernel_initializer=last_kernel_initializer)

        self._output_spec = TensorSpec(())

    def forward(self, observation, state):
        """Computes a value given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state (nest[tuple]): a nest structure of state tuples (h, c)

        Returns:
            value (torch.Tensor): a 1D tensor
            new_state (nest[tuple]): the updated states
        """
        observation, state = super().forward(observation, state)
        value, state = self._encoding_net(observation, state)
        return torch.squeeze(value, -1), state

    @property
    def state_spec(self):
        return self._encoding_net.state_spec
