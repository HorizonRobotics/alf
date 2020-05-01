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
import functools

import torch
import torch.nn as nn

import alf.layers as layers
from alf.networks import EncodingNetwork, LSTMEncodingNetwork, ParallelEncodingNetwork
from alf.networks import Network
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
import alf.utils.math_ops as math_ops


@gin.configurable
class QNetwork(Network):
    """Create an instance of QNetwork."""

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 input_preprocessor_ctors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 name="QNetwork"):
        """Creates an instance of ``QNetwork`` for estimating action-value of
        discrete actions. The action-value is defined as the expected return
        starting from the given input observation and taking the given action.
        It takes observation as input and outputs an action-value tensor with
        the shape of ``[batch_size, num_of_actions]``.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the tensor spec of the action
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
                ``input_tensor_spec`` as the input to compute the processed
                tensor spec. For example, see ``alf.nest.utils.NestConcat``. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layer sizes.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a default ``variance_scaling_initializer``
                will be used.
        """
        super(QNetwork, self).__init__(input_tensor_spec, name=name)

        num_actions = action_spec.maximum - action_spec.minimum + 1
        self._output_spec = TensorSpec((num_actions, ))

        self._encoding_net = EncodingNetwork(
            input_tensor_spec=input_tensor_spec,
            input_preprocessor_ctors=input_preprocessor_ctors,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        last_kernel_initializer = functools.partial(torch.nn.init.uniform_, \
                                    a=-0.003, b=0.003)

        self._final_layer = layers.FC(
            self._encoding_net.output_spec.shape[0],
            num_actions,
            activation=math_ops.identity,
            kernel_initializer=last_kernel_initializer,
            bias_init_value=-0.2)

    def forward(self, observation, state=()):
        """Computes action values given an observation.

        Args:
            observation (nest): consistent with ``input_tensor_spec``
            state: empty for API consistent with ``QRNNNetwork``

        Returns:
            tuple:
            - action_value (torch.Tensor): a tensor of the size
              ``[batch_size, num_actions]``
            - state: empty
        """
        encoded_obs, _ = self._encoding_net(observation)
        action_value = self._final_layer(encoded_obs)
        return action_value, state

    def make_parallel(self, n):
        """Create a ``ParallelQNetwork`` using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        """
        return ParallelQNetwork(self, n, "parallel_" + self._name)


class ParallelQNetwork(Network):
    """Perform ``n`` Q-value computations in parallel."""

    def __init__(self, q_network: QNetwork, n: int, name="ParallelQNetwork"):
        """
        Args:
            q_network (QNetwork): non-parallelized q network
            n (int): make ``n`` replicas from ``q_network`` with different
                parameter initializations.
            name (str):
        """
        super().__init__(
            input_tensor_spec=q_network.input_tensor_spec, name=name)
        self._encoding_net = q_network._encoding_net.make_parallel(n)
        self._final_layer = q_network._final_layer.make_parallel(n)
        self._output_spec = TensorSpec((n, ) +
                                       tuple(q_network.output_spec.shape))

    def forward(self, inputs, state=()):
        """Compute action values given an observation.

        Args:
            inputs (nest): consistent with ``input_tensor_spec``.
            state: empty for API consistent with ``QRNNNetwork``.

        Returns:
            tuple:
            - action_value (Tensor): a tensor of shape :math:`[B,n,k]`, where
              :math:`B` is the batch size, :math:`n` is the num of replicas, and
              :math:`k` is the number of actions.
            - state: empty
        """
        encoded_obs, _ = self._encoding_net(inputs)
        action_value = self._final_layer(encoded_obs)
        return action_value, state


@gin.configurable
class QRNNNetwork(Network):
    """Create a RNN-based that outputs temporally correlated q-values."""

    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 input_preprocessor_ctors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 value_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 name="QRNNNetwork"):
        """Creates an instance of `QRNNNetwork` for estimating action-value of
        discrete actions. The action-value is defined as the expected return
        starting from the given inputs (observation and state) and taking the
        given action. It takes observation and state as input and outputs an
        action-value tensor with the shape of [batch_size, num_of_actions].
        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the tensor spec of the action
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
            value_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a default
                variance_scaling_initializer will be used.
        """
        super().__init__(input_tensor_spec, name=name)

        num_actions = action_spec.maximum - action_spec.minimum + 1
        self._output_spec = TensorSpec((num_actions, ))

        self._encoding_net = LSTMEncodingNetwork(
            input_tensor_spec=input_tensor_spec,
            input_preprocessor_ctors=input_preprocessor_ctors,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            pre_fc_layer_params=fc_layer_params,
            hidden_size=lstm_hidden_size,
            post_fc_layer_params=value_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        last_kernel_initializer = functools.partial(torch.nn.init.uniform_, \
                                    a=-0.003, b=0.003)

        self._final_layer = layers.FC(
            self._encoding_net.output_spec.shape[0],
            num_actions,
            activation=math_ops.identity,
            kernel_initializer=last_kernel_initializer,
            bias_init_value=-0.2)

    def forward(self, observation, state):
        """Computes action values given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state (nest[tuple]): a nest structure of state tuples (h, c)

        Returns:
            action_value (torch.Tensor): a tensor of the size [batch_size, num_actions]
            new_state (nest[tuple]): the updated states
        """
        encoded_obs, state = self._encoding_net(observation, state)
        action_value = self._final_layer(encoded_obs)
        return action_value, state

    @property
    def state_spec(self):
        return self._encoding_net.state_spec
