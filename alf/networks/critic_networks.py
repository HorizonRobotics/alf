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
"""CriticNetworks"""

import gin
import functools
import math

import torch
import torch.nn as nn

import alf.utils.math_ops as math_ops
import alf.nest as nest
from alf.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec

from .network import Network
from .encoding_networks import EncodingNetwork, LSTMEncodingNetwork, ParallelEncodingNetwork


def _check_action_specs_for_critic_networks(action_spec,
                                            action_input_preprocessor_ctors):
    if len(nest.flatten(action_spec)) > 1:
        raise ValueError('Only a single action is supported by this network')

    if action_spec.is_discrete:
        assert action_input_preprocessor_ctors is not None, (
            'CriticNetwork only supports continuous actions. The given ' +
            'action spec {} is discrete. Use QNetwork instead. '.format(
                action_spec) +
            'Alternatively, specify `action_input_preprocessor_ctors` to transform '
            + 'discrete actions to continuous action embeddings first.')


@gin.configurable
class CriticNetwork(Network):
    """Creates an instance of ``CriticNetwork`` for estimating action-value of
    continuous or discrete actions. The action-value is defined as the expected
    return starting from the given input observation and taking the given action.
    This module takes observation as input and action as input and outputs an
    action-value tensor with the shape of ``[batch_size]``.
    """

    def __init__(self,
                 input_tensor_spec,
                 observation_input_preprocessor_ctors=None,
                 observation_preprocessing_combiner=None,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 action_input_preprocessor_ctors=None,
                 action_fc_layer_params=None,
                 joint_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 name="CriticNetwork"):
        """

        Args:
            input_tensor_spec: A tuple of ``TensorSpec``s ``(observation_spec, action_spec)``
                representing the inputs.
            observation_input_preprocessor_ctors (nested ``InputPreprocessor``
                constructors): a nest of ``InputPreprocessor`` constructors.
                They are used to create the corresponding ``InputPreprocessor``
                instances,  each of which will be applied to the corresponding
                observation input.
            observation_preprocessing_combiner (NestCombiner): preprocessing called
                on complex observation inputs.
            observation_conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            observation_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for observations.
            action_input_preprocessor_ctors  (nested ``InputPreprocessor``
                constructors): a nest of ``InputPreprocessor`` constructors.
                They are used to create the corresponding ``InputPreprocessor``
                instances,  each of which will be applied to the corresponding
                action input.
            action_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for actions.
            joint_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a variance_scaling_initializer
                with uniform distribution will be used.
            name (str):
        """
        super().__init__(input_tensor_spec, name=name)

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                gain=math.sqrt(1.0 / 3),
                mode='fan_in',
                distribution='uniform')

        observation_spec, action_spec = input_tensor_spec

        _check_action_specs_for_critic_networks(
            action_spec, action_input_preprocessor_ctors)

        self._single_action_spec = action_spec
        self._obs_encoder = EncodingNetwork(
            observation_spec,
            input_preprocessor_ctors=observation_input_preprocessor_ctors,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        self._action_encoder = EncodingNetwork(
            action_spec,
            input_preprocessor_ctors=action_input_preprocessor_ctors,
            fc_layer_params=action_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        last_kernel_initializer = functools.partial(
            torch.nn.init.uniform_, a=-0.003, b=0.003)

        self._joint_encoder = EncodingNetwork(
            TensorSpec((self._obs_encoder.output_spec.shape[0] +
                        self._action_encoder.output_spec.shape[0], )),
            fc_layer_params=joint_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_size=1,
            last_activation=math_ops.identity,
            last_kernel_initializer=last_kernel_initializer)

        self._output_spec = TensorSpec(())

    def forward(self, inputs, state=()):
        """Computes action-value given an observation.

        Args:
            inputs:  A tuple of Tensors consistent with ``input_tensor_spec``
            state: empty for API consistent with ``CriticRNNNetwork``

        Returns:
            tuple:
            - action_value (torch.Tensor): a tensor of the size ``[batch_size]``
            - state: empty
        """
        observations, actions = inputs
        actions = actions.to(torch.float32)

        encoded_obs, _ = self._obs_encoder(observations)
        encoded_action, _ = self._action_encoder(actions)
        joint = torch.cat([encoded_obs, encoded_action], -1)
        action_value, _ = self._joint_encoder(joint)
        return torch.squeeze(action_value, -1), state

    def make_parallel(self, n):
        """Create a ``ParallelCriticNetwork`` using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        """
        return ParallelCriticNetwork(self, n, "parallel_" + self._name)


class ParallelCriticNetwork(Network):
    """Perform ``n`` critic computations in parallel."""

    def __init__(self,
                 critic_network: CriticNetwork,
                 n: int,
                 name="ParallelCriticNetwork"):
        """
        It create a parallelized version of ``critic_network``.

        Args:
            critic_network (CriticNetwork): non-parallelized critic network
            n (int): make ``n`` replicas from ``critic_network`` with different
                initialization.
            name (str):
        """
        super().__init__(
            input_tensor_spec=critic_network.input_tensor_spec, name=name)
        self._obs_encoder = critic_network._obs_encoder.make_parallel(n)
        self._action_encoder = critic_network._action_encoder.make_parallel(n)
        self._joint_encoder = critic_network._joint_encoder.make_parallel(n)
        self._output_spec = TensorSpec((n, ))

    def forward(self, inputs, state=()):
        """Computes action-value given an observation.

        Args:
            inputs (tuple):  A tuple of Tensors consistent with `input_tensor_spec``.
            state (tuple): Empty for API consistent with ``CriticRNNNetwork``.

        Returns:
            tuple:
            - action_value (torch.Tensor): a tensor of shape :math:`[B,n]`, where
                :math:`B` is the batch size.
            - state: empty
        """
        observations, actions = inputs
        actions = actions.to(torch.float32)

        encoded_obs, _ = self._obs_encoder(observations)
        encoded_action, _ = self._action_encoder(actions)
        joint = torch.cat([encoded_obs, encoded_action], -1)
        action_value, _ = self._joint_encoder(joint)
        return torch.squeeze(action_value, -1), state


@gin.configurable
class CriticRNNNetwork(Network):
    """Creates an instance of ``CriticRNNNetwork`` for estimating action-value
    of continuous or discrete actions. The action-value is defined as the
    expected return starting from the given inputs (observation and state) and
    taking the given action. It takes observation and state as input and outputs
    an action-value tensor with the shape of [batch_size].
    """

    def __init__(self,
                 input_tensor_spec,
                 observation_input_preprocessor_ctors=None,
                 observation_preprocessing_combiner=None,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 action_input_preprocessor_ctors=None,
                 action_fc_layer_params=None,
                 joint_fc_layer_params=None,
                 lstm_hidden_size=100,
                 critic_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 name="CriticRNNNetwork"):
        """

        Args:
            input_tensor_spec: A tuple of ``TensorSpec``s ``(observation_spec, action_spec)``
                representing the inputs.
            observation_input_preprocessor_ctors (nested ``InputPreprocessor``
                constructors): a nest of ``InputPreprocessor`` constructors.
                They are used to create the corresponding ``InputPreprocessor``
                instances,  each of which will be applied to the corresponding
                observation input.
            observation_preprocessing_combiner (NestCombiner): preprocessing called
                on complex observation inputs.
            observation_conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            observation_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for observations.
            action_input_preprocessor_ctors  (nested ``InputPreprocessor``
                constructors): a nest of ``InputPreprocessor`` constructors.
                They are used to create the corresponding ``InputPreprocessor``
                instances,  each of which will be applied to the corresponding
                action input.
            action_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for actions.
            joint_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            lstm_hidden_size (int or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            critic_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a ``variance_scaling_initializer``
                with uniform distribution will be used.
            name (str):
        """
        super().__init__(input_tensor_spec, name=name)

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                gain=math.sqrt(1.0 / 3),
                mode='fan_in',
                distribution='uniform')

        observation_spec, action_spec = input_tensor_spec

        _check_action_specs_for_critic_networks(
            action_spec, action_input_preprocessor_ctors)

        self._single_action_spec = action_spec
        self._obs_encoder = EncodingNetwork(
            observation_spec,
            input_preprocessor_ctors=observation_input_preprocessor_ctors,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        self._action_encoder = EncodingNetwork(
            action_spec,
            input_preprocessor_ctors=action_input_preprocessor_ctors,
            fc_layer_params=action_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        self._joint_encoder = EncodingNetwork(
            TensorSpec((self._obs_encoder.output_spec.shape[0] +
                        self._action_encoder.output_spec.shape[0], )),
            fc_layer_params=joint_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        last_kernel_initializer = functools.partial(
            torch.nn.init.uniform_, a=-0.003, b=0.003)

        self._lstm_encoding_net = LSTMEncodingNetwork(
            input_tensor_spec=self._joint_encoder.output_spec,
            hidden_size=lstm_hidden_size,
            post_fc_layer_params=critic_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_size=1,
            last_activation=math_ops.identity,
            last_kernel_initializer=last_kernel_initializer)

        self._output_spec = TensorSpec(())

    def forward(self, inputs, state):
        """Computes action-value given an observation.

        Args:
            inputs:  A tuple of Tensors consistent with ``input_tensor_spec``
            state (nest[tuple]): a nest structure of state tuples ``(h, c)``

        Returns:
            tuple:
            - action_value (torch.Tensor): a tensor of the size ``[batch_size]``
            - new_state (nest[tuple]): the updated states
        """
        observations, actions = inputs
        actions = actions.to(torch.float32)

        encoded_obs, _ = self._obs_encoder(observations)
        encoded_action, _ = self._action_encoder(actions)
        joint = torch.cat([encoded_obs, encoded_action], -1)
        encoded_joint, _ = self._joint_encoder(joint)
        action_value, state = self._lstm_encoding_net(encoded_joint, state)
        return torch.squeeze(action_value, -1), state

    @property
    def state_spec(self):
        return self._lstm_encoding_net.state_spec
