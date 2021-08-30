# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import functools
import math

import torch

import alf
import alf.utils.math_ops as math_ops
import alf.nest as nest
from alf.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec

from .encoding_networks import EncodingNetwork, LSTMEncodingNetwork, ParallelEncodingNetwork


def _check_action_specs_for_critic_networks(
        action_spec, action_input_processors, action_preprocessing_combiner):

    if len(nest.flatten(action_spec)) > 1:
        assert action_preprocessing_combiner is not None, (
            "An action combiner is needed when there are multiple action specs:"
            " {}".format(action_spec))

    def _check_individual(spec, proc):
        if spec.is_discrete:
            assert proc is not None, (
                'CriticNetwork only supports continuous actions. One of given '
                + 'action specs {} is discrete. Use QNetwork instead. '.format(
                    spec) +
                'Alternatively, specify `action_input_processors` to transform '
                + 'discrete actions to continuous action embeddings first.')

    if action_input_processors is None:
        action_input_processors = nest.map_structure(lambda _: None,
                                                     action_spec)

    nest.map_structure(_check_individual, action_spec, action_input_processors)


@alf.configurable
class CriticNetwork(EncodingNetwork):
    """Creates an instance of ``CriticNetwork`` for estimating action-value of
    continuous or discrete actions. The action-value is defined as the expected
    return starting from the given input observation and taking the given action.
    This module takes observation as input and action as input and outputs an
    action-value tensor with the shape of ``[batch_size]``.

    The network take a tuple of (observation, action) as input to computes the
    action-value given an observation.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=TensorSpec(()),
                 observation_input_processors=None,
                 observation_preprocessing_combiner=None,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 action_input_processors=None,
                 action_preprocessing_combiner=None,
                 action_fc_layer_params=None,
                 joint_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_fc_bn=False,
                 use_naive_parallel_network=False,
                 name="CriticNetwork"):
        """

        Args:
            input_tensor_spec: A tuple of ``TensorSpec``s ``(observation_spec, action_spec)``
                representing the inputs.
            output_tensor_spec (TensorSpec): spec for the output
            observation_input_preprocessors (nested InputPreprocessor): a nest of
                ``InputPreprocessor``, each of which will be applied to the
                corresponding observation input.
            observation_preprocessing_combiner (NestCombiner): preprocessing called
                on complex observation inputs.
            observation_conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            observation_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for observations.
            action_input_processors (nested InputPreprocessor): a nest of
                ``InputPreprocessor``, each of which will be applied to the
                corresponding action input.
            action_preprocessing_combiner (NestCombiner): preprocessing called
                to combine complex action inputs.
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
            use_fc_bn (bool): whether use Batch Normalization for the internal
                FC layers (i.e. FC layers beside the last one).
            use_naive_parallel_network (bool): if True, will use
                ``NaiveParallelNetwork`` when ``make_parallel`` is called. This
                might be useful in cases when the ``NaiveParallelNetwork``
                has an advantange in terms of speed over ``ParallelNetwork``.
                You have to test to see which way is faster for your particular
                situation.
            name (str):
        """
        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                gain=math.sqrt(1.0 / 3),
                mode='fan_in',
                distribution='uniform')

        observation_spec, action_spec = input_tensor_spec

        obs_encoder = EncodingNetwork(
            observation_spec,
            input_preprocessors=observation_input_processors,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn,
            name=name + ".obs_encoder")

        _check_action_specs_for_critic_networks(action_spec,
                                                action_input_processors,
                                                action_preprocessing_combiner)
        action_encoder = EncodingNetwork(
            action_spec,
            input_preprocessors=action_input_processors,
            preprocessing_combiner=action_preprocessing_combiner,
            fc_layer_params=action_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            use_fc_bn=use_fc_bn,
            name=name + ".action_encoder")

        last_kernel_initializer = functools.partial(
            torch.nn.init.uniform_, a=-0.003, b=0.003)

        super().__init__(
            input_tensor_spec=input_tensor_spec,
            output_tensor_spec=output_tensor_spec,
            input_preprocessors=(obs_encoder, action_encoder),
            preprocessing_combiner=alf.layers.NestConcat(dim=-1),
            fc_layer_params=joint_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_size=output_tensor_spec.numel,
            last_activation=math_ops.identity,
            use_fc_bn=use_fc_bn,
            last_kernel_initializer=last_kernel_initializer,
            name=name)
        self._use_naive_parallel_network = use_naive_parallel_network

    def make_parallel(self, n):
        """Create a parallel critic network using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        If ``use_naive_parallel_network`` is True, use ``NaiveParallelNetwork``
        to create the parallel network.
        """
        if self._use_naive_parallel_network:
            return alf.networks.NaiveParallelNetwork(self, n)
        else:
            return super().make_parallel(n, True)


@alf.configurable
class CriticRNNNetwork(LSTMEncodingNetwork):
    """Creates an instance of ``CriticRNNNetwork`` for estimating action-value
    of continuous or discrete actions. The action-value is defined as the
    expected return starting from the given inputs (observation and state) and
    taking the given action. It takes observation and state as input and outputs
    an action-value tensor with the shape of [batch_size].
    """

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=TensorSpec(()),
                 observation_input_processors=None,
                 observation_preprocessing_combiner=None,
                 observation_conv_layer_params=None,
                 observation_fc_layer_params=None,
                 action_input_processors=None,
                 action_preprocessing_combiner=None,
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
            ourput_tensor_spec (TensorSpec): spec for the output
            observation_input_preprocessors (nested InputPreprocessor): a nest of
                ``InputPreprocessor``, each of which will be applied to the
                corresponding observation input.
            observation_preprocessing_combiner (NestCombiner): preprocessing called
                on complex observation inputs.
            observation_conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            observation_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes for observations.
            action_input_processors (nested InputPreprocessor): a nest of
                ``InputPreprocessor``, each of which will be applied to the
                corresponding action input.
            action_preprocessing_combiner (NestCombiner): preprocessing called
                to combine complex action inputs.
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
        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                gain=math.sqrt(1.0 / 3),
                mode='fan_in',
                distribution='uniform')

        observation_spec, action_spec = input_tensor_spec

        obs_encoder = EncodingNetwork(
            observation_spec,
            input_preprocessors=observation_input_processors,
            preprocessing_combiner=observation_preprocessing_combiner,
            conv_layer_params=observation_conv_layer_params,
            fc_layer_params=observation_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        _check_action_specs_for_critic_networks(action_spec,
                                                action_input_processors,
                                                action_preprocessing_combiner)
        action_encoder = EncodingNetwork(
            action_spec,
            input_preprocessors=action_input_processors,
            preprocessing_combiner=action_preprocessing_combiner,
            fc_layer_params=action_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)

        last_kernel_initializer = functools.partial(
            torch.nn.init.uniform_, a=-0.003, b=0.003)

        super().__init__(
            input_tensor_spec=input_tensor_spec,
            output_tensor_spec=output_tensor_spec,
            input_preprocessors=(obs_encoder, action_encoder),
            preprocessing_combiner=alf.layers.NestConcat(dim=-1),
            pre_fc_layer_params=joint_fc_layer_params,
            hidden_size=lstm_hidden_size,
            post_fc_layer_params=critic_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_size=output_tensor_spec.numel,
            last_activation=math_ops.identity,
            last_kernel_initializer=last_kernel_initializer)

    def make_parallel(self, n):
        """Create a parallel critic RNN network using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        If ``use_naive_parallel_network`` is True, use ``NaiveParallelNetwork``
        to create the parallel network.
        """
        return super().make_parallel(n, True)
