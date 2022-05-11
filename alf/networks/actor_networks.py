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
"""ActorNetworks"""

import functools
import math
from typing import Callable

import torch
import torch.nn as nn

import alf
from .encoding_networks import EncodingNetwork, LSTMEncodingNetwork
from .preprocessor_networks import PreprocessorNetwork
import alf.layers as layers
import alf.nest as nest
from alf.initializers import variance_scaling_init
from alf.networks import Network
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, math_ops, spec_utils


@alf.configurable
class ActorNetworkBase(Network):
    """A base class for ``ActorNetwork`` and ``ActorRNNNetwork``.

    Can also be used to create customized actor networks by providing
    different encoding network creators.
    """

    def __init__(self,
                 input_tensor_spec: alf.NestedTensorSpec,
                 action_spec: alf.NestedTensorSpec,
                 encoding_network_ctor: Callable = EncodingNetwork,
                 squashing_func=torch.tanh,
                 name="ActorNetworkBase",
                 **encoder_kwargs):
        """
        Args:
            input_tensor_spec: the tensor spec of the input.
            action_spec: the tensor spec of the action.
            encoding_network_ctor: the creator of the encoding network that does
                the heavy lifting of the actor.
            squashing_func: the activation function used to squashing
                the output to the range :math:`(-1, 1)`. Default to ``tanh``.
            name: name of the network
            encoder_kwargs: the extra keyword arguments to the encoding network
        """
        super().__init__(input_tensor_spec, name=name)

        if encoder_kwargs.get('kernel_initializer', None) is None:
            encoder_kwargs['kernel_initializer'] = functools.partial(
                variance_scaling_init,
                gain=math.sqrt(1.0 / 3),
                mode='fan_in',
                distribution='uniform')

        self._action_spec = action_spec
        flat_action_spec = nest.flatten(action_spec)
        self._flat_action_spec = flat_action_spec

        is_continuous = [
            single_action_spec.is_continuous
            for single_action_spec in flat_action_spec
        ]
        assert all(is_continuous), "only continuous action is supported"

        self._encoding_net = encoding_network_ctor(
            input_tensor_spec,
            name=self.name + '.encoding_net',
            **encoder_kwargs)

        last_kernel_initializer = functools.partial(torch.nn.init.uniform_, \
                                    a=-0.003, b=0.003)
        self._action_layers = nn.ModuleList()
        self._squashing_func = squashing_func
        for single_action_spec in flat_action_spec:
            self._action_layers.append(
                layers.FC(
                    self._encoding_net.output_spec.shape[0],
                    single_action_spec.shape[0],
                    kernel_initializer=last_kernel_initializer))

    def forward(self, observation, state=()):
        """Computes action given an observation.

        Args:
            inputs:  A tensor consistent with ``input_tensor_spec``
            state: empty for API consistent with ``ActorRNNNetwork``

        Returns:
            tuple:
            - action (torch.Tensor): a tensor consistent with ``action_spec``
            - state: empty
        """

        encoded_obs, state = self._encoding_net(observation, state)

        actions = []
        i = 0
        for layer, spec in zip(self._action_layers, self._flat_action_spec):
            pre_activation = layer(encoded_obs)
            action = self._squashing_func(pre_activation)
            action = spec_utils.scale_to_spec(action, spec)

            if alf.summary.should_summarize_output():
                alf.summary.scalar(
                    name='summarize_output/' + self.name + '.action_layer.' +
                    str(i) + '.pre_activation.output_norm.' +
                    common.exe_mode_name(),
                    data=torch.mean(
                        pre_activation.norm(
                            dim=list(range(1, pre_activation.ndim)))))
                a_name = (
                    'summarize_output/' + self.name + '.action_layer.' + str(i)
                    + '.action.output_norm.' + common.exe_mode_name())
                alf.summary.scalar(
                    name=a_name,
                    data=torch.mean(
                        action.norm(dim=list(range(1, action.ndim)))))

            actions.append(action)
            i += 1

        output_actions = nest.pack_sequence_as(self._action_spec, actions)
        return output_actions, state

    @property
    def state_spec(self):
        """Return the state spec of the actor network. It is simply the state spec
        of the encoding network."""
        return self._encoding_net.state_spec


@alf.configurable
class ActorNetwork(ActorNetworkBase):
    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 squashing_func=torch.tanh,
                 kernel_initializer=None,
                 name="ActorNetwork"):
        """Creates an instance of ``ActorNetwork``, which maps the inputs to
        actions (single or nested) through a sequence of deterministic layers.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input.
            action_spec (BoundedTensorSpec): the tensor spec of the action.
            input_preprocessors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with ``input_tensor_spec`` (after reshaping).
                If any element is None, then it will be treated as ``math_ops.identity``.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
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
            squashing_func (Callable): the activation function used to squashing
                the output to the range :math:`(-1, 1)`. Default to ``tanh``.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a ``variance_scaling_initializer``
                with uniform distribution will be used.
            name (str): name of the network
        """
        super(ActorNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            encoding_network_ctor=EncodingNetwork,
            squashing_func=squashing_func,
            name=name,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)


@alf.configurable
class ActorRNNNetwork(ActorNetworkBase):
    def __init__(self,
                 input_tensor_spec: TensorSpec,
                 action_spec: BoundedTensorSpec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 actor_fc_layer_params=None,
                 activation=torch.relu_,
                 squashing_func=torch.tanh,
                 kernel_initializer=None,
                 name="ActorRNNNetwork"):
        """Creates an instance of `ActorRNNNetwork`, which maps the inputs
        (observation and states) to actions (single or nested) through a
        sequence of deterministic layers.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input.
            action_spec (BoundedTensorSpec): the tensor spec of the action.
            input_preprocessors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
                corresponding input. If not None, then it must
                have the same structure with ``input_tensor_spec`` (after reshaping).
                If any element is None, then it will be treated as ``math_ops.identity``.
                This arg is helpful if you want to have separate preprocessings
                for different inputs by configuring a gin file without changing
                the code. For example, embedding a discrete input before concatenating
                it to another continuous vector.
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
            lstm_hidden_size (int or tuple[int]): the hidden size(s)
                of the LSTM cell(s). Each size corresponds to a cell. If there
                are multiple sizes, then lstm cells are stacked.
            actor_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            squashing_func (Callable): the activation function used to squashing
                the output to the range :math:`(-1, 1)`. Default to ``tanh``.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a variance_scaling_initializer
                with uniform distribution will be used.
            name (str): name of the network
        """
        super(ActorRNNNetwork, self).__init__(
            input_tensor_spec,
            action_spec,
            encoding_network_ctor=LSTMEncodingNetwork,
            squashing_func=squashing_func,
            name=name,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            pre_fc_layer_params=fc_layer_params,
            hidden_size=lstm_hidden_size,
            post_fc_layer_params=actor_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)
