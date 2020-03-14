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
"""ActorDistributionNetwork and ActorRNNDistributionNetwork."""

import gin

import torch
import torch.nn as nn

from .encoding_networks import EncodingNetwork, LSTMEncodingNetwork
from .projection_networks import NormalProjectionNetwork, CategoricalProjectionNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.networks.network import DistributionNetwork, Network


@gin.configurable
class ActorDistributionNetwork(DistributionNetwork):
    """Outputs temporally correlated actions."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu,
                 kernel_initializer=None,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork,
                 name="ActorDistributionNetwork"):
        """Creates an instance of `ActorDistributionNetwork`.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the action spec
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
            activation (nn.functional): activation used for hidden layers.
            kernel_initializer (Callable): initializer for all the layers
                excluding the projection net. If none is provided a default
                xavier_uniform will be used.
            discrete_projection_net_ctor (ProjectionNetwork): constructor that
                generates a discrete projection network that outputs discrete
                actions.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                continuous actions.
            name (str):
        """
        super(ActorDistributionNetwork, self).__init__(
            input_tensor_spec,
            input_preprocessors,
            preprocessing_combiner,
            name=name)

        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_

        self._action_spec = action_spec
        self._encoding_net = EncodingNetwork(
            input_tensor_spec=self._processed_input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)
        self._create_projection_net(discrete_projection_net_ctor,
                                    continuous_projection_net_ctor)

    def _create_projection_net(self, discrete_projection_net_ctor,
                               continuous_projection_net_ctor):
        if self._action_spec.is_discrete:
            self._projection_net = discrete_projection_net_ctor(
                input_size=self._encoding_net.output_spec.shape[0],
                action_spec=self._action_spec)
        else:
            self._projection_net = continuous_projection_net_ctor(
                input_size=self._encoding_net.output_spec.shape[0],
                action_spec=self._action_spec)

    def forward(self, observation, state=()):
        """Computes an action distribution given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state: empty for API consistent with ActorRNNDistributionNetwork

        Returns:
            act_dist (torch.distributions): action distribution
            state: empty
        """
        observation, state = Network.forward(self, observation, state)
        act_dist, _ = self._projection_net(self._encoding_net(observation)[0])
        return act_dist, state


@gin.configurable
class ActorDistributionRNNNetwork(ActorDistributionNetwork):
    """Outputs temporally uncorrelated actions."""

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 lstm_hidden_size=100,
                 actor_fc_layer_params=None,
                 activation=torch.relu,
                 kernel_initializer=None,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork,
                 name="ActorRNNDistributionNetwork"):
        """Creates an instance of `ActorRNNDistributionNetwork`.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            action_spec (TensorSpec): the action spec
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
            actor_fc_layer_params (tuple[int]): a tuple of integers representing hidden
                FC layers that are applied after the lstm cell's output.
            activation (nn.functional): activation used for hidden layers.
            kernel_initializer (Callable): initializer for all the layers
                excluding the projection net. If none is provided a default
                xavier_uniform will be used.
            discrete_projection_net_ctor (ProjectionNetwork): constructor that
                generates a discrete projection network that outputs discrete
                actions.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                continuous actions.
            name (str):
        """
        super(ActorDistributionRNNNetwork, self).__init__(
            input_tensor_spec,
            action_spec,
            input_preprocessors,
            preprocessing_combiner,
            conv_layer_params,
            fc_layer_params,
            name=name)

        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_

        self._encoding_net = LSTMEncodingNetwork(
            input_tensor_spec=self._processed_input_tensor_spec,
            conv_layer_params=conv_layer_params,
            pre_fc_layer_params=fc_layer_params,
            hidden_size=lstm_hidden_size,
            post_fc_layer_params=actor_fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer)
        self._create_projection_net(discrete_projection_net_ctor,
                                    continuous_projection_net_ctor)

    def forward(self, observation, state):
        """Computes an action distribution given an observation.

        Args:
            observation (torch.Tensor): consistent with `input_tensor_spec`
            state (nest[tuple]): a nest structure of state tuples (h, c)

        Returns:
            act_dist (torch.distributions): action distribution
            new_state (nest[tuple]): the updated states
        """
        observation, state = Network.forward(self, observation, state)
        encoding, state = self._encoding_net(observation, state)
        act_dist, _ = self._projection_net(encoding)
        return act_dist, state

    @property
    def state_spec(self):
        return self._encoding_net.state_spec
