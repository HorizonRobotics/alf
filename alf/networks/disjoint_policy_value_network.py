# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from typing import Callable

import torch
import alf
from alf.tensor_specs import TensorSpec
from alf.data_structures import namedtuple
from .network import Network
from .projection_networks import NormalProjectionNetwork, CategoricalProjectionNetwork
from .encoding_networks import EncodingNetwork

DisjointPolicyValueNetworkState = namedtuple(
    'DisjointPolicyValueNetworkState', ['actor', 'value', 'aux'],
    default_value=())


def _create_projection_net_based_on_actor_spec(
        discrete_projection_net_ctor: Callable[..., Network],
        continuous_projection_net_ctor: Callable[..., Network], input_size,
        action_spec):
    def _create_individually(spec):
        constructor = (discrete_projection_net_ctor
                       if spec.is_discrete else continuous_projection_net_ctor)
        return constructor(input_size=input_size, action_spec=spec)

    return alf.nest.map_structure(_create_individually, action_spec)


# TODO(breakds): Make this more flexible to allow recurrent networks
@alf.configurable
class DisjointPolicyValueNetwork(Network):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 encoding_network_ctor=EncodingNetwork,
                 is_sharing_encoder: bool = False,
                 kernel_initializer=None,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork,
                 name='DisjointPolicyValueNetwork'):
        super().__init__(input_tensor_spec=observation_spec, name=name)

        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_

        self._actor_encoder = encoding_network_ctor(
            input_tensor_spec=observation_spec,
            kernel_initializer=kernel_initializer)

        projection_net = _create_projection_net_based_on_actor_spec(
            discrete_projection_net_ctor=discrete_projection_net_ctor,
            continuous_projection_net_ctor=continuous_projection_net_ctor,
            input_size=self._actor_encoder.output_spec.shape[0],
            action_spec=action_spec)

        if alf.nest.is_nested(projection_net):
            # Force picking up the parameters inside those project networks into
            # this DisjointPolicyValueNetwork instance.
            self._projection_net_module_list = torch.nn.ModuleList(
                alf.nest.flatten(projection_net))
            self._actor_branch = alf.nn.Sequential(
                self._actor_encoder,
                alf.nn.Branch(projection_net, name='NestedProjection'))
        else:
            self._actor_branch = alf.nn.Sequential(self._actor_encoder,
                                                   projection_net)

        # Like the value head Aux head is outputing value estimation
        self._aux_branch = alf.nn.Sequential(
            self._actor_encoder,
            alf.layers.FC(
                input_size=self._actor_encoder.output_spec.shape[0],
                output_size=1), alf.layers.Reshape(shape=()))

        if is_sharing_encoder:
            self._value_branch = alf.nn.Sequential(
                self._actor_encoder, alf.layers.Detach(),
                alf.layers.FC(
                    input_size=self._actor_encoder.output_spec.shape[0],
                    output_size=1), alf.layers.Reshape(shape=()))
        else:
            value_encoder = encoding_network_ctor(
                input_tensor_spec=observation_spec,
                kernel_initializer=kernel_initializer)

            self._value_branch = alf.nn.Sequential(
                value_encoder,
                alf.layers.FC(
                    input_size=self._actor_encoder.output_spec.shape[0],
                    output_size=1), alf.layers.Reshape(shape=()))

    def forward(self, observation, state: DisjointPolicyValueNetworkState):
        action_distribution, actor_state = self._actor_branch(
            observation, state=state.actor)

        value, value_state = self._value_branch(observation, state=state.value)

        aux, aux_state = self._value_branch(observation, state=state.aux)

        return action_distribution, value, aux, DisjointPolicyValueNetworkState(
            actor=actor_state, value=value_state, aux=aux_state)

    @property
    def state_spec(self):
        return DisjointPolicyValueNetworkState(
            actor=self._actor_branch.state_spec,
            value=self._value_branch.state_spec,
            aux=self._aux_branch.state_spec)
