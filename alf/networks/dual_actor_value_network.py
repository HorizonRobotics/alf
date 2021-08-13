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

import torch

import alf
from alf.tensor_specs import TensorSpec
from alf.networks.network import Network
from alf.networks.projection_networks import CategoricalProjectionNetwork
from alf.networks.encoding_networks import EncodingNetwork
from alf.data_structures import namedtuple

DualActorValueNetworkState = namedtuple(
    'DualActorValueNetworkState', ['actor', 'value', 'aux'], default_value=())


@alf.configurable
class DualActorValueNetwork(Network):
    def __init__(self,
                 observation_spec: TensorSpec,
                 action_spec: TensorSpec,
                 encoding_network_ctor: callable = EncodingNetwork,
                 is_sharing_encoder: bool = False,
                 name='DualActorValueNetwork'):
        super().__init__(input_tensor_spec=observation_spec, name=name)

        self._actor_encoder = encoding_network_ctor(
            input_tensor_spec=observation_spec)

        self._actor_head = alf.nn.Sequential(
            self._actor_encoder,
            # TODO(breakds): Consider continuous cases as well
            CategoricalProjectionNetwork(
                input_size=self._actor_encoder.output_spec.shape[0],
                action_spec=action_spec))

        # Like the value head Aux head is outputing value estimation
        self._aux_head = alf.nn.Sequential(
            self._actor_encoder,
            alf.layers.FC(
                input_size=self._actor_encoder.output_spec.shape[0],
                output_size=1), alf.layers.Reshape(shape=()))

        if is_sharing_encoder:
            self._value_head = alf.nn.Sequential(
                self._actor_encoder, alf.layers.Detach(),
                alf.layers.FC(
                    input_size=self._actor_encoder.output_spec.shape[0],
                    output_size=1), alf.layers.Reshape(shape=()))
        else:
            value_encoder = encoding_network_ctor(
                input_tensor_spec=observation_spec)
            self._value_head = alf.nn.Sequential(
                value_encoder,
                alf.layers.FC(
                    input_size=self._actor_encoder.output_spec.shape[0],
                    output_size=1), alf.layers.Reshape(shape=()))

    def forward(self, observation, state: DualActorValueNetworkState):
        action_distribution, actor_state = self._actor_head(
            observation, state=state.actor)

        value, value_state = self._value_head(observation, state=state.value)

        aux, aux_state = self._value_head(observation, state=state.aux)

        return action_distribution, value, aux, DualActorValueNetworkState(
            actor=actor_state, value=value_state, aux=aux_state)

    @property
    def state_spec(self):
        return DualActorValueNetworkState(
            actor=self._actor_head.state_spec,
            value=self._value_head.state_spec,
            aux=self._aux_head.state_spec)
