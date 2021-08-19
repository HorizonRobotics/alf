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

from typing import Callable, Tuple

import torch
import alf
from alf.tensor_specs import TensorSpec
from alf.data_structures import namedtuple
from .network import Network
from .projection_networks import NormalProjectionNetwork, CategoricalProjectionNetwork
from .encoding_networks import EncodingNetwork


def _create_projection_net_based_on_action_spec(
        discrete_projection_net_ctor: Callable[..., Network],
        continuous_projection_net_ctor: Callable[..., Network],
        input_size: int, action_spec):
    """Create project network(s) for the potentially nested action spec.

    This function basically creates a projection network for each of the leaf
    tensor spec in the action spec. Those networks are packed into the same
    nested structure as the input action spec and returned as a whole.

    Args:

        discrete_projection_net_ctor (Callable[..., Network]): constructor that
            generates a discrete projection network that outputs discrete
            actions.

        continuous_projection_net_ctor (Callable[..., Network): constructor that
            generates a continuous projection network that outputs continuous
            actions.

        input_size (int): the input_size for the projection network, which usually
            comes from the output of an encoding network.

        action_spec (nest of TensorSpec): speficifies the shape and type of the
            output action. The type of each invidual projection network in the
            output is derived from this.

    """

    def _create_individually(spec):
        constructor = (discrete_projection_net_ctor
                       if spec.is_discrete else continuous_projection_net_ctor)
        return constructor(input_size=input_size, action_spec=spec)

    return alf.nest.map_structure(_create_individually, action_spec)


# TODO(breakds): Add its recurrent network counterpart
@alf.configurable
class DisjointPolicyValueNetwork(Network):
    """A composite network with a policy component and a value component.

    This network capture a category of network as proposed in the Phasic Policy
    Gradient paper. It consists of two components and 3 heads:

    - Value Component: a single value head that estimates the value function

    - Policy Component: 1 policy head that outputs the action distribution, and
         1 auxiliary value head that behaves as a secondary value function
         estimator

    The output of this network is a triplet, corresponding to the 3 heads in the
    order of (action distribution, value function, auxiliary value function).

    About Architecture:

      The Value Component and the Policy Component may share the same encoding
      network or have their own encoding network. When the encoding network is
      shared, it is called the "shared" architecture. If the encoding network is
      not shared, it is called the "dual" architecture.

      NOTE that in the "shared" architecture, the encoder is detached before
      connecting to the value head. This means that the value head will have no
      power to optimize and update the parameters of the encoder under such
      constraint.

    See https://github.com/HorizonRobotics/alf/issues/965 for a graphical
    illustration of such two different architectures.

    """

    # TODO(breakds): Add type hints when nest of tensor type is defined
    def __init__(self,
                 observation_spec,
                 action_spec,
                 encoding_network_ctor=EncodingNetwork,
                 is_sharing_encoder: bool = False,
                 kernel_initializer=None,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=NormalProjectionNetwork,
                 name='DisjointPolicyValueNetwork'):
        """The constructor of DisjointPolicyValueNetwork

        Note that there are two projection constructor parameters. They exist
        because in the case when the action spec is a nest of different types
        where some of them are discrete and some of them are continuous,
        corresponding projection networks can be created for the two parties
        individually and respectively.

        Args:

            observation_spec (nest of TesnorSpec): specifies the shape and type
                of the input observation.

            action_spec (nest of TensorSpec): speficifies the shape and type of
                the output action. The type of output action distribution is
                implicitly derived from this.

            encoding_network_ctor (Callable[..., Network]): A constructor that
                creates the encoding network. Depending whether the encoding
                network is shared between the value component and the policy
                component, 1 or 2 encoding network will be created using this
                constructor.

            is_sharing_encoder (bool): When set to true, the encoding network is
                shared between the value and the policy component, resulting in
                a "shared" architecture disjoint network. When set to false, the
                encoding network is not shared, resulting in a "dual"
                architecture disjoint network.

            kernel_initializer (Callable): initializer for all the layers
                excluding the projection net. If none is provided a default
                xavier_uniform will be used.

            discrete_projection_net_ctor (Callable[..., Network]): constructor
                that generates a discrete projection network that outputs
                discrete actions.

            continuous_projection_net_ctor (Callable[..., Network): constructor
                that generates a continuous projection network that outputs
                continuous actions.

            name(str): the name of the network

        """
        super().__init__(input_tensor_spec=observation_spec, name=name)

        # +------------------------------------+
        # | Step 1: The policy network encoder |
        # +------------------------------------+

        if kernel_initializer is None:
            kernel_initializer = torch.nn.init.xavier_uniform_

        self._actor_encoder = encoding_network_ctor(
            input_tensor_spec=observation_spec,
            kernel_initializer=kernel_initializer)

        # +------------------------------------------+
        # | Step 2: Projection for the policy branch |
        # +------------------------------------------+

        self._policy_head = _create_projection_net_based_on_action_spec(
            discrete_projection_net_ctor=discrete_projection_net_ctor,
            continuous_projection_net_ctor=continuous_projection_net_ctor,
            input_size=self._actor_encoder.output_spec.shape[0],
            action_spec=action_spec)
        # In the case when the projection net is multi-fold (because actor
        # spec is a nest), force picking up the parameters inside those
        # project networks into this DisjointPolicyValueNetwork instance.
        self._projection_net_module_list = torch.nn.ModuleList(
            alf.nest.flatten(self._policy_head))

        # +------------------------------------------+
        # | Step 3: Value head of the aux branch     |
        # +------------------------------------------+

        # Note that the aux branch value head belongs to the policy component.

        # Like the value head Aux head is outputing value estimation
        self._aux_head = alf.nn.Sequential(
            alf.layers.FC(
                input_size=self._actor_encoder.output_spec.shape[0],
                output_size=1), alf.layers.Reshape(shape=()))

        # +------------------------------------------+
        # | Step 4: Assemble network + value head    |
        # +------------------------------------------+

        if is_sharing_encoder:
            # Use the same encoder, but the encoder is DETACHED.
            self._value_head = alf.nn.Sequential(
                alf.layers.Detach(),
                alf.layers.FC(
                    input_size=self._actor_encoder.output_spec.shape[0],
                    output_size=1),
                alf.layers.Reshape(shape=()),
                input_tensor_spec=self._actor_encoder.output_spec)

            all_branches = ((self._policy_head, self._aux_head),
                            self._value_head)
            self._composition = alf.nn.Sequential(
                self._actor_encoder,
                alf.nn.Branch(all_branches, name='AllBranches'))
        else:
            policy_component = alf.nn.Sequential(
                self._actor_encoder,
                alf.nn.Branch((self._policy_head, self._aux_head),
                              name='PolicyComponent'))

            # When not sharing encoder, create a separate encoder for the value
            # component.
            self._value_encoder = encoding_network_ctor(
                input_tensor_spec=observation_spec,
                kernel_initializer=kernel_initializer)

            self._value_head = alf.nn.Sequential(
                alf.layers.FC(
                    input_size=self._actor_encoder.output_spec.shape[0],
                    output_size=1), alf.layers.Reshape(shape=()))

            value_component = alf.nn.Sequential(self._value_encoder,
                                                self._value_head)

            self._composition = alf.nn.Branch(
                (policy_component, value_component), name='AllBranches')

    def forward(self, observation, state):
        """Computes the action distribution, aux value and value estimation

        Args:

            observation (nested torch.Tensor): a tensor that is consistent with
                the encoding network

            state: the state for RNN based network for interface compatibility,
                which is always set to empty in this case because this network
                does not have an RNN-based encoder

        Returns:
            act_dist (torch.distributions): action distribution
            state: empty

        """
        ((action_distribution, aux), value), _ = self._composition(
            observation, state=())
        return (action_distribution, value, aux), ()
