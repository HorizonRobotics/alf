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
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.data_structures import namedtuple
from alf.networks import Network, NormalProjectionNetwork, CategoricalProjectionNetwork, EncodingNetwork


def _create_projection_net_based_on_action_spec(
        discrete_projection_net_ctor: Callable[[int, BoundedTensorSpec],
                                               Network],
        continuous_projection_net_ctor: Callable[
            [int, BoundedTensorSpec], Network], input_size: int, action_spec):
    """Create project network(s) for the potentially nested action spec.

    This function basically creates a projection network for each of the leaf
    tensor spec in the action spec. Those networks are packed into the same
    nested structure as the input action spec and returned as a whole.

    Args:

        discrete_projection_net_ctor (Callable[[int, BoundedTensorSpec],
            Network]): constructor that generates a discrete projection network
            that outputs discrete actions.
        continuous_projection_net_ctor (Callable[[int, BoundedTensorSpec],
            Network): constructor that generates a continuous projection network
            that outputs continuous actions.
        input_size (int): the input_size for the projection network, which usually
            comes from the output of an encoding network.
        action_spec (nest of TensorSpec): specifies the shape and type of the
            output action. The type of each individual projection network in the
            output is derived from this.

    """

    def _create_individually(spec):
        constructor = (discrete_projection_net_ctor
                       if spec.is_discrete else continuous_projection_net_ctor)
        return constructor(input_size=input_size, action_spec=spec)

    return alf.nest.map_structure(_create_individually, action_spec)


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

    NOTE:

    1. The is_sharing_encoder = True situation corresponds to the 'detached'
       arch in OpenAI's implementation and the Single-Network PPG in the
       original paper. However, OpenAI's implementation and paper has an
       important difference regarding this. In the paper, it reads (quoted):

       During the policy phase, we detach the value function gradient at the
       last layer shared between the policy and value heads, preventing the
       value function gradient from influencing shared parameters. During the
       auxiliary phase, we take the value function gradient with respect to all
       parameters, including shared parameters.

       In their implementation, the "true" (as opposed to the aux) value head is
       always detached, in both policy and aux phase.

       Our implementation follows the OpenAI's implementation, which keeps the
       true value head always detached.

    2. In OpenAI's implementation, the FC and Conv layers are initialized in a
       non-standard way. Here in our implementation we initialize such layers
       with standard approaches.

    """

    # TODO(breakds): Add type hints when nest of tensor type is defined
    def __init__(self,
                 observation_spec,
                 action_spec,
                 encoding_network_ctor=EncodingNetwork,
                 is_sharing_encoder: bool = False,
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
            action_spec (nest of TensorSpec): specifies the shape and type of
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
            discrete_projection_net_ctor (Callable[[int, BoundedTensorSpec],
                Network]): constructor that generates a discrete projection
                network that outputs discrete actions.
            continuous_projection_net_ctor (Callable[[int, BoundedTensorSpec],
                Network): constructor that generates a continuous projection
                network that outputs continuous actions.
            name(str): the name of the network

        """
        super().__init__(input_tensor_spec=observation_spec, name=name)

        # +------------------------------------+
        # | Step 1: The policy network encoder |
        # +------------------------------------+

        self._actor_encoder = encoding_network_ctor(
            input_tensor_spec=observation_spec)

        encoder_output_size = self._actor_encoder.output_spec.shape[0]

        # +------------------------------------------+
        # | Step 2: Projection for the policy branch |
        # +------------------------------------------+

        self._policy_head = _create_projection_net_based_on_action_spec(
            discrete_projection_net_ctor=discrete_projection_net_ctor,
            continuous_projection_net_ctor=continuous_projection_net_ctor,
            input_size=encoder_output_size,
            action_spec=action_spec)

        # +------------------------------------------+
        # | Step 3: Value head of the aux branch     |
        # +------------------------------------------+

        # Note that the aux branch value head belongs to the policy component.

        # Like the value head Aux head is outputting value estimation
        self._aux_head = alf.nn.Sequential(
            alf.layers.FC(input_size=encoder_output_size, output_size=1),
            alf.layers.Reshape(()))

        # +------------------------------------------+
        # | Step 4: Assemble network + value head    |
        # +------------------------------------------+

        if is_sharing_encoder:
            self._composition = alf.nn.Sequential(
                self._actor_encoder,
                alf.nn.Branch(
                    self._policy_head,
                    alf.nn.Sequential(
                        # Use the same encoder, but the encoder is DETACHED.
                        alf.layers.Detach(),
                        alf.layers.FC(
                            input_size=encoder_output_size, output_size=1),
                        alf.layers.Reshape(()),
                        input_tensor_spec=self._actor_encoder.output_spec),
                    alf.layers.Identity()))
        else:
            # When not sharing encoder, create a separate encoder for the value
            # component.
            self._value_encoder = encoding_network_ctor(
                input_tensor_spec=observation_spec)

            self._composition = alf.nn.Sequential(
                alf.nn.Branch(
                    alf.nn.Sequential(
                        self._actor_encoder,
                        alf.nn.Branch(
                            self._policy_head,
                            alf.layers.Identity(),
                            name='PolicyComponent')),
                    alf.nn.Sequential(
                        self._value_encoder,
                        alf.layers.FC(
                            input_size=encoder_output_size, output_size=1),
                        alf.layers.Reshape(()))),
                # Order: policy, value, aux value
                lambda heads: (heads[0][0], heads[1], heads[0][1]))

    def forward(self, observation, state, require_aux: bool = True):
        """Computes the action distribution, aux value and value estimation

        In PPG's policy phase update, auxiliary estimation is not needed as it
        does not participate in computing the policy phase loss. Depending on
        whether require_aux is set to True or False, forward will choose to
        compute auxiliary value estimation or not accordingly.

        NOTE: Although by not computing the auxiliary value estimation it saves
        a tiny bit of computation, the main reason we want to prevent it from
        being computed for PPG's policy phase is to make PPG work with DDP (Data
        Distributed Parallel). DDP need to wait for all parameters that
        contributes to the output of ``forward()`` to go through ``backward()``.
        If auxiliary value estimation were computed, DDP will panic since it
        will not go through ``backward()`` in the policy phase update.

        Args:
            observation (nested torch.Tensor): a tensor that is consistent with
                the encoding network
            state: the state(s) for RNN based network
            require_aux: When set to False, return () as the auxiliary value
                estimation in the output.
        Returns:
            output (Triplet): network output in the order of policy (action
                distribution), value function estimation, auxiliary value
                function estimation
            state (Triplet): RNN states in the order of policy, value, aux value

        """
        (action_distribution, value,
         encoded), output_state = self._composition(
             observation, state=state)

        if require_aux:
            aux, _ = self._aux_head(encoded)
            return (action_distribution, value, aux), output_state
        else:
            return (action_distribution, value, ()), output_state
