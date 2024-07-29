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
"""DynamicsNetwork"""

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import alf
import alf.utils.math_ops as math_ops
import alf.nest as nest
from alf.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec

from .network import Network
from .encoding_networks import EncodingNetwork
from .projection_networks import NormalProjectionNetwork


@alf.configurable
class DynamicsNetwork(Network):
    """Create an instance of DynamicsNetwork."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 joint_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 prob=False,
                 continuous_projection_net_ctor=NormalProjectionNetwork,
                 name="DynamicsNetwork"):
        """Creates an instance of `DynamicsNetwork` for predicting the next
        observation given current observation and action.

        Args:
            input_tensor_spec: A tuple of TensorSpecs (observation_spec, action_spec)
                representing the inputs.
            joint_fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a variance_scaling_initializer
                with uniform distribution will be used.
            prob (bool): If True, use the probabistic mode of network; otherwise,
                use the deterministic mode of network.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                a distribution.
            name (str):
        """
        super().__init__(input_tensor_spec, name=name)

        observation_spec, action_spec = input_tensor_spec
        out_size = output_tensor_spec.shape[0]

        flat_action_spec = nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                gain=1.0 / 2.0,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=math_ops.identity)

        self._single_action_spec = flat_action_spec[0]

        self._prob = prob
        if self._prob:
            self._joint_encoder = EncodingNetwork(
                TensorSpec(
                    (observation_spec.shape[0] + action_spec.shape[0], )),
                fc_layer_params=joint_fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer)

            # the output spec is named as ``action_spec`` in projection_net
            self._projection_net = continuous_projection_net_ctor(
                # note that in the case of multi-replica, should use [-1]
                input_size=self._joint_encoder.output_spec.shape[-1],
                action_spec=output_tensor_spec,
                squash_mean=False,
                scale_distribution=False,
                state_dependent_std=True)
        else:
            self._joint_encoder = EncodingNetwork(
                TensorSpec(
                    (observation_spec.shape[0] + action_spec.shape[0], )),
                fc_layer_params=joint_fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
                last_activation=math_ops.identity,
                last_layer_size=out_size)
            self._projection_net = None

        self._output_spec = TensorSpec((out_size, ))

    def forward(self, inputs, state=()):
        """Computes prediction given inputs.

        Args:
            inputs:  A tuple of Tensors consistent with `input_tensor_spec`
            state: empty for API consistency

        Returns:
            out: a tensor of the size [B, n, d] if self._prob is False
                and a distribution if self._prob is True.
            state: empty
        """
        observations, actions = inputs
        encoded_obs = observations
        encoded_action = actions
        joint = torch.cat([encoded_obs, encoded_action], -1)
        out, _ = self._joint_encoder(joint)
        if self._projection_net is not None:
            out, _ = self._projection_net(out)

        return out, state

    def make_parallel(self, n):
        """Create a ``ParallelCriticNetwork`` using ``n`` replicas of ``self``.
        The initialized network parameters will be different.
        """
        return ParallelDynamicsNetwork(self, n, "parallel_" + self._name)


class ParallelDynamicsNetwork(Network):
    """Create ``n`` DynamicsNetwork in parallel."""

    def __init__(self,
                 dynamics_network: DynamicsNetwork,
                 n: int,
                 name="ParallelDynamicsNetwork"):
        """
        It create a parallelized version of ``DynamicsNetwork``.

        Args:
            dynamics_network (DynamicsNetwork): non-parallelized dynamics network
            n (int): make ``n`` replicas from ``dynamics_network`` with different
                initializations.
            name (str):
        """
        super().__init__(
            input_tensor_spec=dynamics_network.input_tensor_spec, name=name)
        self._joint_encoder = dynamics_network._joint_encoder.make_parallel(
            n, True)
        self._prob = dynamics_network._prob
        if self._prob:
            self._projection_net = \
                            dynamics_network._projection_net.make_parallel(n)
        else:
            self._projection_net = None

        self._output_spec = TensorSpec((n, ) +
                                       dynamics_network.output_spec.shape)

    def forward(self, inputs, state=()):
        """Computes prediction given inputs.

        Args:
            inputs:  A tuple of Tensors consistent with `input_tensor_spec`
            state: empty for API consistency

        Returns:
            out: a tensor of the size [B, n, d] if self._prob is False
                and a distribution if self._prob is True.
            state: empty
        """
        observations, actions = inputs
        encoded_obs = observations
        encoded_action = actions
        joint = torch.cat([encoded_obs, encoded_action], -1)
        out, _ = self._joint_encoder(joint)
        if self._projection_net is not None:
            out, _ = self._projection_net(out)

        return out, state
