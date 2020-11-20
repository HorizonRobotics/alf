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

import gin
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from alf.initializers import variance_scaling_init
import alf.nest as nest
from alf.networks.param_networks import ParamNetwork
from alf.tensor_specs import TensorSpec
import alf.utils.math_ops as math_ops
from .network import Network


@gin.configurable
class ParamDynamicsNetwork(ParamNetwork):
    """Create an instance of DynamicsNetwork."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 name="ParamDynamicsNetwork"):
        """Creates an instance of `DynamicsNetwork` for predicting the next
        observation given current observation and action.

        Args:
            input_tensor_spec: A tuple of TensorSpecs (observation_spec, action_spec)
                representing the inputs.
            fc_layer_params (tuple[int]): a tuple of integers representing
                hidden FC layer sizes FC layers after merging observations and
                actions.
            activation (nn.functional): activation used for hidden layers. The
                last layer will not be activated.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If none is provided a variance_scaling_initializer
                with uniform distribution will be used.
            prob (bool): If True, use the probabistic mode of network; otherwise,
                use the determinstic mode of network.
            continuous_projection_net_ctor (ProjectionNetwork): constructor that
                generates a continuous projection network that outputs
                a distribution.
            name (str):
        """

        observation_spec, action_spec = input_tensor_spec
        output_size = output_tensor_spec.shape[0]

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

        super().__init__(
            TensorSpec((observation_spec.shape[0] + action_spec.shape[0], )),
            fc_layer_params=fc_layer_params,
            activation=activation,
            kernel_initializer=kernel_initializer,
            last_layer_param=(output_size, ),
            last_activation=math_ops.identity)

    def forward(self, inputs, state=()):
        """Computes prediction given inputs.

        Args:
            inputs:  A tuple of Tensors consistent with `input_tensor_spec`
            state: empty for API consistency

        Returns:
            out: a tensor of the size [B, N, D].
            state: empty
        """
        observations, actions = inputs
        joint = torch.cat([observations, actions], -1)
        joint = joint.squeeze()
        out, _ = super().forward(joint)

        return out, state
