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
"""Networks for Hypernetwork Algorithm."""

import gin
import torch
import torch.nn as nn

from alf.algorithms.hypernetwork_layers import ParamFC, ParamConv2D
from alf.networks.network import Network
from alf.tensor_specs import TensorSpec
from alf.utils import common


@gin.configurable
class ParamConvNet(Network):
    """A convolutional network with input network parameters. """

    def __init__(self,
                 input_channels,
                 input_size,
                 conv_layer_params,
                 same_padding=False,
                 pooling_kernel=None,
                 activation=torch.relu_,
                 use_bias=True,
                 flatten_output=False,
                 name="ParamConvNet"):

        input_size = common.tuplify2d(input_size)
        super().__init__(
            input_tensor_spec=TensorSpec((input_channels, ) + input_size),
            name=name)

        assert isinstance(conv_layer_params, tuple)
        assert len(conv_layer_params) > 0

        self._flatten_output = flatten_output
        self._conv_layer_params = conv_layer_params
        self._conv_layers = nn.ModuleList()
        self._param_length = None
        for paras in conv_layer_params:
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            if same_padding:  # overwrite paddings
                kernel_size = common.tuplify2d(kernel_size)
                padding = ((kernel_size[0] - 1) // 2,
                           (kernel_size[1] - 1) // 2)
            self._conv_layers.append(
                ParamConv2D(
                    input_channels,
                    filters,
                    kernel_size,
                    activation=activation,
                    strides=strides,
                    pooling_kernel=pooling_kernel,
                    padding=padding,
                    use_bias=use_bias))
            input_channels = filters

    @property
    def param_length(self):
        if self._param_length is None:
            length = 0
            for conv_l in self._conv_layers:
                length = length + conv_l.weight_length + conv_l.bias_length
            self._param_length = length
        return self._param_length

    def set_parameters(self, theta):
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        pos = 0
        for conv_l in self._conv_layers:
            weight_length = conv_l.weight_length
            conv_l.set_weight(theta[:, pos:pos + weight_length])
            pos = pos + weight_length
            if conv_l.bias is not None:
                bias_length = conv_l.bias_length
                conv_l.set_bias(theta[:, pos:pos + bias_length])
                pos = pos + bias_length

    def forward(self, inputs, state=()):
        """The empty state just keeps the interface same with other networks."""
        x = inputs
        for conv_l in self._conv_layers[:-1]:
            x = conv_l(x, keep_group_dim=False)
        x = self._conv_layers[-1](x)
        if self._flatten_output:
            x = x.reshape(*x.shape[:-3], -1)
        return x, state


@gin.configurable
class ParamNetwork(Network):
    """ParamNetwork

    """

    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 last_layer_size=None,
                 last_activation=None,
                 name="ParamNetwork"):

        super().__init__(input_tensor_spec=input_tensor_spec, name=name)

        self._param_length = None
        self._conv_net = None
        if conv_layer_params:
            assert isinstance(conv_layer_params, tuple), \
                "The input params {} should be tuple".format(conv_layer_params)
            assert len(input_tensor_spec.shape) == 3, \
                "The input shape {} should be like (C,H,W)!".format(
                    input_tensor_spec.shape)
            input_channels, height, width = input_tensor_spec.shape
            self._conv_net = ParamConvNet(
                input_channels, (height, width),
                conv_layer_params,
                activation=activation,
                flatten_output=True)
            input_size = self._conv_net.output_spec.shape[0]
        else:
            assert input_tensor_spec.ndim == 1, \
                "The input shape {} should be like (N,)!".format(
                    input_tensor_spec.shape)
            input_size = input_tensor_spec.shape[0]

        self._fc_layers = nn.ModuleList()
        if fc_layer_params is None:
            fc_layer_params = []
        else:
            assert isinstance(fc_layer_params, tuple)
            fc_layer_params = list(fc_layer_params)

        for size in fc_layer_params:
            self._fc_layers.append(
                ParamFC(input_size, size, activation=activation))
            input_size = size

        if last_layer_size is not None or last_activation is not None:
            assert last_layer_size is not None and last_activation is not None, \
            "Both last_layer_size and last_activation need to be specified!"

            self._fc_layers.append(
                ParamFC(
                    input_size, last_layer_size, activation=last_activation))
            input_size = last_layer_size

        self._output_spec = TensorSpec((input_size, ),
                                       dtype=self._input_tensor_spec.dtype)

    @property
    def param_length(self):
        if self._param_length is None:
            length = 0
            if self._conv_net is not None:
                length += self._conv_net.param_length
            for fc_l in self._fc_layers:
                length = length + fc_l.weight_length + fc_l.bias_length
            self._param_length = length
        return self._param_length

    def set_parameters(self, theta):
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        conv_theta = theta[:, :self._conv_net.param_length]
        fc_theta = theta[:, self._conv_net.param_length:]
        self._conv_net.set_parameters(conv_theta)

        pos = 0
        for fc_l in self._fc_layers:
            weight_length = fc_l.weight_length
            fc_l.set_weight(fc_theta[:, pos:pos + weight_length])
            pos = pos + weight_length
            if fc_l.bias is not None:
                bias_length = fc_l.bias_length
                fc_l.set_bias(fc_theta[:, pos:pos + bias_length])
                pos = pos + bias_length

    def forward(self, inputs, state=()):
        """The empty state just keeps the interface same with other networks."""
        if self._conv_net is not None:
            x, state = self._conv_net(inputs, state=state)
        for fc_l in self._fc_layers:
            x = fc_l(x)
        return x, state
