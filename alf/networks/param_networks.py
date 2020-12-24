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
"""Networks with input parameters."""

import functools
import gin
import torch
import torch.nn as nn

from alf.initializers import variance_scaling_init
from alf.layers import ParamFC, ParamConv2D
from alf.networks.network import Network
from alf.tensor_specs import TensorSpec
from alf.utils import common


@gin.configurable
class ParamConvNet(Network):
    def __init__(self,
                 input_channels,
                 input_size,
                 conv_layer_params,
                 same_padding=False,
                 activation=torch.relu_,
                 use_bias=False,
                 kernel_initializer=None,
                 flatten_output=False,
                 name="ParamConvNet"):
        """A fully 2D conv network that does not maintain its own network parameters,
        but accepts them from users. If the given parameter tensor has an extra batch
        dimension (first dimension), it performs parallel operations.

        Args:
            input_channels (int): number of channels in the input image
            input_size (int or tuple): the input image size (height, width)
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format 
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            same_padding (bool): similar to TF's conv2d ``same`` padding mode. If
                True, the user provided paddings in `conv_layer_params` will be
                replaced by automatically calculated ones; if False, it
                corresponds to TF's ``valid`` padding mode (the user can still
                provide custom paddings though)
            activation (torch.nn.functional): activation for all the layers
            use_bias (bool): whether use bias
            kernel_initializer (Callable): initializer for all the layers.
            flatten_output (bool): If False, the output will be an image
                structure of shape ``(B, n, C, H, W)``; otherwise the output
                will be flattened into a feature of shape ``(B, n, C*H*W)``.
            name (str):
        """

        input_size = common.tuplify2d(input_size)
        super().__init__(
            input_tensor_spec=TensorSpec((input_channels, ) + input_size),
            name=name)

        assert isinstance(conv_layer_params, tuple)
        assert len(conv_layer_params) > 0

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        self._flatten_output = flatten_output
        self._conv_layer_params = conv_layer_params
        self._conv_layers = nn.ModuleList()
        self._param_length = None
        for paras in conv_layer_params:
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            pooling_kernel = paras[4] if len(paras) > 4 else None
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
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer))
            input_channels = filters

    @property
    def param_length(self):
        """Get total number of parameters for all layers. """
        if self._param_length is None:
            length = 0
            for conv_l in self._conv_layers:
                length = length + conv_l.weight_length + conv_l.bias_length
            self._param_length = length
        return self._param_length

    def set_parameters(self, theta, reinitialize=False):
        """Distribute parameters to corresponding layers. """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        pos = 0
        for conv_l in self._conv_layers:
            weight_length = conv_l.weight_length
            conv_l.set_weight(
                theta[:, pos:pos + weight_length], reinitialize=reinitialize)
            pos = pos + weight_length
            if conv_l.bias is not None:
                bias_length = conv_l.bias_length
                conv_l.set_bias(
                    theta[:, pos:pos + bias_length], reinitialize=reinitialize)
                pos = pos + bias_length
        self._output_spec = None

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.
        """
        x = inputs
        for conv_l in self._conv_layers[:-1]:
            x = conv_l(x, keep_group_dim=False)
        x = self._conv_layers[-1](x)
        if self._flatten_output:
            x = x.reshape(*x.shape[:-3], -1)
        return x, state


@gin.configurable
class ParamNetwork(Network):
    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 last_layer_param=None,
                 last_activation=None,
                 name="ParamNetwork"):
        """A network with Fc and conv2D layers that does not maintain its own 
        network parameters, but accepts them from users. If the given parameter 
        tensor has an extra batch dimension (first dimension), it performs 
        parallel operations.

        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format 
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where 
                ``use_bias`` is optional.
            activation (torch.nn.functional): activation for all the layers
            kernel_initializer (Callable): initializer for all the layers.
            last_layer_param (tuple): an optional tuple of the format
                ``(size, use_bias)``, where ``use_bias`` is optional,
                it appends an additional layer at the very end. 
                Note that if ``last_activation`` is specified, 
                ``last_layer_param`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.
            name (str):
        """

        super().__init__(input_tensor_spec=input_tensor_spec, name=name)

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        self._param_length = None
        self._conv_net = None
        if conv_layer_params:
            assert isinstance(conv_layer_params, tuple), \
                "The input params {} should be tuple".format(conv_layer_params)
            assert input_tensor_spec.ndim == 3, \
                "The input shape {} should be like (C,H,W)!".format(
                    input_tensor_spec.shape)
            input_channels, height, width = input_tensor_spec.shape
            self._conv_net = ParamConvNet(
                input_channels, (height, width),
                conv_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
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

        for params in fc_layer_params:
            size = params[0]
            if len(params) > 1:
                self._fc_layers.append(
                    ParamFC(
                        input_size,
                        size,
                        activation=activation,
                        use_bias=params[1],
                        kernel_initializer=kernel_initializer))
            else:
                self._fc_layers.append(
                    ParamFC(
                        input_size,
                        size,
                        activation=activation,
                        kernel_initializer=kernel_initializer))
            input_size = size

        if last_layer_param is not None or last_activation is not None:
            assert last_layer_param is not None and last_activation is not None, \
            "Both last_layer_param and last_activation need to be specified!"

            last_layer_size = last_layer_param[0]
            if len(last_layer_param) > 1:
                self._fc_layers.append(
                    ParamFC(
                        input_size,
                        last_layer_size,
                        activation=last_activation,
                        use_bias=last_layer_param[1],
                        kernel_initializer=kernel_initializer))
            else:
                self._fc_layers.append(
                    ParamFC(
                        input_size,
                        last_layer_size,
                        activation=last_activation,
                        kernel_initializer=kernel_initializer))
            input_size = last_layer_size

        self._output_spec = TensorSpec((input_size, ),
                                       dtype=self._input_tensor_spec.dtype)

    @property
    def param_length(self):
        """Get total number of parameters for all layers. """
        if self._param_length is None:
            length = 0
            if self._conv_net is not None:
                length += self._conv_net.param_length
            for fc_l in self._fc_layers:
                length = length + fc_l.weight_length + fc_l.bias_length
            self._param_length = length
        return self._param_length

    def set_parameters(self, theta, reinitialize=False):
        """Distribute parameters to corresponding layers. """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert (theta.ndim == 2 and theta.shape[1] == self.param_length), (
            "Input theta has wrong shape %s. Expecting shape (, %d)" %
            self.param_length)
        if self._conv_net is not None:
            split = self._conv_net.param_length
            conv_theta = theta[:, :split]
            self._conv_net.set_parameters(
                conv_theta, reinitialize=reinitialize)
            fc_theta = theta[:, self._conv_net.param_length:]
        else:
            fc_theta = theta

        pos = 0
        for fc_l in self._fc_layers:
            weight_length = fc_l.weight_length
            fc_l.set_weight(
                fc_theta[:, pos:pos + weight_length],
                reinitialize=reinitialize)
            pos = pos + weight_length
            if fc_l.bias is not None:
                bias_length = fc_l.bias_length
                fc_l.set_bias(
                    fc_theta[:, pos:pos + bias_length],
                    reinitialize=reinitialize)
                pos = pos + bias_length
        # self._output_spec = None

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (Tensor):
            state: not used, just keeps the interface same with other networks.
        """
        x = inputs
        if self._conv_net is not None:
            x, state = self._conv_net(x, state=state)
        for fc_l in self._fc_layers:
            x = fc_l(x)
        return x, state
