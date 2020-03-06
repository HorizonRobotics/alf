# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Some basic layers."""

import gin

import torch
import torch.nn as nn

from alf.networks.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec


def identity(x):
    """PyTorch doesn't have an identity activation. This can be used as a
    placeholder.
    """
    return x


@gin.configurable
class FC(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 activation=identity,
                 use_bias=True,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A fully connected layer that's also responsible for activation and
        customized weights initialization. An auto gain calculation might depend
        on the activation following the linear layer. Suggest using this wrapper
        module instead of nn.Linear if you really care about weight std after
        init.

        Args:
            input_size (int): input size
            output_size (int): output size
            activation (torch.nn.functional):
            use_bias (bool): whether use bias
            kernel_initializer (Callable): initializer for all the layers
            kernel_init_gain (float): a scaling factor (gain) applied to
                the std of kernel init distribution
            bias_init_value (float): a constant
        """
        super(FC, self).__init__()
        self._activation = activation
        self._linear = nn.Linear(input_size, output_size, bias=use_bias)
        # variance_scaling_init(
        #     self._linear.weight.data,
        #     gain=kernel_init_gain,
        #     nonlinearity=self._activation.__name__)
        # nn.init.xavier_normal_(self._linear.weight)
        if kernel_initializer is None:
            variance_scaling_init(
                self._linear.weight.data,
                gain=kernel_init_gain,
                nonlinearity=self._activation.__name__)
        else:
            kernel_initializer(self._linear.weight)

        if use_bias:
            nn.init.constant_(self._linear.bias.data, bias_init_value)
            #nn.init.zeros_(self._linear.bias.data)

    def forward(self, inputs):
        return self._activation(self._linear(inputs))

    @property
    def weight(self):
        return self._linear.weight

    @property
    def bias(self):
        return self._linear.bias


@gin.configurable
class Conv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=torch.relu,
                 strides=1,
                 padding=0,
                 use_bias=True,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A 2D Conv layer that's also responsible for activation and customized
        weights initialization. An auto gain calculation might depend on the
        activation following the conv layer. Suggest using this wrapper module
        instead of nn.Conv2d if you really care about weight std after init.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            activation (torch.nn.functional):
            strides (int or tuple):
            padding (int or tuple):
            use_bias (bool):
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution
            bias_init_value (float): a constant
        """
        super(Conv2D, self).__init__()
        self._activation = activation
        self._conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=strides,
            padding=padding,
            bias=use_bias)
        variance_scaling_init(
            self._conv2d.weight.data,
            gain=kernel_init_gain,
            nonlinearity=self._activation.__name__)
        if use_bias:
            nn.init.constant_(self._conv2d.bias.data, bias_init_value)

    def forward(self, img):
        return self._activation(self._conv2d(img))

    @property
    def weight(self):
        return self._conv2d.weight

    @property
    def bias(self):
        return self._conv2d.bias


@gin.configurable
class ConvTranspose2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=torch.relu,
                 strides=1,
                 padding=0,
                 use_bias=True,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A 2D ConvTranspose layer that's also responsible for activation and
        customized weights initialization. An auto gain calculation might depend
        on the activation following the conv layer. Suggest using this wrapper
        module instead of nn.ConvTranspose2d if you really care about weight std
        after init.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            activation (torch.nn.functional):
            strides (int or tuple):
            padding (int or tuple):
            use_bias (bool):
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution
            bias_init_value (float): a constant
        """
        super(ConvTranspose2D, self).__init__()
        self._activation = activation
        self._conv_trans2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=strides,
            padding=padding,
            bias=use_bias)
        variance_scaling_init(
            self._conv_trans2d.weight.data,
            gain=kernel_init_gain,
            nonlinearity=self._activation.__name__,
            transposed=True)
        if use_bias:
            nn.init.constant_(self._conv_trans2d.bias.data, bias_init_value)

    def forward(self, img):
        return self._activation(self._conv_trans2d(img))

    @property
    def weight(self):
        return self._conv_trans2d.weight

    @property
    def bias(self):
        return self._conv_trans2d.bias
