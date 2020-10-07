# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Basic layers for Hypernetwork Algorithm."""

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from alf.utils import common


@gin.configurable
class ParamFC(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 activation=torch.relu_,
                 use_bias=True):
        """A fully connected layer that does not maintain its own weight and bias,
        but accepts both from users. If the given parameter (weight and bias)
        tensor has an extra batch dimension (first dimension), it performs
        parallel FC operation.

        Args:
            input_size (int): input size
            output_size (int): output size
            activation (torch.nn.functional):
            use_bias (bool): whether use bias
        """
        super(ParamFC, self).__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation
        self._use_bias = use_bias

        self._weight_length = output_size * input_size
        self.set_weight(torch.randn(1, self._weight_length))
        if use_bias:
            self._bias_length = output_size
            self.set_bias(torch.randn(1, self._bias_length))
        else:
            self._bias_length = 0
            self._bias = None

    @property
    def weight(self):
        """Get stored weight tensor or batch of weight tensors."""
        return self._weight

    @property
    def bias(self):
        """Get stored bias tensor or batch of bias tensors."""
        return self._bias

    @property
    def weight_length(self):
        """Get the n_element of a single weight tensor. """
        return self._weight_length

    @property
    def bias_length(self):
        """Get the n_element of a single bias tensor. """
        return self._bias_length

    def set_weight(self, weight):
        """Store a weight tensor or batch of weight tensors."""
        assert (weight.ndim == 2 and weight.shape[1] == self._weight_length), (
            "Input weight has wrong shape %s. Expecting shape (n, %d)" %
            (weight.shape, self._weight_length))
        if weight.shape[0] == 1:
            # non-parallel weight
            self._groups = 1
        elif weight.ndim == 2:
            # parallel weight
            self._groups = weight.shape[0]
        self._weight = weight.view(self._groups, self._output_size,
                                   self._input_size)

    def set_bias(self, bias):
        """Store a bias tensor or batch of bias tensors."""
        assert (bias.ndim == 2 and bias.shape[1] == self._bias_length), (
            "Input bias has wrong shape %s. Expecting shape (n, %d)" %
            (bias.shape, self._bias_length))
        if self._groups == 1:
            # non-parallel bias
            assert bias.shape[0] == 1, (
                "Input bias has wrong shape %s. Expecting shape (%d, %d)" %
                (bias.shape, 1, self.bias_length))
        else:
            # parallel weight
            assert (bias.ndim == 2 and bias.shape[0] == self._groups), (
                "Input bias has wrong shape %s. Expecting shape (%d, %d)" %
                (bias.shape, self._group, self.bias_length))
        self._bias = bias  # [n, bias_length]

    def forward(self, inputs):
        """Forward

        Args:
            inputs (torch.Tensor): with shape ``[B, D] (groups=1)``
                                        or ``[B, n, D] (groups=n)``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``n``: number of replicas
                - ``D``: input dimension
                When the shape of inputs is ``[B, D]``, all the n linear
                operations will take inputs as the same shared inputs.
                When the shape of inputs is ``[B, n, D]``, each linear operator
                will have its own input data by slicing inputs.

        Returns:
            torch.Tensor with shape ``[B, n, D]`` or ``[B, D]``
                where the meaning of the symbols are:
                - ``B``: batch
                - ``n``: number of replicas
                - ``D'``: output dimension
        """
        if self._groups == 1:
            # non-parallel layer
            assert (inputs.ndim == 2
                    and inputs.shape[1] == self._input_size), (
                        "Input inputs has wrong shape %s. Expecting (B, %d)" %
                        (inputs.shape, self._input_size))
            inputs = inputs.unsqueeze(0)  # [1, B, D]
        else:
            # parallel layer
            if inputs.ndim == 2:
                # case 1: non-parallel inputs
                assert inputs.shape[1] == self._input_size, (
                    "Input inputs has wrong shape %s. Expecting (B, %d)" %
                    (inputs.shape, self._input_size))
                inputs = inputs.unsqueeze(0).expand(self._groups,
                                                    *inputs.shape)
            elif inputs.ndim == 3:
                # case 2: parallel inputs
                assert (
                    inputs.shape[1] == self._groups
                    and inputs.shape[2] == self._input_size
                ), ("Input inputs has wrong shape %s. Expecting (B, %d, %d)" %
                    (inputs.shape, self._groups, self._input_size))
                inputs = inputs.transpose(0, 1)  # [n, B, D]
            else:
                raise ValueError("Wrong inputs.ndim=%d" % inputs.ndim)

        if self._bias is not None:
            res = torch.baddbmm(
                self._bias.unsqueeze(1), inputs, self._weight.transpose(1, 2))
        else:
            res = torch.bmm(inputs, self._weight.transpose(1, 2))
        res = res.transpose(0, 1)  # [B, n, D]
        res = res.squeeze(1)  # [B, D] if n=1

        return self._activation(res)


@gin.configurable
class ParamConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=torch.relu_,
                 strides=1,
                 pooling_kernel=None,
                 padding=0,
                 use_bias=False):
        """A 2D conv layer that does not maintain its own weight and bias,
        but accepts both from users. If the given parameter (weight and bias)
        tensor has an extra batch dimension (first dimension), it performs
        parallel FC operation.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            activation (torch.nn.functional):
            strides (int or tuple):
            pooling_kernel (int or tuple):
            padding (int or tuple):
            use_bias (bool): whether use bias
        """
        super(ParamConv2D, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._activation = activation
        self._kernel_size = common.tuplify2d(kernel_size)
        self._kH, self._kW = self._kernel_size
        self._strides = strides
        self._pooling_kernel = pooling_kernel
        self._padding = padding
        self._use_bias = use_bias

        self._weight_length = out_channels * in_channels * self._kH * self._kW
        self.set_weight(torch.randn(1, self._weight_length))
        if use_bias:
            self._bias_length = out_channels
            self.set_bias(torch.randn(1, self._bias_length))
        else:
            self._bias_length = 0
            self._bias = None

    @property
    def weight(self):
        """Get stored weight tensor or batch of weight tensors."""
        return self._weight

    @property
    def bias(self):
        """Get stored bias tensor or batch of bias tensors."""
        return self._bias

    @property
    def weight_length(self):
        """Get the n_element of a single weight tensor. """
        return self._weight_length

    @property
    def bias_length(self):
        """Get the n_element of a single bias tensor. """
        return self._bias_length

    def set_weight(self, weight):
        """Store a weight tensor or batch of weight tensors."""
        assert (weight.ndim == 2 and weight.shape[1] == self._weight_length), (
            "Input weight has wrong shape %s. Expecting shape (n, %d)" %
            (weight.shape, self._weight_length))
        if weight.shape[0] == 1:
            # non-parallel weight
            self._groups = 1
            self._weight = weight.view(self._out_channels, self._in_channels,
                                       self._kH, self._kW)
        else:
            # parallel weight
            self._groups = weight.shape[0]
            weight = weight.view(self._groups, self._out_channels,
                                 self._in_channels, self._kH, self._kW)
            self._weight = weight.reshape(self._groups * self._out_channels,
                                          self._in_channels, self._kH,
                                          self._kW)

    def set_bias(self, bias):
        """Store a bias tensor or batch of bias tensors."""
        assert (bias.ndim == 2 and bias.shape[1] == self._bias_length), (
            "Input bias has wrong shape %s. Expecting shape (n, %d)" %
            (bias.shape, self._bias_length))
        if self._groups == 1:
            # non-parallel bias
            assert bias.shape[0] == 1, (
                "Input bias has wrong shape %s. Expecting shape (%d, %d)" %
                (bias.shape, 1, self.bias_length))
        else:
            # parallel weight
            assert bias.shape[0] == self._groups, (
                "Input bias has wrong shape %s. Expecting shape (%d, %d)" %
                (bias.shape, self._group, self.bias_length))
        self._bias = bias.reshape(-1)

    def forward(self, img, keep_group_dim=True):
        """Forward

        Args:
            img (torch.Tensor): with shape ``[B, C, H, W] (groups=1)``
                                        or ``[B, n, C, H, W] (groups=n)``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``n``: number of replicas
                - ``C``: number of channels
                - ``H``: image height
                - ``W``: image width.
                When the shape of img is ``[B, C, H, W]``, all the n 2D Conv
                operations will take img as the same shared input.
                When the shape of img is ``[B, n, C, H, W]``, each 2D Conv operator
                will have its own input data by slicing img.

        Returns:
            torch.Tensor with shape ``[B, n, C', H', W']``
                where the meaning of the symbols are:
                - ``B``: batch
                - ``n``: number of replicas
                - ``C'``: number of output channels
                - ``H'``: output height
                - ``W'``: output width
        """
        if self._groups == 1:
            # non-parallel layer
            assert (img.ndim == 4 and img.shape[1] == self._in_channels), (
                "Input img has wrong shape %s. Expecting (B, %d, H, W)" %
                (img.shape, self._in_channels))
        else:
            # parallel layer
            if img.ndim == 4:
                if img.shape[1] == self._in_channels:
                    # case 1: non-parallel input
                    img = img.repeat(1, self._groups, 1, 1)
                else:
                    # case 2: parallel input
                    assert img.shape[1] == self._groups * self._in_channels, (
                        "Input img has wrong shape %s. Expecting (B, %d, H, W) or (B, %d, H, W)"
                        % (img.shape, self._in_channels,
                           self._groups * self._in_channels))
            elif img.ndim == 5:
                # case 3: parallel input with unmerged group dim
                assert (
                    img.shape[1] == self._groups
                    and img.shape[2] == self._in_channels
                ), ("Input img has wrong shape %s. Expecting (B, %d, %d, H, W)"
                    % (img.shape, self._groups, self._in_channels))
                # merge group and channel dim
                img = img.reshape(img.shape[0], img.shape[1] * img.shape[2],
                                  *img.shape[3:])
            else:
                raise ValueError("Wrong img.ndim=%d" % img.ndim)

        res = self._activation(
            F.conv2d(
                img,
                self._weight,
                bias=self._bias,
                stride=self._strides,
                padding=self._padding,
                groups=self._groups))
        if self._pooling_kernel is not None:
            res = F.max_pool2d(res, self._pooling_kernel)

        if self._groups > 1 and keep_group_dim:
            # reshape back: [B, n*C', H', W'] -> [B, n, C', H', W']
            res = res.reshape(res.shape[0], self._groups, self._out_channels,
                              res.shape[2], res.shape[3])

        return res
