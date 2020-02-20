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

import gin
import numpy as np

import torch
import torch.nn as nn

import alf.layers as layers

from alf.tensor_specs import TensorSpec


def _tuplify2d(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (x, x)


@gin.configurable
class ImageEncodingNetwork(nn.Module):
    """
    A general template class for creating convolutional encoding networks.
    """

    def __init__(self,
                 input_channels,
                 conv_layer_params,
                 activation=torch.relu,
                 flatten_output=False):
        """
        Initialize the layers for encoding an image into a latent vector.

        Args:
            input_channels (int): number of channels in the input image
            conv_layer_params (list[tuple]): a non-empty list of elements
                (num_filters, kernel_size, strides, padding), where padding is
                optional
            activation (torch.nn.functional): activation for all the layers
            flatten_output (bool): If False, the output will be an image
                structure of shape `BxCxHxW`; otherwise the output will be
                flattened into a feature of shape `BxN`
        """
        super(ImageEncodingNetwork, self).__init__()

        assert isinstance(conv_layer_params, list)
        assert len(conv_layer_params) > 0

        self._flatten_output = flatten_output
        self._conv_layer_params = conv_layer_params
        self._conv_layers = nn.ModuleList()
        for paras in conv_layer_params:
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            self._conv_layers.append(
                layers.Conv2D(
                    input_channels,
                    filters,
                    kernel_size,
                    activation=activation,
                    strides=strides,
                    padding=padding))
            input_channels = filters

    def output_shape(self, input_size):
        """Return the output shape given the input image size.

        How to calculate the output size:
        https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d

            H = (H1 - HF + 2P) // strides + 1

        where H = output size, H1 = input size, HF = size of kernel, P = padding

        Args:
            input_size (int or tuple): the input image size (height, width)

        Returns:
            a tuple representing the output shape
        """
        input_size = _tuplify2d(input_size)
        height, width = input_size
        for paras in self._conv_layer_params:
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            kernel_size = _tuplify2d(kernel_size)
            strides = _tuplify2d(strides)
            padding = _tuplify2d(padding)
            height = (
                height - kernel_size[0] + 2 * padding[0]) // strides[0] + 1
            width = (width - kernel_size[1] + 2 * padding[1]) // strides[1] + 1
        shape = (filters, height, width)
        if not self._flatten_output:
            return shape
        else:
            return (np.prod(shape), )

    def forward(self, inputs):
        assert len(inputs.size()) == 4, \
            "The input dims {} are incorrect! Should be (B,C,H,W)".format(
                inputs.size())
        z = inputs
        for conv_l in self._conv_layers:
            z = conv_l(z)
        if self._flatten_output:
            z = z.view(z.size()[0], -1)
        return z


@gin.configurable
class ImageDecodingNetwork(nn.Module):
    """
    A general template class for creating transposed convolutional decoding networks.
    """

    def __init__(self,
                 input_size,
                 transconv_layer_params,
                 start_decoding_size,
                 start_decoding_channels,
                 preprocess_fc_layer_params=None,
                 activation=torch.relu,
                 output_activation=torch.tanh):
        """
        Initialize the layers for decoding a latent vector into an image.

        Args:
            input_size (int): the size of the input latent vector
            transconv_layer_params (list[tuple]): a non-empty list of elements
                (num_filters, kernel_size, strides, padding), where `padding` is
                optional.
            start_decoding_size (int or tuple): the initial height and width
                we'd like to have for the feature map
            start_decoding_channels (int): the initial number of channels we'd
                like to have for the feature map. Note that we always first
                project an input latent vector into a vector of an appropriate
                length so that it can be reshaped into (`start_decoding_channels`,
                `start_decoding_height`, `start_decoding_width`).
            preprocess_fc_layer_params (tuple[int]): a list of fc layer units.
                These fc layers are used for preprocessing the latent vector before
                transposed convolutions.
            activation (nn.functional): activation for hidden layers
            output_activation (nn.functional): activation for the output layer.
                Usually our image inputs are normalized to [0, 1] or [-1, 1],
                so this function should be `torch.sigmoid` or
                `torch.tanh`.
        """
        super(ImageDecodingNetwork, self).__init__()

        assert isinstance(transconv_layer_params, list)
        assert len(transconv_layer_params) > 0

        self._preprocess_fc_layers = nn.ModuleList()
        if preprocess_fc_layer_params is not None:
            for size in preprocess_fc_layer_params:
                self._preprocess_fc_layers.append(
                    layers.FC(input_size, size, activation=activation))
                input_size = size

        start_decoding_size = _tuplify2d(start_decoding_size)
        # Python assumes "channels_first" !
        self._start_decoding_shape = [
            start_decoding_channels, start_decoding_size[0],
            start_decoding_size[1]
        ]
        self._preprocess_fc_layers.append(
            layers.FC(
                input_size,
                np.prod(self._start_decoding_shape),
                activation=activation))

        self._transconv_layer_params = transconv_layer_params
        self._transconv_layers = nn.ModuleList()
        in_channels = start_decoding_channels
        for i, paras in enumerate(transconv_layer_params):
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            act = activation
            if i == len(transconv_layer_params) - 1:
                act = output_activation
            self._transconv_layers.append(
                layers.ConvTranspose2D(
                    in_channels,
                    filters,
                    kernel_size,
                    activation=act,
                    strides=strides,
                    padding=padding))
            in_channels = filters

    def output_shape(self):
        """Return the output image shape given the start_decoding_shape.

        How to calculate the output size:
        https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d

            H = (H1-1) * strides + HF - 2P

        where H = output size, H1 = input size, HF = size of kernel, P = padding

        Returns:
            a tuple representing the output shape (C,H,W)
        """
        _, height, width = self._start_decoding_shape
        for paras in self._transconv_layer_params:
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            kernel_size = _tuplify2d(kernel_size)
            strides = _tuplify2d(strides)
            padding = _tuplify2d(padding)
            height = (
                height - 1) * strides[0] + kernel_size[0] - 2 * padding[0]
            width = (width - 1) * strides[1] + kernel_size[1] - 2 * padding[1]
        return (filters, height, width)

    def forward(self, inputs):
        """Returns an image of shape (B,C,H,W)."""
        assert len(inputs.size()) == 2, \
            "The input dims {} are incorrect! Should be (B,N)".format(
                inputs.size())
        z = inputs
        for fc_l in self._preprocess_fc_layers:
            z = fc_l(z)
        z = z.view(-1, *self._start_decoding_shape)
        for deconv_l in self._transconv_layers:
            z = deconv_l(z)
        return z


class EncodingNetwork(nn.Module):
    """Feed Forward network with CNN and FC layers."""

    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=layers.identity,
                 last_layer_size=None,
                 last_activation=None):
        """Create an EncodingNetwork

        This EncodingNetwork allows the last layer to have different settings
        from the other layers.

        Args:
            input_tensor_spec (TensorSpec): the tensor spec of the input
            conv_layer_params (list[tuple[int]]): a list of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (list[int]): a list of integers representing FC layer
                sizes.
            activation (nn.functional): activation used for hidden layers
            last_layer_size (int): an optional size of the last layer
            last_activation (nn.functional): activation function of the last
                layer. If None, it will be the same with `activation`.
        """
        super(EncodingNetwork, self).__init__()
        assert isinstance(input_tensor_spec, TensorSpec), \
            "The spec must be an instance of TensorSpec!"

        self._img_encoding_net = None
        if conv_layer_params:
            assert len(input_tensor_spec.shape) == 3, \
                "The input shape {} should be (C,H,W)!".format(
                    input_tensor_spec.shape)
            input_channels, height, width = input_tensor_spec.shape
            self._img_encoding_net = ImageEncodingNetwork(
                input_channels,
                conv_layer_params,
                activation,
                flatten_output=True)
            input_size = self._img_encoding_net.output_shape((height,
                                                              width))[0]
        else:
            assert len(input_tensor_spec.shape) == 1, \
                "The input shape {} should be (N,)!".format(
                    input_tensor_spec.shape)
            input_size = input_tensor_spec.shape[0]

        self._fc_layers = nn.ModuleList()
        if fc_layer_params is None:
            fc_layer_params = []
        else:
            assert isinstance(fc_layer_params, list)
        if last_layer_size is not None:
            fc_layer_params.append(last_layer_size)
        for i, size in enumerate(fc_layer_params):
            act = activation
            if i == len(fc_layer_params) - 1:
                act = (activation
                       if last_activation is None else last_activation)
            self._fc_layers.append(layers.FC(input_size, size, activation=act))
            input_size = size

    def forward(self, inputs):
        z = inputs
        if self._img_encoding_net is not None:
            z = self._img_encoding_net(inputs)
        for fc in self._fc_layers:
            z = fc(z)
        return z
