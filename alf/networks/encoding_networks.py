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

import abc
import gin
import numpy as np

import torch
import torch.nn as nn

import alf
import alf.layers as layers
from alf.networks.network import Network
from alf.tensor_specs import TensorSpec


def _tuplify2d(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (x, x)


@gin.configurable
class ImageEncodingNetwork(Network):
    """
    A general template class for creating convolutional encoding networks.
    """

    def __init__(self,
                 input_channels,
                 input_size,
                 conv_layer_params,
                 activation=torch.relu,
                 flatten_output=False,
                 name="ImageEncodingNetwork"):
        """
        Initialize the layers for encoding an image into a latent vector.
        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        How to calculate the output size:
        https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d

            H = (H1 - HF + 2P) // strides + 1

        where H = output size, H1 = input size, HF = size of kernel, P = padding

        Args:
            input_channels (int): number of channels in the input image
            input_size (int or tuple): the input image size (height, width)
            conv_layer_params (tuppe[tuple]): a non-empty tuple of
                tuple (num_filters, kernel_size, strides, padding), where
                padding is optional
            activation (torch.nn.functional): activation for all the layers
            flatten_output (bool): If False, the output will be an image
                structure of shape `BxCxHxW`; otherwise the output will be
                flattened into a feature of shape `BxN`
        """
        input_size = _tuplify2d(input_size)
        super(ImageEncodingNetwork, self).__init__(
            input_tensor_spec=TensorSpec((input_channels, ) + input_size),
            name=name)

        assert isinstance(conv_layer_params, tuple)
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

    def forward(self, inputs, state=()):
        """The empty state just keeps the interface same with other networks."""
        # call super to handle nested inputs
        z, state = super().forward(inputs, state)
        for conv_l in self._conv_layers:
            z = conv_l(z)
        if self._flatten_output:
            z = torch.reshape(z, (z.size()[0], -1))
        return z, state


@gin.configurable
class ImageDecodingNetwork(Network):
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
                 output_activation=torch.tanh,
                 name="ImageDecodingNetwork"):
        """
        Initialize the layers for decoding a latent vector into an image.
        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        How to calculate the output size:
        https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d

            H = (H1-1) * strides + HF - 2P

        where H = output size, H1 = input size, HF = size of kernel, P = padding

        Args:
            input_size (int): the size of the input latent vector
            transconv_layer_params (tuple[tuple]): a non-empty
                tuple of tuple (num_filters, kernel_size, strides, padding),
                where `padding` is optional.
            start_decoding_size (int or tuple): the initial height and width
                we'd like to have for the feature map
            start_decoding_channels (int): the initial number of channels we'd
                like to have for the feature map. Note that we always first
                project an input latent vector into a vector of an appropriate
                length so that it can be reshaped into (`start_decoding_channels`,
                `start_decoding_height`, `start_decoding_width`).
            preprocess_fc_layer_params (tuple[int]): a tuple of fc
                layer units. These fc layers are used for preprocessing the
                latent vector before transposed convolutions.
            activation (nn.functional): activation for hidden layers
            output_activation (nn.functional): activation for the output layer.
                Usually our image inputs are normalized to [0, 1] or [-1, 1],
                so this function should be `torch.sigmoid` or
                `torch.tanh`.
            name (str):
        """
        super(ImageDecodingNetwork, self).__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(transconv_layer_params, tuple)
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

    def forward(self, inputs, state=()):
        """Returns an image of shape (B,C,H,W). The empty state just keeps the
        interface same with other networks.
        """
        # call super to handle nested inputs
        z, state = super().forward(inputs, state)
        for fc_l in self._preprocess_fc_layers:
            z = fc_l(z)
        z = torch.reshape(z, [-1] + self._start_decoding_shape)
        for deconv_l in self._transconv_layers:
            z = deconv_l(z)
        return z, state


@gin.configurable
class EncodingNetwork(Network):
    """Feed Forward network with CNN and FC layers."""

    def __init__(self,
                 input_tensor_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu,
                 last_layer_size=None,
                 last_activation=None,
                 name="EncodingNetwork"):
        """Create an EncodingNetwork

        This EncodingNetwork allows the last layer to have different settings
        from the other layers.

        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then `preprocessing_combiner` must not be
                None.
            input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding input. If not None, then it must have the same
                structure with `input_tensor_spec`. This arg is helpful if you
                want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            fc_layer_params (tuple[int]): a tuple of integers
                representing FC layer sizes.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            last_layer_size (int): an optional size of the last layer
            last_activation (nn.functional): activation function of the last
                layer. If None, it will be the SAME with `activation`.
            name (str):
        """
        super(EncodingNetwork, self).__init__(
            input_tensor_spec,
            input_preprocessors,
            preprocessing_combiner,
            name=name)

        self._img_encoding_net = None
        if conv_layer_params:
            assert isinstance(conv_layer_params, tuple), \
                "The input params {} should be tuple".format(conv_layer_params)
            assert len(self._processed_input_tensor_spec.shape) == 3, \
                "The input shape {} should be like (C,H,W)!".format(
                    self._processed_input_tensor_spec.shape)
            input_channels, height, width = self._processed_input_tensor_spec.shape
            self._img_encoding_net = ImageEncodingNetwork(
                input_channels, (height, width),
                conv_layer_params,
                activation,
                flatten_output=True)
            input_size = self._img_encoding_net.output_spec.shape[0]
        else:
            assert len(self._processed_input_tensor_spec.shape) == 1, \
                "The input shape {} should be like (N,)!".format(
                    self._processed_input_tensor_spec.shape)
            input_size = self._processed_input_tensor_spec.shape[0]

        self._fc_layers = nn.ModuleList()
        if fc_layer_params is None:
            fc_layer_params = []
        else:
            assert isinstance(fc_layer_params, tuple)
            fc_layer_params = list(fc_layer_params)

        if last_layer_size is not None:
            fc_layer_params.append(last_layer_size)
        for i, size in enumerate(fc_layer_params):
            act = activation
            if i == len(fc_layer_params) - 1:
                act = (activation
                       if last_activation is None else last_activation)
            self._fc_layers.append(layers.FC(input_size, size, activation=act))
            input_size = size

        self._output_size = input_size

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (nested Tensor):
        """
        # call super to preprocess inputs
        z, state = super().forward(inputs, state)
        if self._img_encoding_net is not None:
            z, _ = self._img_encoding_net(z)
        for fc in self._fc_layers:
            z = fc(z)
        return z, state

    @property
    def output_spec(self):
        if self._output_spec is None:
            self._output_spec = TensorSpec(
                (self._output_size, ),
                dtype=self._processed_input_tensor_spec.dtype)
        return self._output_spec


@gin.configurable
class LSTMEncodingNetwork(Network):
    """LSTM cells followed by an encoding network."""

    def __init__(self,
                 input_tensor_spec,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 pre_fc_layer_params=None,
                 hidden_size=(100, ),
                 post_fc_layer_params=None,
                 activation=torch.relu,
                 last_layer_size=None,
                 last_activation=None,
                 name="LSTMEncodingNetwork"):
        """Creates an instance of `LSTMEncodingNetwork`.

        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then `preprocessing_combiner` must not be
                None.
            input_preprocessors (nested InputPreprocessor): a nest of
                `InputPreprocessor`, each of which will be applied to the
                corresponding input. If not None, then it must have the same
                structure with `input_tensor_spec`. This arg is helpful if you
                want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                `input_tensor_spec` as the input to compute the processed
                tensor spec. For example, see `alf.nest.utils.NestConcat`. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format `(filters, kernel_size, strides, padding)`,
                where `padding` is optional.
            pre_fc_layer_params (tuple[int]): a tuple of integers
                representing FC layers that are applied before the LSTM cells.
            hidden_size (int or tuple[int]): the hidden size(s) of
                the lstm cell(s). Each size corresponds to a cell. If there are
                multiple sizes, then lstm cells are stacked.
            post_fc_layer_params (tuple[int]): an optional tuple of
                integers representing hidden FC layers that are applied after
                the LSTM cells.
            activation (nn.functional): activation for all the layers but the
                last layer.
            last_layer_size (int): an optional size of the last layer
            last_activation (nn.functional): activation function of the last
                layer. If None, it will be the same with `activation`.
        """
        super(LSTMEncodingNetwork, self).__init__(
            input_tensor_spec,
            input_preprocessors,
            preprocessing_combiner,
            name=name)

        self._pre_encoding_net = EncodingNetwork(
            input_tensor_spec=self._processed_input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=pre_fc_layer_params,
            activation=activation)
        input_size = self._pre_encoding_net.output_spec.shape[0]

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        else:
            assert isinstance(hidden_size, tuple)

        self._cells = nn.ModuleList()
        self._state_spec = []
        for hs in hidden_size:
            self._cells.append(
                torch.nn.LSTMCell(input_size=input_size, hidden_size=hs))
            self._state_spec.append(self._create_lstm_cell_state_spec(hs))
            input_size = hs

        self._post_encoding_net = EncodingNetwork(
            input_tensor_spec=TensorSpec((input_size, )),
            fc_layer_params=post_fc_layer_params,
            activation=activation,
            last_layer_size=last_layer_size,
            last_activation=last_activation)

    def _create_lstm_cell_state_spec(self, hidden_size, dtype=torch.float32):
        """Create LSTMCell state specs given the hidden size and dtype. According to
        PyTorch LSTMCell doc:

        https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell

        Each LSTMCell has two states: h and c with the same shape.

        Args:
            hidden_size (int): the number of units in the hidden state
            dtype (torch.dtype): dtype of the specs

        Returns:
            specs (tuple[TensorSpec]):
        """
        state_spec = TensorSpec(shape=(hidden_size, ), dtype=dtype)
        return (state_spec, state_spec)

    def forward(self, inputs, state):
        """
        Args:
            inputs (nested torch.Tensor):
            state (list[tuple]): a list of tuples, where each tuple is a pair
                of `h_state` and `c_state`.

        Returns:
            output (torch.Tensor): output of the network
            new_state (list[tuple]): the updated states
        """
        # call super to preprocess inputs
        inputs, state = super().forward(inputs, state)

        assert isinstance(state, list)
        for s in state:
            assert isinstance(s, tuple) and len(s) == 2, \
                "Each LSTMCell state should be a tuple of (h,c)!"
        assert len(self._cells) == len(state)

        new_state = []
        h_state, _ = self._pre_encoding_net(inputs)
        for cell, s in zip(self._cells, state):
            h_state, c_state = cell(h_state, s)
            new_state.append((h_state, c_state))
        output, _ = self._post_encoding_net(h_state)
        return output, new_state

    @property
    def state_spec(self):
        return self._state_spec

    @property
    def output_spec(self):
        return self._post_encoding_net.output_spec
