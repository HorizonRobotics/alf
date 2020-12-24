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

import abc
import copy
import functools
import gin
import numpy as np

import torch
import torch.nn as nn

from .network import Network
from .preprocessor_networks import PreprocessorNetwork
import alf
import alf.layers as layers
from alf.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops


@gin.configurable
class ImageEncodingNetwork(Network):
    """
    A general template class for creating convolutional encoding networks.
    """

    def __init__(self,
                 input_channels,
                 input_size,
                 conv_layer_params,
                 same_padding=False,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 flatten_output=False,
                 name="ImageEncodingNetwork"):
        """
        Initialize the layers for encoding an image into a latent vector.
        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        How to calculate the output size:
        `<https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d>`_::

            H = (H1 - HF + 2P) // strides + 1

        where H = output size, H1 = input size, HF = size of kernel, P = padding.

        Regarding padding: in the previous TF version, we have two padding modes:
        ``valid`` and ``same``. For the former, we always have no padding (P=0); for
        the latter, it's also called "half padding" (P=(HF-1)//2 when strides=1
        and HF is an odd number the output has the same size with the input.
        Currently, PyTorch don't support different left and right paddings and
        P is always (HF-1)//2. So if HF is an even number, the output size will
        decrease by 1 when strides=1).

        Args:
            input_channels (int): number of channels in the input image
            input_size (int or tuple): the input image size (height, width)
            conv_layer_params (tuppe[tuple]): a non-empty tuple of
                tuple (num_filters, kernel_size, strides, padding), where
                padding is optional
            same_padding (bool): similar to TF's conv2d ``same`` padding mode. If
                True, the user provided paddings in `conv_layer_params` will be
                replaced by automatically calculated ones; if False, it
                corresponds to TF's ``valid`` padding mode (the user can still
                provide custom paddings though)
            activation (torch.nn.functional): activation for all the layers
            kernel_initializer (Callable): initializer for all the layers.
            flatten_output (bool): If False, the output will be an image
                structure of shape ``BxCxHxW``; otherwise the output will be
                flattened into a feature of shape ``BxN``.
        """
        input_size = common.tuplify2d(input_size)
        super().__init__(
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
            if same_padding:  # overwrite paddings
                kernel_size = common.tuplify2d(kernel_size)
                padding = ((kernel_size[0] - 1) // 2,
                           (kernel_size[1] - 1) // 2)
            self._conv_layers.append(
                layers.Conv2D(
                    input_channels,
                    filters,
                    kernel_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    strides=strides,
                    padding=padding))
            input_channels = filters

    def forward(self, inputs, state=()):
        """The empty state just keeps the interface same with other networks."""
        z = inputs
        for conv_l in self._conv_layers:
            z = conv_l(z)
        if self._flatten_output:
            z = torch.reshape(z, (z.size()[0], -1))
        return z, state


@gin.configurable
class ParallelImageEncodingNetwork(Network):
    """
    A Parallel Image Encoding Network that can be used to perform n
    independent image encodings in parallel.
    """

    def __init__(self,
                 input_channels,
                 input_size,
                 n,
                 conv_layer_params,
                 same_padding=False,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 flatten_output=False,
                 name="ParallelImageEncodingNetwork"):
        """
        Args:
            input_channels (int): number of channels in the input image
            input_size (int or tuple): the input image size (height, width)
            n (int): number of parallel networks
            conv_layer_params (tuppe[tuple]): a non-empty tuple of
                tuple (num_filters, kernel_size, strides, padding), where
                padding is optional
            same_padding (bool): similar to TF's conv2d ``same`` padding mode. If
                True, the user provided paddings in `conv_layer_params` will be
                replaced by automatically calculated ones; if False, it
                corresponds to TF's ``valid`` padding mode (the user can still
                provide custom paddings though)
            activation (torch.nn.functional): activation for all the layers
            kernel_initializer (Callable): initializer for all the layers.
            flatten_output (bool): If False, the output will be an image
                structure of shape ``(B, n, C, H, W)``; otherwise the output
                will be flattened into a feature of shape ``(B, n, C*H*W)``.
        """
        input_size = common.tuplify2d(input_size)
        super().__init__(
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
            if same_padding:  # overwrite paddings
                kernel_size = common.tuplify2d(kernel_size)
                padding = ((kernel_size[0] - 1) // 2,
                           (kernel_size[1] - 1) // 2)
            self._conv_layers.append(
                layers.ParallelConv2D(
                    input_channels,
                    filters,
                    kernel_size,
                    n,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    strides=strides,
                    padding=padding))
            input_channels = filters

    def forward(self, inputs, state=()):
        """Forward

        Args:
            inputs (torch.Tensor): with shape ``[B, C, H, W]``
                                        or ``[B, n, C, H, W]``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``n``: number of replicas
                - ``C``: number of channels
                - ``H``: image height
                - ``W``: image width.
                When the shape of inputs is ``[B, C, H, W]``, the same input is
                shared among all the n replicas.
                When the shape of img is ``[B, n, C, H, W]``, each replica
                will have its own data by slicing inputs.

            state: an empty state just keeps the interface same with other
                networks.

        Returns:
            - a tensor of shape ``(B, n, C, H, W)`` if ``flatten_output=False``
              ``(B, n, C*H*W)`` if ``flatten_output=True``
            - the empty state just to keep the interface same with other networks
        """
        z = inputs
        for conv_l in self._conv_layers:
            z = conv_l(z)
        if self._flatten_output:
            z = torch.reshape(z, (*z.size()[:2], -1))
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
                 same_padding=False,
                 preprocess_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 output_activation=torch.tanh,
                 name="ImageDecodingNetwork"):
        """
        Initialize the layers for decoding a latent vector into an image.
        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        How to calculate the output size:
        `<https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d>`_::

            H = (H1-1) * strides + HF - 2P + OP

        where H = output size, H1 = input size, HF = size of kernel, P = padding,
        OP = output_padding (currently hardcoded to be 0 for this class).

        Regarding padding: in the previous TF version, we have two padding modes:
        ``valid`` and ``same``. For the former, we always have no padding (P=0); for
        the latter, it's also called ``half padding`` (P=(HF-1)//2 when strides=1
        and HF is an odd number the output has the same size with the input.
        Currently, PyTorch doesn't support different left and right paddings and
        P is always (HF-1)//2. So if HF is an even number, the output size will
        increaseby 1 when strides=1).

        Args:
            input_size (int): the size of the input latent vector
            transconv_layer_params (tuple[tuple]): a non-empty
                tuple of tuple (num_filters, kernel_size, strides, padding),
                where ``padding`` is optional.
            start_decoding_size (int or tuple): the initial height and width
                we'd like to have for the feature map
            start_decoding_channels (int): the initial number of channels we'd
                like to have for the feature map. Note that we always first
                project an input latent vector into a vector of an appropriate
                length so that it can be reshaped into (``start_decoding_channels``,
                ``start_decoding_height``, ``start_decoding_width``).
            same_padding (bool): similar to TF's conv2d ``same`` padding mode. If
                True, the user provided paddings in ``transconv_layer_params`` will
                be replaced by automatically calculated ones; if False, it
                corresponds to TF's ``valid`` padding mode (the user can still
                provide custom paddings though).
            preprocess_fc_layer_params (tuple[int]): a tuple of fc
                layer units. These fc layers are used for preprocessing the
                latent vector before transposed convolutions.
            activation (nn.functional): activation for hidden layers
            kernel_initializer (Callable): initializer for all the layers.
            output_activation (nn.functional): activation for the output layer.
                Usually our image inputs are normalized to [0, 1] or [-1, 1],
                so this function should be ``torch.sigmoid`` or
                ``torch.tanh``.
            name (str):
        """
        super().__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(transconv_layer_params, tuple)
        assert len(transconv_layer_params) > 0

        self._preprocess_fc_layers = nn.ModuleList()
        if preprocess_fc_layer_params is not None:
            for size in preprocess_fc_layer_params:
                self._preprocess_fc_layers.append(
                    layers.FC(
                        input_size,
                        size,
                        activation=activation,
                        kernel_initializer=kernel_initializer))
                input_size = size

        start_decoding_size = common.tuplify2d(start_decoding_size)
        # pytorch assumes "channels_first" !
        self._start_decoding_shape = [
            start_decoding_channels, start_decoding_size[0],
            start_decoding_size[1]
        ]
        self._preprocess_fc_layers.append(
            layers.FC(
                input_size,
                np.prod(self._start_decoding_shape),
                activation=activation,
                kernel_initializer=kernel_initializer))

        self._transconv_layer_params = transconv_layer_params
        self._transconv_layers = nn.ModuleList()
        in_channels = start_decoding_channels
        for i, paras in enumerate(transconv_layer_params):
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            if same_padding:  # overwrite paddings
                kernel_size = common.tuplify2d(kernel_size)
                padding = ((kernel_size[0] - 1) // 2,
                           (kernel_size[1] - 1) // 2)
            act = activation
            if i == len(transconv_layer_params) - 1:
                act = output_activation
            self._transconv_layers.append(
                layers.ConvTranspose2D(
                    in_channels,
                    filters,
                    kernel_size,
                    activation=act,
                    kernel_initializer=kernel_initializer,
                    strides=strides,
                    padding=padding))
            in_channels = filters

    def forward(self, inputs, state=()):
        """Returns an image of shape ``(B,C,H,W)``. The empty state just keeps the
        interface same with other networks.
        """
        z = inputs
        for fc_l in self._preprocess_fc_layers:
            z = fc_l(z)
        z = torch.reshape(z, [-1] + self._start_decoding_shape)
        for deconv_l in self._transconv_layers:
            z = deconv_l(z)
        return z, state


@gin.configurable
class ParallelImageDecodingNetwork(Network):
    """
    A Parallel Image Decoding Network that can be used to perform n
    independent image decodings in parallel.
    """

    def __init__(self,
                 input_size,
                 n,
                 transconv_layer_params,
                 start_decoding_size,
                 start_decoding_channels,
                 same_padding=False,
                 preprocess_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 output_activation=torch.tanh,
                 name="ImageDecodingNetwork"):
        """
        Args:
            input_size (int): the size of the input latent vector
            n (int): number of parallel networks
            transconv_layer_params (tuple[tuple]): a non-empty
                tuple of tuple (num_filters, kernel_size, strides, padding),
                where ``padding`` is optional.
            start_decoding_size (int or tuple): the initial height and width
                we'd like to have for the feature map
            start_decoding_channels (int): the initial number of channels we'd
                like to have for the feature map. Note that we always first
                project an input latent vector into a vector of an appropriate
                length so that it can be reshaped into (``start_decoding_channels``,
                ``start_decoding_height``, ``start_decoding_width``).
            same_padding (bool): similar to TF's conv2d ``same`` padding mode. If
                True, the user provided paddings in ``transconv_layer_params`` will
                be replaced by automatically calculated ones; if False, it
                corresponds to TF's ``valid`` padding mode (the user can still
                provide custom paddings though).
            preprocess_fc_layer_params (tuple[int]): a tuple of fc
                layer units. These fc layers are used for preprocessing the
                latent vector before transposed convolutions.
            activation (nn.functional): activation for hidden layers
            kernel_initializer (Callable): initializer for all the layers.
            output_activation (nn.functional): activation for the output layer.
                Usually our image inputs are normalized to [0, 1] or [-1, 1],
                so this function should be ``torch.sigmoid`` or
                ``torch.tanh``.
            name (str):
        """
        super().__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(transconv_layer_params, tuple)
        assert len(transconv_layer_params) > 0

        self._preprocess_fc_layers = nn.ModuleList()
        if preprocess_fc_layer_params is not None:
            for size in preprocess_fc_layer_params:
                self._preprocess_fc_layers.append(
                    layers.ParallelFC(
                        input_size,
                        size,
                        n,
                        activation=activation,
                        kernel_initializer=kernel_initializer))
                input_size = size

        start_decoding_size = common.tuplify2d(start_decoding_size)
        # pytorch assumes "channels_first" !
        self._start_decoding_shape = [
            start_decoding_channels, start_decoding_size[0],
            start_decoding_size[1]
        ]
        self._preprocess_fc_layers.append(
            layers.ParallelFC(
                input_size,
                np.prod(self._start_decoding_shape),
                n,
                activation=activation,
                kernel_initializer=kernel_initializer))

        self._transconv_layer_params = transconv_layer_params
        self._transconv_layers = nn.ModuleList()
        in_channels = start_decoding_channels
        for i, paras in enumerate(transconv_layer_params):
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            if same_padding:  # overwrite paddings
                kernel_size = common.tuplify2d(kernel_size)
                padding = ((kernel_size[0] - 1) // 2,
                           (kernel_size[1] - 1) // 2)
            act = activation
            if i == len(transconv_layer_params) - 1:
                act = output_activation
            self._transconv_layers.append(
                layers.ParallelConvTranspose2D(
                    in_channels,
                    filters,
                    kernel_size,
                    n,
                    activation=act,
                    kernel_initializer=kernel_initializer,
                    strides=strides,
                    padding=padding))
            in_channels = filters
        self._n = n

    def forward(self, inputs, state=()):
        """Forward

        Args:
            inputs (torch.Tensor): with shape ``[B, N]``
                                        or ``[B, n, N]``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``n``: number of replicas
                - ``N``: dimension of the feature vector to be decoded.
                When the shape of inputs is ``[B, N]``, the same input is
                shared among all the n replicas.
                When the shape of img is ``[B, n, N]``, each replica
                will have its own data by slicing inputs.

            state: an empty state just keeps the interface same with other
                networks.

        Returns:
            - an image of shape ``(B, n, C, H, W)``
            - the empty state just to keep the interface same with other networks
        """
        z = inputs
        for fc_l in self._preprocess_fc_layers:
            z = fc_l(z)
        z = torch.reshape(z, [-1, self._n] + self._start_decoding_shape)
        for deconv_l in self._transconv_layers:
            z = deconv_l(z)
        return z, state


@gin.configurable
class EncodingNetwork(PreprocessorNetwork):
    """Feed Forward network with CNN and FC layers which allows the last layer
    to have different settings from the other layers.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=None,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_fc_bn=False,
                 last_layer_size=None,
                 last_activation=None,
                 last_kernel_initializer=None,
                 last_use_fc_bn=False,
                 name="EncodingNetwork"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            output_tensor_spec (None|TensorSpec): spec for the output. If None,
                the output tensor spec will be assumed as
                ``TensorSpec((output_size, ))``, where ``output_size`` is
                inferred from network output. Otherwise, the output tensor
                spec will be ``output_tensor_spec`` and the network output
                will be reshaped according to ``output_tensor_spec``.
                Note that ``output_tensor_spec`` is only used for reshaping
                the network outputs for interpretation purpose and is not used
                for specifying any network layers.
            input_preprocessors (nested InputPreprocessor): a nest of
                ``InputPreprocessor``, each of which will be applied to the
                corresponding input. If not None, then it must have the same
                structure with ``input_tensor_spec``. This arg is helpful if you
                want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                ``input_tensor_spec`` as the input to compute the processed
                tensor spec. For example, see ``alf.nest.utils.NestConcat``. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            fc_layer_params (tuple[int]): a tuple of integers
                representing FC layer sizes.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If None, a variance_scaling_initializer will be
                used.
            use_fc_bn (bool): whether use Batch Normalization for fc layers.
            last_layer_size (int): an optional size of an additional layer
                appended at the very end. Note that if ``last_activation`` is
                specified, ``last_layer_size`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_size``. Note that if
                ``last_layer_size`` is not None, ``last_activation`` has to be
                specified explicitly.
            last_use_fc_bn (bool): whether use Batch Normalization for the last
                fc layer.
            last_kernel_initializer (Callable): initializer for the the
                additional layer specified by ``last_layer_size``.
                If None, it will be the same with ``kernel_initializer``. If
                ``last_layer_size`` is None, ``last_kernel_initializer`` will
                not be used.
            name (str):
        """
        super().__init__(
            input_tensor_spec,
            input_preprocessors,
            preprocessing_combiner,
            name=name)

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

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
                activation=activation,
                kernel_initializer=kernel_initializer,
                flatten_output=True)
            input_size = self._img_encoding_net.output_spec.shape[0]
        else:
            assert self._processed_input_tensor_spec.ndim == 1, \
                "The input shape {} should be like (N,)!".format(
                    self._processed_input_tensor_spec.shape)
            input_size = self._processed_input_tensor_spec.shape[0]

        self._fc_layers = nn.ModuleList()
        if fc_layer_params is None:
            fc_layer_params = []
        else:
            assert isinstance(fc_layer_params, tuple)
            fc_layer_params = list(fc_layer_params)

        for size in fc_layer_params:
            self._fc_layers.append(
                layers.FC(
                    input_size,
                    size,
                    activation=activation,
                    use_bn=use_fc_bn,
                    kernel_initializer=kernel_initializer))
            input_size = size

        if last_layer_size is not None or last_activation is not None:
            assert last_layer_size is not None and last_activation is not None, \
            "Both last_layer_size and last_activation need to be specified!"

            if last_kernel_initializer is None:
                common.warning_once(
                    "last_kernel_initializer is not specified "
                    "for the last layer of size {}.".format(last_layer_size))
                last_kernel_initializer = kernel_initializer

            self._fc_layers.append(
                layers.FC(
                    input_size,
                    last_layer_size,
                    activation=last_activation,
                    use_bn=last_use_fc_bn,
                    kernel_initializer=last_kernel_initializer))
            input_size = last_layer_size

        if output_tensor_spec is not None:
            assert output_tensor_spec.numel == input_size, (
                "network output "
                "size {a} is inconsisent with specified out_tensor_spec "
                "of size {b}".format(a=input_size, b=output_tensor_spec.numel))
            self._output_spec = TensorSpec(
                output_tensor_spec.shape,
                dtype=self._processed_input_tensor_spec.dtype)
            self._reshape_output = True
        else:
            self._output_spec = TensorSpec(
                (input_size, ), dtype=self._processed_input_tensor_spec.dtype)
            self._reshape_output = False

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (nested Tensor):
        """
        # call super to preprocess inputs
        z, state = super().forward(inputs, state)
        if self._img_encoding_net is not None:
            z, _ = self._img_encoding_net(z)
        if alf.summary.should_summarize_output():
            name = ('summarize_output/' + self.name + '.fc.0.' + 'input_norm.'
                    + common.exe_mode_name())
            alf.summary.scalar(
                name=name, data=torch.mean(z.norm(dim=list(range(1, z.ndim)))))
        i = 0
        for fc in self._fc_layers:
            z = fc(z)
            if alf.summary.should_summarize_output():
                name = ('summarize_output/' + self.name + '.fc.' + str(i) +
                        '.output_norm.' + common.exe_mode_name())
                alf.summary.scalar(
                    name=name,
                    data=torch.mean(z.norm(dim=list(range(1, z.ndim)))))
            i += 1

        if self._reshape_output:
            z = z.reshape(z.shape[0], *self._output_spec.shape)
        return z, state

    def make_parallel(self, n):
        """Make a parallelized version of this network.

        A parallel network has ``n`` copies of network with the same structure but
        different independently initialized parameters.

        For supported network structures (currently, networks with only FC layers)
        it will create ``ParallelEncodingNetwork`` (PEN). Otherwise, it will
        create a ``NaiveParallelNetwork`` (NPN). However, PEN is not always
        faster than NPN. Especially for small ``n`` and large batch_size. See
        ``test_make_parallel()`` in critic_networks_test.py for detail.

        Returns:
            Network: A parallel network
        """
        if (self.saved_args.get('input_preprocessors') is None and
            (self._preprocessing_combiner == math_ops.identity or isinstance(
                self._preprocessing_combiner,
                (alf.nest.utils.NestSum, alf.nest.utils.NestConcat)))):
            parallel_enc_net_args = dict(**self.saved_args)
            parallel_enc_net_args.update(n=n, name="parallel_" + self.name)
            return ParallelEncodingNetwork(**parallel_enc_net_args)
        else:
            common.warning_once(
                " ``NaiveParallelNetwork`` is used by ``make_parallel()`` !")
            return super().make_parallel(n)


@gin.configurable
class ParallelEncodingNetwork(PreprocessorNetwork):
    """Parallel feed-forward network with FC layers which allows the last layer
    to have different settings from the other layers.
    """

    def __init__(self,
                 input_tensor_spec,
                 n,
                 output_tensor_spec=None,
                 input_preprocessors=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_fc_bn=False,
                 last_layer_size=None,
                 last_activation=None,
                 last_kernel_initializer=None,
                 last_use_fc_bn=False,
                 name="ParallelEncodingNetwork"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            n (int): number of parallel networks
            output_tensor_spec (None|TensorSpec): spec for the output, excluding
                the dimension of paralle networks ``n``. If None, the output
                tensor spec will be assumed as ``TensorSpec((n, output_size, ))``,
                where ``output_size`` is inferred from network output.
                Otherwise, the output tensor spec will be
                ``TensorSpec((n, *output_tensor_spec.shape))`` and
                the network output will be reshaped accordingly.
                Note that ``output_tensor_spec`` is only used for reshaping
                the network outputs for interpretation purpose and is not used
                for specifying any network layers.
            input_preprocessors (None): must be ``None``.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                ``input_tensor_spec`` as the input to compute the processed
                tensor spec. For example, see ``alf.nest.utils.NestConcat``. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            fc_layer_params (tuple[int]): a tuple of integers
                representing FC layer sizes.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer. If None, a variance_scaling_initializer will be
                used.
            use_fc_bn (bool): whether use Batch Normalization for fc layers.
            last_layer_size (int): an optional size of an additional layer
                appended at the very end. Note that if ``last_activation`` is
                specified, ``last_layer_size`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_size``. Note that if
                ``last_layer_size`` is not None, ``last_activation`` has to be
                specified explicitly.
            last_kernel_initializer (Callable): initializer for the the
                additional layer specified by ``last_layer_size``.
                If None, it will be the same with ``kernel_initializer``. If
                ``last_layer_size`` is None, ``last_kernel_initializer`` will
                not be used.
            last_use_fc_bn (bool): whether use Batch Normalization for the last
                fc layer.
            name (str):
        """
        super().__init__(
            input_tensor_spec,
            input_preprocessors=None,
            preprocessing_combiner=preprocessing_combiner,
            name=name)

        # TODO: handle input_preprocessors
        assert input_preprocessors is None

        if kernel_initializer is None:
            kernel_initializer = functools.partial(
                variance_scaling_init,
                mode='fan_in',
                distribution='truncated_normal',
                nonlinearity=activation)

        self._img_encoding_net = None
        if conv_layer_params:
            assert isinstance(conv_layer_params, tuple), \
                "The input params {} should be tuple".format(conv_layer_params)
            assert len(self._processed_input_tensor_spec.shape) == 3, \
                "The input shape {} should be like (C,H,W)!".format(
                    self._processed_input_tensor_spec.shape)
            input_channels, height, width = self._processed_input_tensor_spec.shape
            self._img_encoding_net = ParallelImageEncodingNetwork(
                input_channels, (height, width),
                n,
                conv_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
                flatten_output=True)
            input_size = self._img_encoding_net.output_spec.shape[1]
        else:
            assert self._processed_input_tensor_spec.ndim == 1, \
                "The input shape {} should be like (N,)!".format(
                    self._processed_input_tensor_spec.shape)
            input_size = self._processed_input_tensor_spec.shape[0]

        self._fc_layers = nn.ModuleList()
        if fc_layer_params is None:
            fc_layer_params = []
        else:
            assert isinstance(fc_layer_params, tuple)
            fc_layer_params = list(fc_layer_params)

        for size in fc_layer_params:
            self._fc_layers.append(
                layers.ParallelFC(
                    input_size,
                    size,
                    n,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    use_bn=use_fc_bn))
            input_size = size

        if last_layer_size is not None or last_activation is not None:
            assert last_layer_size is not None and last_activation is not None, \
            "Both last_layer_size and last_activation need to be specified!"

            if last_kernel_initializer is None:
                common.warning_once(
                    "last_kernel_initializer is not specified "
                    "for the last layer of size {}.".format(last_layer_size))
                last_kernel_initializer = kernel_initializer

            self._fc_layers.append(
                layers.ParallelFC(
                    input_size,
                    last_layer_size,
                    n,
                    activation=last_activation,
                    kernel_initializer=last_kernel_initializer,
                    use_bn=last_use_fc_bn))
            input_size = last_layer_size

        if output_tensor_spec is not None:
            assert output_tensor_spec.numel == input_size, (
                "network output "
                "size {a} is inconsisent with specified out_tensor_spec "
                "of size {b}".format(a=input_size, b=output_tensor_spec.numel))
            self._output_spec = TensorSpec(
                (n, *output_tensor_spec.shape),
                dtype=self._processed_input_tensor_spec.dtype)
            self._reshape_output = True
        else:
            self._output_spec = TensorSpec(
                (n, input_size), dtype=self._processed_input_tensor_spec.dtype)
            self._reshape_output = False

        self._n = n

    def forward(self, inputs, state=()):
        """
        Args:
            inputs (nested Tensor):
        """
        # call super to preprocess inputs
        z, state = super().forward(inputs, state, max_outer_rank=2)
        if self._img_encoding_net is None and len(self._fc_layers) == 0:
            if z.ndim == 2:
                z = z.unsqueeze(1).expand(-1, self._n, *z.shape[1:])
        else:
            if self._img_encoding_net is not None:
                z, _ = self._img_encoding_net(z)
            for fc in self._fc_layers:
                z = fc(z)
        if self._reshape_output:
            z = z.reshape(z.shape[0], *self._output_spec.shape)
        return z, state


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
                 lstm_output_layers=-1,
                 post_fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 last_layer_size=None,
                 last_activation=None,
                 last_kernel_initializer=None,
                 name="LSTMEncodingNetwork"):
        """
        Args:
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            input_preprocessors (nested InputPreprocessor): a nest of
                ``InputPreprocessor``, each of which will be applied to the
                corresponding input. If not None, then it must have the same
                structure with ``input_tensor_spec``. This arg is helpful if you
                want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector.
            preprocessing_combiner (NestCombiner): preprocessing called on
                complex inputs. Note that this combiner must also accept
                ``input_tensor_spec`` as the input to compute the processed
                tensor spec. For example, see ``alf.nest.utils.NestConcat``. This
                arg is helpful if you want to combine inputs by configuring a
                gin file without changing the code.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format ``(filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            pre_fc_layer_params (tuple[int]): a tuple of integers
                representing FC layers that are applied before the LSTM cells.
            hidden_size (int or tuple[int]): the hidden size(s) of
                the lstm cell(s). Each size corresponds to a cell. If there are
                multiple sizes, then lstm cells are stacked.
            lstm_output_layers (None|int|list[int]): -1 means the output from
                the last lstm layer. ``None`` means all lstm layers.
            post_fc_layer_params (tuple[int]): an optional tuple of
                integers representing hidden FC layers that are applied after
                the LSTM cells.
            activation (nn.functional): activation for all the layers but the
                last layer.
            kernel_initializer (Callable): initializer for all the layers but
                the last layer.
            last_layer_size (int): an optional size of an additional layer
                appended at the very end. Note that if ``last_activation`` is
                specified, ``last_layer_size`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_size``. Note that if
                ``last_layer_size`` is not None, ``last_activation`` has to be
                specified explicitly.
            last_kernel_initializer (Callable): initializer for the the
                additional layer specified by ``last_layer_size``.
                If None, it will be the same with ``kernel_initializer``. If
                ``last_layer_size`` is None, ``last_kernel_initializer`` will
                not be used.
        """
        super().__init__(input_tensor_spec, name=name)

        if (input_preprocessors or preprocessing_combiner or conv_layer_params
                or pre_fc_layer_params):
            self._pre_encoding_net = EncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                input_preprocessors=input_preprocessors,
                preprocessing_combiner=preprocessing_combiner,
                conv_layer_params=conv_layer_params,
                fc_layer_params=pre_fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer)
            input_size = self._pre_encoding_net.output_spec.shape[0]
        else:
            self._pre_encoding_net = lambda x: (x, ())
            input_size = input_tensor_spec.shape[0]

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

        if lstm_output_layers is None:
            lstm_output_layers = list(range(len(hidden_size)))
        elif type(lstm_output_layers) == int:
            lstm_output_layers = [lstm_output_layers]
        self._lstm_output_layers = lstm_output_layers
        self._lstm_output_layers = copy.copy(lstm_output_layers)
        input_size = sum(hidden_size[i] for i in lstm_output_layers)

        if post_fc_layer_params is None and last_layer_size is None:
            self._post_encoding_net = lambda x: (x, ())
            self._output_spec = TensorSpec((input_size, ))
        else:
            self._post_encoding_net = EncodingNetwork(
                input_tensor_spec=TensorSpec((input_size, )),
                fc_layer_params=post_fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
                last_layer_size=last_layer_size,
                last_activation=last_activation,
                last_kernel_initializer=last_kernel_initializer)
            self._output_spec = self._post_encoding_net.output_spec

    def _create_lstm_cell_state_spec(self, hidden_size, dtype=torch.float32):
        """Create LSTMCell state specs given the hidden size and dtype, according to
        PyTorch `LSTMCell doc <https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell>`_.

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
                of ``h_state`` and ``c_state``.

        Returns:
            tuple:
            - output (torch.Tensor): output of the network
            - new_state (list[tuple]): the updated states
        """
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

        if len(self._lstm_output_layers) == 1:
            lstm_output = new_state[self._lstm_output_layers[0]][0]
        else:
            lstm_output = [new_state[l][0] for l in self._lstm_output_layers]
            h_state = torch.cat(lstm_output, -1)

        output, _ = self._post_encoding_net(h_state)
        return output, new_state

    @property
    def state_spec(self):
        return self._state_spec
