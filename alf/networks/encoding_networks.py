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

import functools
import numpy as np
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from .containers import Sequential, _Sequential, Parallel
from .network import Network, NetworkWrapper
import alf
import alf.layers as layers
from alf.initializers import variance_scaling_init
from alf.tensor_specs import TensorSpec
from alf.utils import common
from alf.nest.utils import get_outer_rank


@alf.configurable
class ImageEncodingNetwork(_Sequential):
    """
    A general template class for creating convolutional encoding networks.
    """

    def __init__(self,
                 input_channels,
                 input_size,
                 conv_layer_params,
                 use_batch_ensemble=False,
                 ensemble_size=10,
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
        `<https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_::

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
            use_batch_ensemble (bool): whether to use Conv2DBatchEnsemble layers.
                If True, Conv2DBatchEnsemble layers will always be created with
                ``output_ensemble_ids=True``, and as a result, the output of the 
                network is a tuple with ensemble_ids; input to the network can be 
                a Tensor or a tuple, for the tuple case, it should contain two 
                tensors, the first one is the input data tensor, the second one
                is the ensemble_ids. 
            ensemble_size (int): ensemble size, only effective if use_batch_ensemble
                is True.
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
        input_tensor_spec = TensorSpec((input_channels, ) + input_size)

        assert isinstance(conv_layer_params, tuple)
        assert len(conv_layer_params) > 0

        if use_batch_ensemble:
            assert ensemble_size > 1
            conv_layer_ctor = functools.partial(
                layers.Conv2DBatchEnsemble,
                ensemble_size=ensemble_size,
                output_ensemble_ids=True)
        else:
            conv_layer_ctor = layers.Conv2D

        nets = []
        for paras in conv_layer_params:
            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            if same_padding:  # overwrite paddings
                kernel_size = common.tuplify2d(kernel_size)
                padding = ((kernel_size[0] - 1) // 2,
                           (kernel_size[1] - 1) // 2)
            nets.append(
                conv_layer_ctor(
                    input_channels,
                    filters,
                    kernel_size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    strides=strides,
                    padding=padding))
            input_channels = filters
        if flatten_output:
            if use_batch_ensemble:
                nets.append(
                    Parallel((alf.layers.Reshape(
                        (-1, )), alf.layers.Identity()),
                             ((), TensorSpec((), dtype=torch.int64))))
            else:
                nets.append(alf.layers.Reshape((-1, )))

        super().__init__(nets, input_tensor_spec=input_tensor_spec, name=name)


@alf.configurable
class ImageDecodingNetwork(_Sequential):
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
        `<https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html>`_::

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
        input_tensor_spec = TensorSpec((input_size, ))

        assert isinstance(transconv_layer_params, tuple)
        assert len(transconv_layer_params) > 0

        nets = []
        if preprocess_fc_layer_params is not None:
            for size in preprocess_fc_layer_params:
                nets.append(
                    layers.FC(
                        input_size,
                        size,
                        activation=activation,
                        kernel_initializer=kernel_initializer))
                input_size = size

        start_decoding_size = common.tuplify2d(start_decoding_size)
        # pytorch assumes "channels_first" !
        start_decoding_shape = [
            start_decoding_channels, start_decoding_size[0],
            start_decoding_size[1]
        ]
        nets.append(
            layers.FC(
                input_size,
                np.prod(start_decoding_shape),
                activation=activation,
                kernel_initializer=kernel_initializer))

        nets.append(alf.layers.Reshape(start_decoding_shape))

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
            nets.append(
                layers.ConvTranspose2D(
                    in_channels,
                    filters,
                    kernel_size,
                    activation=act,
                    kernel_initializer=kernel_initializer,
                    strides=strides,
                    padding=padding))
            in_channels = filters

        super().__init__(nets, input_tensor_spec=input_tensor_spec, name=name)


@alf.configurable
class ImageDecodingNetworkV2(_Sequential):
    """Image decoding using upsampling+convolution.

    Different with ``ImageDecodingNetwork`` which uses transposed convolution to
    transform a smaller input to a larger image output, this class uses upsampling
    followed by convolution layers. The idea is to let conv layer refine the
    upsampling (e.g., nearest neighbor, bilinear, etc) results.

    The difference between transposed conv and upsampling+conv can be found in
    this article: `<https://distill.pub/2016/deconv-checkerboard/>`_. In short,
    upsampling+conv might help reduce checkerboard artifacts that are common in
    the outputs by transposed convolutions.
    """

    def __init__(self,
                 input_size: int,
                 upsample_conv_layer_params: Tuple[Union[int, Tuple[int]]],
                 start_decoding_size: Union[int, Tuple[int]],
                 start_decoding_channels: int,
                 preprocess_fc_layer_params: Tuple[int] = None,
                 upsampling_mode: str = 'nearest',
                 same_padding: bool = False,
                 activation: Callable = torch.relu_,
                 kernel_initializer: Callable = None,
                 output_activation: Callable = torch.tanh,
                 name: str = "ImageDecodingNetworkV2"):
        """An example network of upsampling+conv for decoding images.

        .. code-block:: python

            net = ImageDecodingNetworkV2(input_size=100,
                                         start_decoding_size=10,
                                         start_decoding_channels=8,
                                         same_padding=True,
                                         upsample_conv_layer_params=(
                                            2,
                                            (16, 3, 1),
                                            (32, 3, 1),
                                            2,
                                            (64, 3, 1),
                                            (3, 3, 1)))
            # The image shape: (8,10,10) -> (8,20,20) -> (16,20,20) -> (32,20,20)
            #                  -> (32,40,40) -> (64,40,40) -> (3,40,40)

        Args:
            input_size: the size of the input latent vector
            upsample_conv_layer_params: a tuple of ints or tuples. If the element
                is an int, it represents the scaling factor for a ``torch.nn.Upsample``
                layer; otherwise it should a tuple of ints representing conv params
                ``(num_filters, kernel_size, strides, padding)``,
                where ``padding`` is optional.
            start_decoding_size: the initial height and width we'd like to have
                for the feature map.
            start_decoding_channels: the initial number of channels we'd
                like to have for the feature map. Note that we always first
                project an input latent vector into a vector of an appropriate
                length so that it can be reshaped into (``start_decoding_channels``,
                ``start_decoding_height``, ``start_decoding_width``).
            preprocess_fc_layer_params: if not None, then the input will be fed
                to a list of fc layers specified by this argument, before doing
                deconvolution.
            upsampling_mode: the argument for choosing an upsampling algorithm
                for ``torch.nn.Upsample``.
            same_padding: similar to TF's conv2d ``same`` padding mode. If
                True, the user provided paddings in ``transconv_layer_params`` will
                be replaced by automatically calculated ones; if False, it
                corresponds to TF's ``valid`` padding mode (the user can still
                provide custom paddings though). Please refer to the docstring of
                ``ImageEncodingNetwork`` for definitions of the two padding modes.
            activation: activation for hidden layers
            kernel_initializer: initializer for all the layers.
            output_activation: activation for the output layer.
                Usually our image inputs are normalized to [0, 1] or [-1, 1],
                so this function should be ``torch.sigmoid`` or ``torch.tanh``.
            name (str):
        """
        input_tensor_spec = TensorSpec((input_size, ))

        assert isinstance(upsample_conv_layer_params, tuple)
        assert len(upsample_conv_layer_params) > 0

        start_decoding_size = common.tuplify2d(start_decoding_size)
        # pytorch assumes "channels_first" !
        start_decoding_shape = [
            start_decoding_channels, start_decoding_size[0],
            start_decoding_size[1]
        ]

        nets = []
        if preprocess_fc_layer_params is not None:
            for size in preprocess_fc_layer_params:
                nets.append(
                    layers.FC(
                        input_size,
                        size,
                        activation=activation,
                        kernel_initializer=kernel_initializer))
                input_size = size

        nets.extend([
            layers.FC(
                input_size,
                np.prod(start_decoding_shape),
                activation=activation,
                kernel_initializer=kernel_initializer),
            alf.layers.Reshape(start_decoding_shape)
        ])

        in_channels = start_decoding_channels
        for i, paras in enumerate(upsample_conv_layer_params):
            if isinstance(paras, int):
                nets.append(
                    torch.nn.Upsample(
                        scale_factor=paras, mode=upsampling_mode))
            else:
                filters, kernel_size, strides = paras[:3]
                padding = paras[3] if len(paras) > 3 else 0
                if same_padding:  # overwrite paddings
                    kernel_size = common.tuplify2d(kernel_size)
                    padding = ((kernel_size[0] - 1) // 2,
                               (kernel_size[1] - 1) // 2)
                act = activation
                if i == len(upsample_conv_layer_params) - 1:
                    act = output_activation
                nets.append(
                    layers.Conv2D(
                        in_channels,
                        filters,
                        kernel_size,
                        activation=act,
                        kernel_initializer=kernel_initializer,
                        strides=strides,
                        padding=padding))
                in_channels = filters

        super().__init__(nets, input_tensor_spec=input_tensor_spec, name=name)


def SpatialBroadcastDecodingNetwork(
        input_size: int,
        output_height: int,
        conv_layer_params: Tuple[Tuple[int]],
        output_width: int = None,
        fc_layer_params: Tuple[int] = None,
        activation: Callable = torch.relu_,
        output_activation: Callable = alf.utils.math_ops.identity,
        name: str = "SpatialBroadcastDecodingNetwork"):
    """Implements the spatial broadcast decoder in

    `Watters et al. 2019,
    Spatial Broadcast Decoder: A Simple Architecture for Learning Disentangled
    Representations in VAEs <https://arxiv.org/abs/1901.07017>`_.

    In short, given a latent embedding and target output height/width, this
    decoder first spatially broadcast the embedding over ``height*width``, append
    a uniform ``xy`` meshgrid in [-1,1], and apply conv layers.

    Args:
        input_size: the latent embedding size
        output_height: the target output image height
        conv_layer_params: a tuple of conv layer params after broadcasting
        output_width: if None, it's equal to ``output_height``
        fc_layer_params: a tuple of fc layers applied to the input embedding before
            broadcasting
        activation: activation of the intermediate conv layers
        output_activation: the final activation
    """

    input_tensor_spec = TensorSpec((input_size, ))
    proj = alf.math.identity
    if fc_layer_params is not None:
        proj = EncodingNetwork(
            input_tensor_spec=input_tensor_spec,
            fc_layer_params=fc_layer_params,
            activation=activation)

    if output_width is None:
        output_width = output_height

    preproc_net = alf.nn.Sequential(
        proj,
        functools.partial(
            alf.utils.tensor_utils.spatial_broadcast,
            im_shape=(output_height, output_width)),
        alf.utils.tensor_utils.append_coordinate,
        input_tensor_spec=input_tensor_spec)

    assert isinstance(conv_layer_params, tuple) and len(conv_layer_params) > 0
    conv_net = ImageEncodingNetwork(
        input_channels=preproc_net.output_spec.shape[0],
        input_size=preproc_net.output_spec.shape[1:],
        conv_layer_params=conv_layer_params[:-1],
        same_padding=True,
        activation=activation)

    last_conv_net = ImageEncodingNetwork(
        input_channels=conv_net.output_spec.shape[0],
        input_size=conv_net.output_spec.shape[1:],
        conv_layer_params=conv_layer_params[-1:],
        same_padding=True,
        activation=output_activation)

    return alf.nn.Sequential(preproc_net, conv_net, last_conv_net, name=name)


@alf.configurable
class AutoShapeImageDeconvNetwork(_Sequential):
    """
    A general template class for creating image deconv (transposed convolutional)
        networks with auto-shape inference (thus named as
        ``AutoShapeImageDeconvNetwork``).
    """

    def __init__(self,
                 input_size: int,
                 transconv_layer_params: Tuple,
                 output_shape: Tuple,
                 start_decoding_channels: int,
                 preprocess_fc_layer_params: Optional[Tuple] = None,
                 activation: Optional[Callable] = torch.relu_,
                 kernel_initializer: Optional[Callable] = None,
                 output_activation: Optional[Callable] = torch.tanh,
                 name="AutoShapeImageDeconvNetwork"):
        """
        Auto-shape inference: instead of specifying an initial start shape for
        image deconv, this class only needs to specify the desired output shape
        for the image and will automatically calculate the desired shape to start
        decoding based on the specified ``transconv_layer_params``
        and uses a FC layer to map the to the desired start shape.

        Args:
            input_size (int): the size of the input latent vector
            transconv_layer_params (tuple[tuple]): a non-empty
                tuple of tuple (num_filters, kernel_size, strides, padding),
                where ``padding`` is optional.
            output_shape (tuple): the complete output size would be
                output_shape = (c, h, w).
            start_decoding_channels (int): the initial number of channels we'd
                like to have for the feature map. Note that we always first
                project an input latent vector into a vector of an appropriate
                length so that it can be reshaped into (``start_decoding_channels``,
                ``start_decoding_height``, ``start_decoding_width``),
                where ``start_decoding_height`` and ``start_decoding_width``
                are automatically inferred based on the specified ``output_shape``
                and ``transconv_layer_params``.
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
        assert len(output_shape) == 3, "the output_shape should be (c, h, w)"
        assert output_shape[0] == transconv_layer_params[-1][0], (
            "channel number mismatch")

        # compute conv shape and padding shape
        out_paddings = []
        out_shape = output_shape[1:]
        for i, paras in enumerate(transconv_layer_params[::-1]):
            filters, kernel_size, stride = paras[:3]
            kernel_size = common.tuplify2d(kernel_size)

            padding = paras[3] if len(paras) > 3 else 0
            padding = common.tuplify2d(padding)
            conv_shape = self._calc_conv_out_shape(out_shape, padding,
                                                   kernel_size, stride)
            out_padding = self._calc_output_padding_shape(
                out_shape, conv_shape, padding, kernel_size, stride)
            out_shape = conv_shape
            out_paddings.append(out_padding)

        input_tensor_spec = TensorSpec((input_size, ))

        assert isinstance(transconv_layer_params, tuple)
        assert len(transconv_layer_params) > 0

        nets = []
        if preprocess_fc_layer_params is not None:
            for size in preprocess_fc_layer_params:
                nets.append(
                    layers.FC(
                        input_size,
                        size,
                        activation=activation,
                        kernel_initializer=kernel_initializer))
                input_size = size

        start_decoding_shape = [
            start_decoding_channels, conv_shape[0], conv_shape[1]
        ]
        nets.append(
            layers.FC(
                input_size,
                np.prod(start_decoding_shape),
                activation=activation,
                kernel_initializer=kernel_initializer))

        nets.append(alf.layers.Reshape(start_decoding_shape))

        in_channels = start_decoding_channels

        for i, paras in enumerate(transconv_layer_params):

            filters, kernel_size, strides = paras[:3]
            padding = paras[3] if len(paras) > 3 else 0
            output_padding = out_paddings[-(i + 1)]

            act = activation
            if i == len(transconv_layer_params) - 1:
                act = output_activation

            nets.append(
                layers.ConvTranspose2D(
                    in_channels,
                    filters,
                    kernel_size,
                    activation=act,
                    kernel_initializer=kernel_initializer,
                    strides=strides,
                    padding=padding,
                    output_padding=output_padding))
            in_channels = filters

        super().__init__(nets, input_tensor_spec=input_tensor_spec, name=name)

    def _calc_conv_out_shape(self, input_size, padding, kernel_size, stride):
        """Calculate the output shape of a conv2d operation.
        Reference:
        `<https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.
        """

        def _conv_out_1d(input_size, padding, kernel_size, stride):
            return int((input_size + 2. * padding - kernel_size) / stride + 1.)

        return tuple(
            _conv_out_1d(x, p, k, stride)
            for x, p, k in zip(input_size, padding, kernel_size))

    def _calc_output_padding_shape(self, input_size, conv_out, padding,
                                   kernel_size, stride):
        """Calculate the necessary output padding to be used for
        ``ConvTranspose2D`` to ensure the image obatained from it will have a
        size that matches the ``input size``.
        """

        def _output_padding_1d(input_size, conv_out, padding, kernel_size,
                               stride):
            return input_size - (
                conv_out - 1) * stride + 2 * padding - kernel_size

        return tuple(_output_padding_1d(x, c, p, k, stride) for x, c, p, k in \
                        zip(input_size, conv_out, padding, kernel_size))


@alf.configurable
class EncodingNetwork(_Sequential):
    """Feed Forward network with CNN and FC layers which allows the last layer
    to have different settings from the other layers.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=None,
                 input_preprocessors=None,
                 input_preprocessors_ctor=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 kernel_initializer=None,
                 use_fc_bn=False,
                 use_fc_ln=False,
                 use_batch_ensemble=False,
                 ensemble_size=10,
                 input_with_ensemble_ids=False,
                 last_layer_size=None,
                 last_activation=None,
                 last_kernel_initializer=None,
                 last_use_fc_bn=False,
                 last_use_fc_ln=False,
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
            input_preprocessors (nested Network|nn.Module|None): a nest of
                preprocessors, each of which will be applied to the
                corresponding input. If not None, then it must have the same
                structure with ``input_tensor_spec``. This arg is helpful if you
                want to have separate preprocessings for different inputs by
                configuring a gin file without changing the code. For example,
                embedding a discrete input before concatenating it to another
                continuous vector.
            input_preprocessors_ctor (Callable): if ``input_preprocessors`` is None
                and ``input_preprocessors_ctor`` is provided, then ``input_preprocessors``
                will be constructed by calling ``input_preprocessors_ctor(input_tensor_spec)``.
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
            use_fc_ln (bool): whether use Layer Normalization for fc layers.
            use_batch_ensemble (bool): whether to use BatchEnsemble FC and Conv2D
                layers. If True, both BatchEnsemble layers will always be created
                with ``output_ensemble_ids=True``, and as a result, the output of
                the network is a tuple with ensemble_ids. 
            ensemble_size (int): ensemble size, only effective if use_batch_ensemble
                is True.
            input_with_ensemble_ids (bool): whether handle inputs with ensemble_ids,
                if True, input to the network should be a tuple of two tensors, the
                first one is the input data tensor and the second one is the 
                ensemble_ids. This option is only effective if use_batch_ensemble 
                is True. 
            last_layer_size (int): an optional size of an additional layer
                appended at the very end. Note that if ``last_activation`` is
                specified, ``last_layer_size`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_size``. Note that if
                ``last_layer_size`` is not None, ``last_activation`` has to be
                specified explicitly.
            last_use_fc_bn (bool): whether use Batch Normalization for the last
                fc layer.
            last_use_fc_ln (bool): whether use Layer Normalization for the last
                fc layer.
            last_kernel_initializer (Callable): initializer for the the
                additional layer specified by ``last_layer_size``.
                If None, it will be the same with ``kernel_initializer``. If
                ``last_layer_size`` is None, ``last_kernel_initializer`` will
                not be used.
            name (str):
        """
        spec = input_tensor_spec
        nets = []

        if not input_preprocessors and input_preprocessors_ctor:
            input_preprocessors = input_preprocessors_ctor(input_tensor_spec)
        if input_preprocessors:
            input_preprocessors = alf.nest.map_structure(
                lambda p: alf.layers.Identity() if p is None else p,
                input_preprocessors)
            net = alf.nn.Parallel(input_preprocessors, input_tensor_spec)
            spec = net.output_spec
            nets.append(net)

        if alf.nest.is_nested(spec):
            assert preprocessing_combiner is not None, \
                ("When a nested input tensor spec is provided, an input " +
                "preprocessing combiner must also be provided!")
            spec = preprocessing_combiner(spec)
            nets.append(preprocessing_combiner)
        else:
            assert isinstance(spec, TensorSpec), \
                "The spec must be an instance of TensorSpec!"

        if input_with_ensemble_ids:
            nets = [
                Parallel(
                    (Sequential(*nets, input_tensor_spec=input_tensor_spec),
                     alf.layers.Identity()),
                    (input_tensor_spec, TensorSpec((), dtype=torch.int64)))
            ]

        if conv_layer_params:
            assert isinstance(conv_layer_params, tuple), \
                "The input params {} should be tuple".format(conv_layer_params)
            assert len(spec.shape) == 3, \
                "The input shape {} should be like (C,H,W)!".format(spec.shape)
            input_channels, height, width = spec.shape
            net = ImageEncodingNetwork(
                input_channels, (height, width),
                conv_layer_params,
                use_batch_ensemble=use_batch_ensemble,
                ensemble_size=ensemble_size,
                activation=activation,
                kernel_initializer=kernel_initializer,
                flatten_output=True)
            spec = net.output_spec
            if use_batch_ensemble:
                spec = spec[0]
            nets.append(net)

        if spec.ndim == 1:
            # for general input_preprocessors and ImageEncodingNetwork,
            # ndim of output_spec should be 1
            input_size = spec.shape[0]
        elif spec.ndim == 2:
            # when CosineEmbeddingPreprocessor is used in input_preprocessors,
            # its output_spec is 2
            input_size = spec.shape[-1]
        else:
            raise ValueError(
                f"The input shape {spec.shape} should be like (N, )"
                "or (N, D, ).")

        if fc_layer_params is None:
            fc_layer_params = []
        else:
            assert isinstance(fc_layer_params, tuple)
            fc_layer_params = list(fc_layer_params)

        if use_batch_ensemble:
            assert ensemble_size > 1
            fc_layer_ctor = functools.partial(
                layers.FCBatchEnsemble,
                ensemble_size=ensemble_size,
                output_ensemble_ids=True)
        else:
            fc_layer_ctor = layers.FC

        for size in fc_layer_params:
            nets.append(
                fc_layer_ctor(
                    input_size,
                    size,
                    activation=activation,
                    use_bn=use_fc_bn,
                    use_ln=use_fc_ln,
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

            nets.append(
                fc_layer_ctor(
                    input_size,
                    last_layer_size,
                    activation=last_activation,
                    use_bn=last_use_fc_bn,
                    use_ln=last_use_fc_ln,
                    kernel_initializer=last_kernel_initializer))
            input_size = last_layer_size

        if output_tensor_spec is not None:
            if spec.numel == 1:
                assert output_tensor_spec.numel == input_size, (
                    "network output "
                    "size {a} is inconsistent with specified out_tensor_spec "
                    "of size {b}".format(
                        a=input_size, b=output_tensor_spec.numel))
            elif spec.numel == 2:
                assert output_tensor_spec.numel % input_size == 0
            if use_batch_ensemble:
                nets.append(
                    Parallel(
                        (alf.layers.Reshape(output_tensor_spec.shape),
                         alf.layers.Identity()),
                        (output_tensor_spec, TensorSpec(
                            (), dtype=torch.int64))))
            else:
                nets.append(alf.layers.Reshape(output_tensor_spec.shape))

        super().__init__(nets, input_tensor_spec=input_tensor_spec, name=name)

    def make_parallel(self, n: int, allow_non_parallel_input=False):
        """Make a parallelized version of ``module``.

        A parallel network has ``n`` copies of network with the same structure but
        different independently initialized parameters. The parallel network can
        process a batch of the data with shape [batch_size, n, ...] using ``n``
        networks with same structure.

        TODO: remove ``allow_non_parallel_input``. This means to make parallel network
        not to accept non-parallel input. It will make the logic more transparent.

        Args:
            n (int): the number of copies
            allow_non_parallel_input (bool): if True, the returned network will
                also accept non-parallel input with shape [batch_size, ...]. In
                this case, the network will check whether the input is parallel
                input. If not, the input will be automatically replicated ``n``
                times at the beginning.
        Returns:
            the parallelized network.
        """
        pnet = super().make_parallel(n)
        if allow_non_parallel_input:
            return _ReplicateInputForParallel(
                self.input_tensor_spec, n, pnet, name=pnet.name)
        else:
            return pnet


class _ReplicateInputForParallel(Network):
    def __init__(self, input_tensor_spec, n, pnet, name):
        super().__init__(
            input_tensor_spec, state_spec=pnet.state_spec, name=name)
        self._input_tensor_spec = input_tensor_spec
        self._n = n
        self._pnet = pnet

    def forward(self, inputs, state=()):
        outer_rank = get_outer_rank(inputs, self._input_tensor_spec)
        if outer_rank == 1:
            inputs = alf.layers.make_parallel_input(inputs, self._n)
        return self._pnet(inputs, state)


@alf.configurable
def ParallelEncodingNetwork(input_tensor_spec,
                            n,
                            output_tensor_spec=None,
                            input_preprocessors=None,
                            preprocessing_combiner=None,
                            conv_layer_params=None,
                            fc_layer_params=None,
                            activation=torch.relu_,
                            kernel_initializer=None,
                            use_fc_bn=False,
                            use_fc_ln=False,
                            last_layer_size=None,
                            last_activation=None,
                            last_kernel_initializer=None,
                            last_use_fc_bn=False,
                            last_use_fc_ln=False,
                            name="ParallelEncodingNetwork"):
    """Parallel encoding network which effectively runs ``n`` individual encoding
    network simultaneuosl.

    Args:
        input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
            the input. If nested, then ``preprocessing_combiner`` must not be
            None.
        n (int): number of parallel networks
        output_tensor_spec (None|TensorSpec): spec for the output, excluding
            the dimension of parallel networks ``n``. If None, the output
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
        use_fc_ln (bool): whether use Layer Normalization for fc layers.
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
        last_use_fc_ln (bool): whether use Layer Normalization for the last
            fc layer.
        name (str):
    Returns:
        the parallelized network
    """
    net = EncodingNetwork(
        input_tensor_spec=input_tensor_spec,
        output_tensor_spec=output_tensor_spec,
        input_preprocessors=input_preprocessors,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        activation=activation,
        kernel_initializer=kernel_initializer,
        use_fc_bn=use_fc_bn,
        last_layer_size=last_layer_size,
        last_activation=last_activation,
        last_kernel_initializer=last_kernel_initializer,
        last_use_fc_bn=last_use_fc_bn,
        name=name)
    return net.make_parallel(n, True)


@alf.configurable
class LSTMEncodingNetwork(_Sequential):
    """LSTM cells followed by an encoding network."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec=None,
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
            output_tensor_spec (None|TensorSpec): spec for the output. If None,
                the output tensor spec will be assumed as
                ``TensorSpec((output_size, ))``, where ``output_size`` is
                inferred from network output. Otherwise, the output tensor
                spec will be ``output_tensor_spec`` and the network output
                will be reshaped according to ``output_tensor_spec``.
                Note that ``output_tensor_spec`` is only used for reshaping
                the network outputs for interpretation purpose and is not used
                for specifying any network layers.
            input_preprocessors (nested Network|nn.Module|None): a nest of
                input preprocessors, each of which will be applied to the
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

        nets = []
        if (input_preprocessors or preprocessing_combiner or conv_layer_params
                or pre_fc_layer_params):
            net = EncodingNetwork(
                input_tensor_spec=input_tensor_spec,
                input_preprocessors=input_preprocessors,
                preprocessing_combiner=preprocessing_combiner,
                conv_layer_params=conv_layer_params,
                fc_layer_params=pre_fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer)
            input_size = net.output_spec.shape[0]
            nets.append(net)
        else:
            input_size = input_tensor_spec.shape[0]

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        else:
            assert isinstance(hidden_size, tuple)

        cells = []
        for hs in hidden_size:
            cells.append(
                alf.nn.LSTMCell(input_size=input_size, hidden_size=hs))
            input_size = hs

        if lstm_output_layers is None:
            lstm_output_layers = list(range(len(hidden_size)))
        elif type(lstm_output_layers) == int:
            lstm_output_layers = [lstm_output_layers]
        lstm_output_layers = [
            len(cells) + i if i < 0 else i for i in lstm_output_layers
        ]
        if lstm_output_layers == [len(cells) - 1]:
            nets.extend(cells)
        else:
            if type(lstm_output_layers) == int:
                lstm_output_layers = [lstm_output_layers]
            lstms = dict(('lstm%s' % i, cell) for i, cell in enumerate(cells))
            lstms['o'] = (
                tuple(
                    'lstm%s' % i
                    for i in lstm_output_layers),  # the inputs for NestConcat
                alf.layers.NestConcat())
            nets.append(alf.nn.Sequential(**lstms, name='lstm_block'))
            input_size = sum(hidden_size[i] for i in lstm_output_layers)

        if post_fc_layer_params is not None or last_layer_size is not None:
            net = EncodingNetwork(
                input_tensor_spec=TensorSpec((input_size, )),
                fc_layer_params=post_fc_layer_params,
                activation=activation,
                kernel_initializer=kernel_initializer,
                last_layer_size=last_layer_size,
                last_activation=last_activation,
                last_kernel_initializer=last_kernel_initializer)
            nets.append(net)
            input_size = net.output_spec.numel

        if output_tensor_spec is not None:
            assert output_tensor_spec.numel == input_size, (
                "network output "
                "size {a} is inconsistent with specified out_tensor_spec "
                "of size {b}".format(a=input_size, b=output_tensor_spec.numel))
            nets.append(alf.layers.Reshape(output_tensor_spec.shape))

        super().__init__(nets, input_tensor_spec=input_tensor_spec, name=name)

    def make_parallel(self, n: int, allow_non_parallel_input=False):
        """Make a parallelized version of ``module``.

        A parallel network has ``n`` copies of network with the same structure but
        different independently initialized parameters. The parallel network can
        process a batch of the data with shape [batch_size, n, ...] using ``n``
        networks with same structure.

        Args:
            n (int): the number of copies
            allow_non_parallel_input (bool): if True, the returned network will
                also accept non-parallel input with shape [batch_size, ...]. In
                this case, the network will check whether the input is parallel
                input. If not, the input will be automatically replicated ``n``
                times at the beginning.
        Returns:
            the parallelized network.
        """
        pnet = super().make_parallel(n)
        if allow_non_parallel_input:
            return _ReplicateInputForParallel(
                self.input_tensor_spec, n, pnet, name=pnet.name)
        else:
            return pnet
