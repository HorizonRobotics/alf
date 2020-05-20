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
import copy

import torch
import torch.nn as nn

from alf.initializers import variance_scaling_init
from alf.nest.utils import get_outer_rank
from alf.tensor_specs import TensorSpec
from alf.utils import common
from alf.utils.math_ops import identity


def normalize_along_batch_dims(x, mean, variance, variance_epsilon):
    """Normalizes a tensor by ``mean`` and ``variance``, which are expected to have
    the same tensor spec with the inner dims of ``x``.

    Args:
        x (Tensor): a tensor of (``[D1, D2, ..] + shape``), where ``D1``, ``D2``, ..
            are arbitrary leading batch dims (can be empty).
        mean (Tensor): a tensor of ``shape``
        variance (Tensor): a tensor of ``shape``
        variance_epsilon (float): A small float number to avoid dividing by 0.
    Returns:
        Normalized tensor.
    """
    spec = TensorSpec.from_tensor(mean)
    assert spec == TensorSpec.from_tensor(variance), \
        "The specs of mean and variance must be equal!"

    bs = BatchSquash(get_outer_rank(x, spec))
    x = bs.flatten(x)

    variance_epsilon = torch.as_tensor(variance_epsilon).to(variance.dtype)
    inv = torch.rsqrt(variance + variance_epsilon)
    x = (x - mean.to(x.dtype)) * inv.to(x.dtype)

    x = bs.unflatten(x)
    return x


class BatchSquash(object):
    """Facilitates flattening and unflattening batch dims of a tensor. Copied
    from `tf_agents`.

    Exposes a pair of matched flatten and unflatten methods. After flattening
    only 1 batch dimension will be left. This facilitates evaluating networks
    that expect inputs to have only 1 batch dimension.
    """

    def __init__(self, batch_dims):
        """Create two tied ops to flatten and unflatten the front dimensions.

        Args:
            batch_dims (int): Number of batch dimensions the flatten/unflatten
                ops should handle.

        Raises:
            ValueError: if batch dims is negative.
        """
        if batch_dims < 0:
            raise ValueError('Batch dims must be non-negative.')
        self._batch_dims = batch_dims
        self._original_tensor_shape = None

    def flatten(self, tensor):
        """Flattens and caches the tensor's batch_dims."""
        if self._batch_dims == 1:
            return tensor
        self._original_tensor_shape = tensor.shape
        return torch.reshape(tensor,
                             (-1, ) + tuple(tensor.shape[self._batch_dims:]))

    def unflatten(self, tensor):
        """Unflattens the tensor's batch_dims using the cached shape."""
        if self._batch_dims == 1:
            return tensor

        if self._original_tensor_shape is None:
            raise ValueError('Please call flatten before unflatten.')

        return torch.reshape(
            tensor, (tuple(self._original_tensor_shape[:self._batch_dims]) +
                     tuple(tensor.shape[1:])))


@gin.configurable
class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self._num_classes = num_classes

    def forward(self, input):
        return nn.functional.one_hot(
            input, num_classes=self._num_classes).to(torch.float32)


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
        module instead of ``nn.Linear`` if you really care about weight std after
        init.
        Args:
            input_size (int): input size
            output_size (int): output size
            activation (torch.nn.functional):
            use_bias (bool): whether use bias
            kernel_initializer (Callable): initializer for the FC layer kernel.
                If none is provided a ``variance_scaling_initializer`` with gain as
                ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to
                the std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        # get the argument list with vals
        self._kwargs = copy.deepcopy(locals())
        self._kwargs.pop('self')
        self._kwargs.pop('__class__')

        super(FC, self).__init__()

        self._activation = activation
        self._linear = nn.Linear(input_size, output_size, bias=use_bias)
        self._kernel_initializer = kernel_initializer
        self._kernel_init_gain = kernel_init_gain
        self._bias_init_value = bias_init_value
        self._use_bias = use_bias
        self.reset_parameters()

    def reset_parameters(self):
        if self._kernel_initializer is None:
            variance_scaling_init(
                self._linear.weight.data,
                gain=self._kernel_init_gain,
                nonlinearity=self._activation)
        else:
            self._kernel_initializer(self._linear.weight.data)

        if self._use_bias:
            nn.init.constant_(self._linear.bias.data, self._bias_init_value)

    def forward(self, inputs):
        return self._activation(self._linear(inputs))

    @property
    def weight(self):
        return self._linear.weight

    @property
    def bias(self):
        return self._linear.bias

    def make_parallel(self, n):
        """Create a ``ParallelFC`` using ``n`` replicas of ``self``.
        The initialized layer parameters will be different.
        """
        return ParallelFC(n=n, **self._kwargs)


@gin.configurable
class ParallelFC(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n,
                 activation=identity,
                 use_bias=True,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """Parallel FC layer.

        It is equivalent to ``n`` separate FC layers with the same
        ``input_size`` and ``output_size``.

        Args:
            input_size (int): input size
            output_size (int): output size
            n (int): n independent ``FC`` layers
            activation (torch.nn.functional):
            use_bias (bool): whether use bias
            kernel_initializer (Callable): initializer for the FC layer kernel.
                If none is provided a ``variance_scaling_initializer`` with gain
                as ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to
                the std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        super().__init__()
        self._activation = activation
        self._weight = nn.Parameter(torch.Tensor(n, output_size, input_size))
        if use_bias:
            self._bias = nn.Parameter(torch.Tensor(n, output_size))
        else:
            self._bias = None

        for i in range(n):
            if kernel_initializer is None:
                variance_scaling_init(
                    self._weight.data[i],
                    gain=kernel_init_gain,
                    nonlinearity=self._activation)
            else:
                kernel_initializer(self._weight.data[i])

        if use_bias:
            nn.init.constant_(self._bias.data, bias_init_value)

    def forward(self, inputs):
        """Forward

        Args:
            inputs (torch.Tensor): with shape ``[B, n, input_size]`` or ``[B, input_size]``
        Returns:
            torch.Tensor with shape ``[B, n, output_size]``
        """
        n, k, l = self._weight.shape
        if inputs.ndim == 2:
            assert inputs.shape[1] == l, (
                "inputs has wrong shape %s. Expecting (B, %d)" % (inputs.shape,
                                                                  l))
            inputs = inputs.unsqueeze(0).expand(n, *inputs.shape)
        elif inputs.ndim == 3:
            assert (inputs.shape[1] == n and inputs.shape[2] == l), (
                "inputs has wrong shape %s. Expecting (B, %d, %d)" %
                (inputs.shape, n, l))
            inputs = inputs.transpose(0, 1)  # [n, B, l]
        else:
            raise ValueError("Wrong inputs.ndim=%d" % inputs.ndim)

        if self.bias is not None:
            y = torch.baddbmm(
                self._bias.unsqueeze(1), inputs,
                self.weight.transpose(1, 2))  # [n, B, k]
        else:
            y = torch.bmm(inputs, self._weight.transpose(1, 2))  # [n, B, k]
        y = y.transpose(0, 1)  # [B, n, k]
        return self._activation(y)

    @property
    def weight(self):
        """Get the weight Tensor.

        Returns:
            Tensor: with shape (n, output_size, input_size). ``weight[i]`` is
                the weight for the i-th FC layer. ``weight[i]`` can be used for
                ``FC`` layer with the same ``input_size`` and ``output_size``
        """
        return self._weight

    @property
    def bias(self):
        """Get the bias Tensor.

        Returns:
            Tensor: with shape (n, output_size). ``bias[i]`` is the bias for the
                i-th FC layer. ``bias[i]`` can be used for ``FC`` layer with
                the same ``input_size`` and ``output_size``
        """
        return self._bias


@gin.configurable
class Conv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=torch.relu_,
                 strides=1,
                 padding=0,
                 use_bias=True,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A 2D Conv layer that's also responsible for activation and customized
        weights initialization. An auto gain calculation might depend on the
        activation following the conv layer. Suggest using this wrapper module
        instead of ``nn.Conv2d`` if you really care about weight std after init.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            activation (torch.nn.functional):
            strides (int or tuple):
            padding (int or tuple):
            use_bias (bool):
            kernel_initializer (Callable): initializer for the conv layer kernel.
                If None is provided a variance_scaling_initializer with gain as
                ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
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

        if kernel_initializer is None:
            variance_scaling_init(
                self._conv2d.weight.data,
                gain=kernel_init_gain,
                nonlinearity=self._activation)
        else:
            kernel_initializer(self._conv2d.weight.data)

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
class ParallelConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 n,
                 activation=torch.relu_,
                 strides=1,
                 padding=0,
                 use_bias=True,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A parallel 2D Conv layer that can be used to perform n independent
        2D convolutions in parallel.

        It is equivalent to ``n`` separate ``Conv2D`` layers with the same
        ``in_channels`` and ``out_channels``.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            n (int): n independent ``Conv2D`` layers
            activation (torch.nn.functional):
            strides (int or tuple):
            padding (int or tuple):
            use_bias (bool):
            kernel_initializer (Callable): initializer for the conv layer kernel.
                If None is provided a ``variance_scaling_initializer`` with gain
                as ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        super(ParallelConv2D, self).__init__()
        self._activation = activation
        self._n = n
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = common.tuplify2d(kernel_size)
        self._conv2d = nn.Conv2d(
            in_channels * n,
            out_channels * n,
            kernel_size,
            groups=n,
            stride=strides,
            padding=padding,
            bias=use_bias)

        for i in range(n):
            if kernel_initializer is None:
                variance_scaling_init(
                    self._conv2d.weight.data[i * out_channels:(i + 1) *
                                             out_channels],
                    gain=kernel_init_gain,
                    nonlinearity=self._activation)
            else:
                kernel_initializer(
                    self._conv2d.weight.data[i * out_channels:(i + 1) *
                                             out_channels])

        # [n*C', C, kernel_size, kernel_size]->[n, C', C, kernel_size, kernel_size]
        self._weight = self._conv2d.weight.view(
            self._n, self._out_channels, self._in_channels,
            self._kernel_size[0], self._kernel_size[1])

        if use_bias:
            nn.init.constant_(self._conv2d.bias.data, bias_init_value)
            # [n*C']->[n, C']
            self._bias = self._conv2d.bias.view(self._n, self._out_channels)
        else:
            self._bias = None

    def forward(self, img):
        """Forward

        Args:
            img (torch.Tensor): with shape ``[B, C, H, W]``
                                        or ``[B, n, C, H, W]``
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

        if img.ndim == 4:
            # the shared input case
            assert img.shape[1] == self._in_channels, (
                "Input img has wrong shape %s. Expecting (B, %d, H, W)" %
                (img.shape, self._in_channels))

            img = img.unsqueeze(1).expand(img.shape[0], self._n,
                                          *img.shape[1:])
        elif img.ndim == 5:
            # the non-shared case
            assert (
                img.shape[1] == self._n
                and img.shape[2] == self._in_channels), (
                    "Input img has wrong shape %s. Expecting (B, %d, %d, H, W)"
                    % (img.shape, self._n, self._in_channels))
        else:
            raise ValueError("Wrong img.ndim=%d" % img.ndim)

        # merge replica and channels
        img = img.reshape(img.shape[0], img.shape[1] * img.shape[2],
                          *img.shape[3:])

        res = self._activation(self._conv2d(img))

        # reshape back: [B, n*C', H', W'] -> [B, n, C', H', W']
        res = res.reshape(res.shape[0], self._n, self._out_channels,
                          *res.shape[2:])
        return res

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias


@gin.configurable
class ConvTranspose2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=torch.relu_,
                 strides=1,
                 padding=0,
                 use_bias=True,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A 2D ConvTranspose layer that's also responsible for activation and
        customized weights initialization. An auto gain calculation might depend
        on the activation following the conv layer. Suggest using this wrapper
        module instead of ``nn.ConvTranspose2d`` if you really care about weight std
        after init.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            activation (torch.nn.functional):
            strides (int or tuple):
            padding (int or tuple):
            use_bias (bool):
            kernel_initializer (Callable): initializer for the conv_trans layer.
                If None is provided a variance_scaling_initializer with gain as
                ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
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
        if kernel_initializer is None:
            variance_scaling_init(
                self._conv_trans2d.weight.data,
                gain=kernel_init_gain,
                nonlinearity=self._activation,
                transposed=True)
        else:
            kernel_initializer(self._conv_trans2d.weight.data)

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


@gin.configurable
class ParallelConvTranspose2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 n,
                 activation=torch.relu_,
                 strides=1,
                 padding=0,
                 use_bias=True,
                 kernel_initializer=None,
                 kernel_init_gain=1.0,
                 bias_init_value=0.0):
        """A parallel ConvTranspose2D layer that can be used to perform n
        independent 2D transposed convolutions in parallel.

        Args:
            in_channels (int): channels of the input image
            out_channels (int): channels of the output image
            kernel_size (int or tuple):
            n (int): n independent ``ConvTranspose2D`` layers
            activation (torch.nn.functional):
            strides (int or tuple):
            padding (int or tuple):
            use_bias (bool):
            kernel_initializer (Callable): initializer for the conv_trans layer.
                If None is provided a ``variance_scaling_initializer`` with gain
                as ``kernel_init_gain`` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                ``kernel_initializer`` is not None.
            bias_init_value (float): a constant
        """
        super(ParallelConvTranspose2D, self).__init__()
        self._activation = activation
        self._n = n
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = common.tuplify2d(kernel_size)
        self._conv_trans2d = nn.ConvTranspose2d(
            in_channels * n,
            out_channels * n,
            kernel_size,
            groups=n,
            stride=strides,
            padding=padding,
            bias=use_bias)

        for i in range(n):
            if kernel_initializer is None:
                variance_scaling_init(
                    self._conv_trans2d.weight.data[i * in_channels:(i + 1) *
                                                   in_channels],
                    gain=kernel_init_gain,
                    nonlinearity=self._activation)
            else:
                kernel_initializer(
                    self._conv_trans2d.weight.data[i * in_channels:(i + 1) *
                                                   in_channels])

        # [n*C, C', kernel_size, kernel_size]->[n, C, C', kernel_size, kernel_size]
        self._weight = self._conv_trans2d.weight.view(
            self._n, self._in_channels, self._out_channels,
            self._kernel_size[0], self._kernel_size[1])

        if use_bias:
            nn.init.constant_(self._conv_trans2d.bias.data, bias_init_value)
            # [n*C]->[n, C]
            self._bias = self._conv_trans2d.bias.view(self._n,
                                                      self._out_channels)
        else:
            self._bias = None

    def forward(self, img):
        """Forward

        Args:
            img (torch.Tensor): with shape ``[B, C, H, W]``
                                        or ``[B, n, C, H, W]``
                where the meaning of the symbols are:
                - ``B``: batch size
                - ``n``: number of replicas
                - ``C``: number of channels
                - ``H``: image height
                - ``W``: image width.
                When the shape of img is ``[B, C, H, W]``, all the n transposed 2D
                Conv operations will take img as the same shared input.
                When the shape of img is ``[B, n, C, H, W]``, each transposed 2D
                Conv operator will have its own input data by slicing img.

        Returns:
            torch.Tensor with shape ``[B, n, C', H', W']``
                where the meaning of the symbols are:
                - ``B``: batch
                - ``n``: number of replicas
                - ``C'``: number of output channels
                - ``H'``: output height
                - ``W'``: output width
        """
        if img.ndim == 4:
            # the shared input case
            assert img.shape[1] == self._in_channels, (
                "Input img has wrong shape %s. Expecting (B, %d, H, W)" %
                (img.shape, self._in_channels))

            img = img.unsqueeze(1).expand(img.shape[0], self._n,
                                          *img.shape[1:])
        elif img.ndim == 5:
            # the non-shared case
            assert (
                img.shape[1] == self._n
                and img.shape[2] == self._in_channels), (
                    "Input img has wrong shape %s. Expecting (B, %d, %d, H, W)"
                    % (img.shape, self._n, self._in_channels))
        else:
            raise ValueError("Wrong img.ndim=%d" % img.ndim)

        # merge replica and channels
        img = img.reshape(img.shape[0], img.shape[1] * img.shape[2],
                          *img.shape[3:])

        res = self._activation(self._conv_trans2d(img))

        # reshape back: [B, n*C', H', W'] -> [B, n, C', H', W']
        res = res.reshape(res.shape[0], self._n, self._out_channels,
                          res.shape[2], res.shape[3])
        return res

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias


class Reshape(nn.Module):
    def __init__(self, shape):
        """A layer for reshape the tensor.

        The result of this layer is a tensor reshaped to ``(B, *shape)`` where
        ``B`` is ``x.shape[0]``

        Args:
            shape (tuple): desired shape not including the batch dimension.
        """
        super().__init__()
        self._shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self._shape)


def _tuplify2d(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (x, x)


def _conv_transpose_2d(in_channels,
                       out_channels,
                       kernel_size,
                       stride=1,
                       padding=0):
    # need output_padding so that output_size is stride * input_size
    # See https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d
    output_padding = stride + 2 * padding - kernel_size
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding)


@gin.configurable(whitelist=['v1_5', 'with_batch_normalization'])
class BottleneckBlock(nn.Module):
    """Bottleneck block for ResNet.

    We allow two slightly different architectures:
    * v1: Placing the stride at the first 1x1 convolution as described in the
      original ResNet paper `Deep residual learning for image recognition
      <https://arxiv.org/abs/1512.03385>`_.
    * v1.5: Placing the stride for downsampling at 3x3 convolution. This variant
      is also known as ResNet V1.5 and improves accuracy according to
      `<https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.
    """

    def __init__(self,
                 in_channels,
                 kernel_size,
                 filters,
                 stride,
                 transpose=False,
                 v1_5=True,
                 with_batch_normalization=True):
        """
        Args:
            kernel_size (int): the kernel size of middle layer at main path
            filters (int): the filters of 3 layer at main path
            stride (int): stride for this block.
            transpose (bool): a bool indicate using ``Conv2D`` or ``Conv2DTranspose``.
                If two BottleneckBlock layers ``L`` and ``LT`` are constructed
                with the same arguments except ``transpose``, it is gauranteed that
                ``LT(L(x)).shape == x.shape`` if ``x.shape[-2:]`` can be divided
                by ``stride``.
            v1_5 (bool): whether to use the ResNet V1.5 structure
            with_batch_normalization (bool): whether to include batch normalization.
                Note that standard ResNet uses batch normalization.
        Return:
            Output tensor for the block
        """
        super().__init__()
        filters1, filters2, filters3 = filters

        conv_fn = _conv_transpose_2d if transpose else nn.Conv2d

        padding = (kernel_size - 1) // 2
        if v1_5:
            a = conv_fn(in_channels, filters1, 1)
            b = conv_fn(filters1, filters2, kernel_size, stride, padding)
        else:
            a = conv_fn(in_channels, filters1, 1, stride)
            b = conv_fn(filters1, filters2, kernel_size, 1, padding)

        nn.init.kaiming_normal_(a.weight.data)
        nn.init.zeros_(a.bias.data)
        nn.init.kaiming_normal_(b.weight.data)
        nn.init.zeros_(b.bias.data)

        c = conv_fn(filters2, filters3, 1)
        nn.init.kaiming_normal_(c.weight.data)
        nn.init.zeros_(c.bias.data)

        s = conv_fn(in_channels, filters3, 1, stride)
        nn.init.kaiming_normal_(s.weight.data)
        nn.init.zeros_(s.bias.data)

        relu = nn.ReLU(inplace=True)

        if with_batch_normalization:
            core_layers = nn.Sequential(a, nn.BatchNorm2d(filters1), relu, b,
                                        nn.BatchNorm2d(filters2), relu, c,
                                        nn.BatchNorm2d(filters3))
            shortcut_layers = nn.Sequential(s, nn.BatchNorm2d(filters3))
        else:
            core_layers = nn.Sequential(a, relu, b, relu, c)
            shortcut_layers = s

        self._core_layers = core_layers
        self._shortcut_layers = shortcut_layers

    def forward(self, inputs):
        core = self._core_layers(inputs)
        shortcut = self._shortcut_layers(inputs)

        return torch.relu_(core + shortcut)

    def calc_output_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        y = self.forward(x)
        return y.shape[1:]
