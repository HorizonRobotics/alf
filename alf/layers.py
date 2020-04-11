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

from alf.nest.utils import get_outer_rank
from alf.networks.initializers import variance_scaling_init
from alf.utils.math_ops import identity
from alf.tensor_specs import TensorSpec


def normalize_along_batch_dims(x, mean, variance, variance_epsilon):
    """Normalizes a tensor by `mean` and `variance`, which are expected to have
    the same tensor spec with the inner dims of `x`.

    Args:
        x (Tensor): a tensor of ([D1, D2, ..] + `shape`), where D1, D2, .. are
            arbitrary leading batch dims (can be empty).
        mean (Tensor): a tensor of `shape`
        variance (Tensor): a tensor of `shape`
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
    from tf_agents.

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
            kernel_initializer (Callable): initializer for the FC layer kernel.
                If none is provided a variance_scaling_initializer with gain as
                `kernel_init_gain` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to
                the std of kernel init distribution. It will be ignored if
                `kernel_initializer` is not None.
            bias_init_value (float): a constant
        """
        super(FC, self).__init__()
        self._activation = activation
        self._linear = nn.Linear(input_size, output_size, bias=use_bias)

        if kernel_initializer is None:
            variance_scaling_init(
                self._linear.weight.data,
                gain=kernel_init_gain,
                nonlinearity=self._activation)
        else:
            kernel_initializer(self._linear.weight.data)

        if use_bias:
            nn.init.constant_(self._linear.bias.data, bias_init_value)

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
                 kernel_initializer=None,
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
            kernel_initializer (Callable): initializer for the conv layer kernel.
                If None is provided a variance_scaling_initializer with gain as
                `kernel_init_gain` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                `kernel_initializer` is not None.
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
class ConvTranspose2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=torch.relu,
                 strides=1,
                 padding=0,
                 use_bias=True,
                 kernel_initializer=None,
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
            kernel_initializer (Callable): initializer for the conv_trans layer.
                If None is provided a variance_scaling_initializer with gain as
                `kernel_init_gain` will be used.
            kernel_init_gain (float): a scaling factor (gain) applied to the
                std of kernel init distribution. It will be ignored if
                `kernel_initializer` is not None.
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

        return nn.functional.relu_(core + shortcut)

    def calc_output_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        y = self.forward(x)
        return y.shape[1:]
