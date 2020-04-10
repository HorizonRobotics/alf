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
import gin
import torch
import torch.nn as nn


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


@gin.configurable(whitelist=['v1_5'])
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
                 v1_5=True):
        """
        Args:
            kernel_size (int): the kernel size of middle layer at main path
            filters (int): the filters of 3 layer at main path
            stride (int): stride for this block
            transpose (bool): a bool indicate using Conv2D or Conv2DTranspose
            v1_5 (bool): whether to use the ResNet V1.5 structure
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

        core_layers = nn.Sequential(a, nn.BatchNorm2d(filters1), nn.ReLU(), b,
                                    nn.BatchNorm2d(filters2), nn.ReLU(), c,
                                    nn.BatchNorm2d(filters3))

        s = conv_fn(in_channels, filters3, 1, stride)
        nn.init.kaiming_normal_(s.weight.data)
        nn.init.zeros_(s.bias.data)

        shortcut_layers = nn.Sequential(s, nn.BatchNorm2d(filters3))

        self._core_layers = core_layers
        self._shortcut_layers = shortcut_layers

    def forward(self, inputs):
        core = self._core_layers(inputs)
        shortcut = self._shortcut_layers(inputs)

        return nn.functional.relu(core + shortcut)

    def calc_output_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        y = self.forward(x)
        return y.shape[1:]
