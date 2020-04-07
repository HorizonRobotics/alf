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

import torch
import torch.nn as nn


def _tuplify2d(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (x, x)


class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 filters,
                 strides=(2, 2),
                 transpose=False):
        """A resnet bottleneck block.

        Reference:
        `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_

        Args:
            kernel_size (tuple[int]|int): the kernel size of middle layer at main path
            filters (tuple[int]): the filters of 3 layer at main path
            strides (tuple[int]|int): stride for first layer in the block
            transpose (bool): a bool indicate using Conv2D or Conv2DTranspose
        Return:
            Output tensor for the block
        """
        super().__init__()
        filters1, filters2, filters3 = filters
        kernel_size = _tuplify2d(kernel_size)

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        a = conv_fn(in_channels, filters1, (1, 1), stride=strides)
        nn.init.kaiming_normal_(a.weight.data, 'relu')

        b = conv_fn(
            filters1,
            filters2,
            kernel_size,
            padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2))
        nn.init.kaiming_normal_(b.weight.data, 'relu')

        c = conv_fn(filters2, filters3, (1, 1))
        nn.init.kaiming_normal_(c.weight.data, 'relu')

        core_layers = nn.Sequential([
            a,
            nn.BatchNorm2d(filters1),
            nn.ReLU(), b,
            nn.BatchNorm2d(filters2),
            nn.ReLU(), c,
            nn.BatchNorm2d(filters3)
        ])

        s = conv_fn(in_channels, filters3, (1, 1), stride=strides)
        nn.init.kaiming_normal_(s.weight.data, 'relu')

        shortcut_layers = nn.Sequential([s, nn.BatchNorm2d(filters3)])

        self._core_layers = core_layers
        self._shortcut_layers = shortcut_layers

    def forward(self, inputs):
        core = self._core_layers(inputs)
        shortcut = self._shortcut_layers(inputs)

        return nn.functional.relu(core + shortcut)
