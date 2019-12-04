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

import tensorflow as tf
from tensorflow.keras import layers


class BottleneckBlock(tf.keras.models.Model):
    def __init__(self,
                 kernel_size,
                 filters,
                 strides=(2, 2),
                 transpose=False,
                 name='BottleneckBlock'):
        """A resnet bottleneck block.

        Reference:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385

        Args:
            kernel_size (tuple[int]|int): the kernel size of middle layer at main path
            filters (tuple[int]): the filters of 3 layer at main path
            strides (tuple[int]|int): stride for first layer in the block
            transpose (bool): a bool indicate using Conv2D or Conv2DTranspose
            name (str): block name
        Return:
            Output tensor for the block
        """
        super().__init__(name=name)
        filters1, filters2, filters3 = filters
        if tf.keras.backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res_branch'
        bn_name_base = 'bn_branch'

        conv_fn = layers.Conv2DTranspose if transpose else layers.Conv2D

        a = conv_fn(
            filters1, (1, 1),
            strides=strides,
            kernel_initializer='he_normal',
            name=conv_name_base + '2a')

        b = conv_fn(
            filters2,
            kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            name=conv_name_base + '2b')

        c = conv_fn(
            filters3, (1, 1),
            kernel_initializer='he_normal',
            name=conv_name_base + '2c')

        core_layers = [
            a,
            layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a'),
            layers.Activation('relu'), b,
            layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b'),
            layers.Activation('relu'), c,
            layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')
        ]

        s = conv_fn(
            filters3, (1, 1),
            strides=strides,
            kernel_initializer='he_normal',
            name=conv_name_base + '1')

        shortcut_layers = [
            s,
            layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')
        ]

        self._core_layers = core_layers
        self._shortcut_layers = shortcut_layers

    def call(self, inputs, training=True):
        core, shortcut = inputs, inputs
        for l in self._core_layers:
            core = l(core, training=training)

        for l in self._shortcut_layers:
            shortcut = l(shortcut, training=training)

        return tf.nn.relu(core + shortcut)
