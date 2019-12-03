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


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               transpose=False):
    """A resnet bottleneck block.

    Reference:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385

    It's extended from "https://github.com/keras-team/keras-applications/ \
       blob/976050c468ff949bcbd9b9cf64fe1d5c81db3f3a/ \
       keras_applications/resnet50.py#L82-L139",
    it supports transposed convolution

    Args:
        input_tensor (): input tensor
        kernel_size (): the kernel size of middle layer at main path
        filters (tuple[int]): the filters of 3 layer at main path
        stage (int): current stage label, used for generating layer names
        block (str): current block label, used for generating layer names
        strides (tuple[int]|int): stride for first layer in the block
        transpose (bool): a bool indicate using Conv2D or Conv2DTranspose
    Return:
        Output tensor for the block
    """
    filters1, filters2, filters3 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    conv_fn = layers.Conv2DTranspose if transpose else layers.Conv2D

    x = conv_fn(
        filters1, (1, 1),
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = conv_fn(
        filters2,
        kernel_size,
        padding='same',
        kernel_initializer='he_normal',
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = conv_fn(
        filters3, (1, 1),
        kernel_initializer='he_normal',
        name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = conv_fn(
        filters3, (1, 1),
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x
