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

import numpy as np
import gin.tf
import tensorflow as tf


@gin.configurable
class ImageEncodingNetwork(tf.keras.models.Model):
    """
    A general template class for creating convolutional encoding networks.
    """

    def __init__(self,
                 conv_layer_params,
                 padding="valid",
                 postprocess_fc_layer_params=None,
                 activation_fn=tf.nn.relu,
                 output_activation_fn=None,
                 name="image_encoding_network"):
        """
        Initialize the layers for encoding an image into a latent vector.

        Args:
            conv_layer_params (list[tuple]): a non-empty list of elements
                (num_filters, kernel_size, strides).
            padding (str): "same" or "valid", see tf.keras.layers.Conv2DTranspose
            postprocess_fc_layer_params (tuple[int]): a list of fc layer units.
                These fc layers are used for postprocessing the conv output. If
                None, then the output would preserve the shape of an image; if (),
                the output will be still flattened.
            activation_fn (tf.nn.activation): activation for conv layers
            output_activation_fn (tf.nn.activation): only used when there are
                postprocessing FC layers
            name (str): network name
        """
        super().__init__(name=name)

        assert isinstance(conv_layer_params, list)
        assert len(conv_layer_params) > 0

        self._postprocess_fc_layers = []
        if postprocess_fc_layer_params:
            for size in postprocess_fc_layer_params[:-1]:
                self._postprocess_fc_layers.append(
                    tf.keras.layers.Dense(size, activation=activation_fn))
            self._postprocess_fc_layers.append(
                tf.keras.layers.Dense(
                    postprocess_fc_layer_params[-1],
                    activation=output_activation_fn))

        self._postprocess_fc_layer_params = postprocess_fc_layer_params
        self._conv_layer_params = conv_layer_params
        self._padding = padding
        self._conv_layers = []
        for filters, kernel_size, strides in conv_layer_params:
            self._conv_layers.append(
                tf.keras.layers.Conv2D(
                    padding=padding,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    activation=activation_fn))

    def output_shape(self, input_size, conv=False):
        """Return the output shape given the input image size.

        How to calculate the output size:

            if padding=="same", then H = (H1 - 1) // strides + 1
            if padding=="valid", then H = (H1 - HF) // strides + 1
        where H = output size, H1 = input size, HF = size of kernel

        Args:
            input_size (tuple): the input image size (height, width)
            conv (bool): whether calculate the intermediate conv output size
                or the final FC output size. If True, return the last conv's
                shape regardless of self._postprocess_fc_layers.

        Returns:
            a tuple representing the output shape
        """
        if self._postprocess_fc_layers and not conv:
            return (self._postprocess_fc_layer_params[-1], )

        height, width = input_size
        for filters, kernel_size, strides in self._conv_layer_params:
            if self._padding == "same":
                height = (height - 1) // strides + 1
                width = (width - 1) // strides + 1
            else:
                height = (height - kernel_size) // strides + 1
                width = (width - kernel_size) // strides + 1
        shape = (width, height, filters)

        if not conv and self._postprocess_fc_layer_params == ():
            return (np.prod(shape), )
        else:
            return shape

    def call(self, inputs):
        z = inputs
        for conv_l in self._conv_layers:
            z = conv_l(z)
        if self._postprocess_fc_layer_params is not None:
            z = tf.reshape(z, [tf.shape(z)[0], -1])
            for fc_l in self._postprocess_fc_layers:
                z = fc_l(z)
        return z


@gin.configurable
class ImageDecodingNetwork(tf.keras.models.Model):
    """
    A general template class for creating transposed convolutional decoding networks.
    """

    def __init__(self,
                 deconv_layer_params,
                 start_decoding_height,
                 start_decoding_width,
                 start_decoding_filters,
                 padding="same",
                 preprocess_fc_layer_params=None,
                 activation_fn=tf.nn.relu,
                 output_activation_fn=tf.nn.tanh,
                 name="image_decoding_network"):
        """
        Initialize the layers for decoding a latent vector into an image.

        Args:
            deconv_layer_params (list[tuple]): a non-empty list of elements
                (num_filters, kernel_size, strides).
            start_decoding_height (int): the initial height we'd like to have for
                the feature map
            start_decoding_width (int): the initial width we'd like to have for
                the feature map
            start_decoding_filters (int): the initial number of filters we'd like
                to have for the feature map. Note that given this value and
                `start_decoding_size`, we always first project an input latent
                vector into a vector of an appropriate length so that it can be
                reshaped into (`start_decoding_size`, `start_decoding_size`,
                `start_decoding_filters`).
            padding (str): "same" or "valid", see tf.keras.layers.Conv2DTranspose
            preprocess_fc_layer_params (tuple[int]): a list of fc layer units.
                These fc layers are used for preprocessing the latent vector before
                transposed convolutions.
            activation_fn (tf.nn.activation): activation for hidden layers
            output_activation_fn (tf.nn.activation): activation for the output
                layer. Usually our image inputs are normalized to [0, 1] or [-1, 1],
                so this function should be tf.nn.sigmoid or tf.nn.tanh.
            name (str): network name
        """
        super().__init__(name=name)

        assert isinstance(deconv_layer_params, list)
        assert len(deconv_layer_params) > 0

        self._preprocess_fc_layers = []
        if preprocess_fc_layer_params is not None:
            for size in preprocess_fc_layer_params:
                self._preprocess_fc_layers.append(
                    tf.keras.layers.Dense(size, activation=activation_fn))

        # We always assume "channels_last" !
        self._start_decoding_shape = [
            start_decoding_height, start_decoding_width, start_decoding_filters
        ]
        self._preprocess_fc_layers.append(
            tf.keras.layers.Dense(
                tf.reduce_prod(self._start_decoding_shape),
                activation=activation_fn))

        self._padding = padding
        self._deconv_layer_params = deconv_layer_params
        self._deconv_layers = []
        for i, (filters, kernel_size,
                strides) in enumerate(deconv_layer_params):
            act_fn = activation_fn
            if i == len(deconv_layer_params) - 1:
                act_fn = output_activation_fn
            self._deconv_layers.append(
                tf.keras.layers.Conv2DTranspose(
                    padding=padding,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    activation=act_fn))

    def output_shape(self):
        """Return the output image shape given the start_decoding_shape.

        How to calculate the output size:
            if padding=="same", then H = H1 * strides
            if padding=="valid", then H = (H1-1) * strides + HF
        where H = output size, H1 = input size, HF = size of kernel

        Returns:
            a tuple representing the output shape
        """
        height, width, _ = self._start_decoding_shape
        for filters, kernel_size, strides in self._deconv_layer_params:
            if self._padding == "same":
                height *= strides
                width *= strides
            else:
                height = (height - 1) * strides + kernel_size
                width = (width - 1) * strides + kernel_size
        return (height, width, filters)

    def call(self, inputs):
        z = inputs
        for fc_l in self._preprocess_fc_layers:
            z = fc_l(z)
        z = tf.reshape(z, [-1] + self._start_decoding_shape)
        for deconv_l in self._deconv_layers:
            z = deconv_l(z)
        return z
