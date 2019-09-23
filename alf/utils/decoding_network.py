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

import gin.tf
import tensorflow as tf


@gin.configurable
class DecodingNetwork(tf.keras.models.Model):
    """
    A general template class for create transposed convolutional decoding networks.
    """

    def __init__(self,
                 start_decoding_size=7,
                 start_decoding_filters=8,
                 padding="same",
                 preprocess_fc_layer_params=None,
                 deconv_layer_params=None,
                 activation_fn=tf.nn.relu,
                 output_activation_fn=tf.nn.tanh):
        """
        Initialize the layers for decoding a latent vector into an image.

        Args:
            start_decoding_size (int): the initial size we'd like to have for
                the feature map
            start_decoding_filters (int): the initial number of fitlers we'd like
                to have for the feature map. Note that given this value and
                `start_decoding_size`, we always first project an input latent
                vector into a vector of an appropriate length so that it can be
                reshaped into (`start_decoding_size`, `start_decoding_size`,
                `start_decoding_filters`).
            padding (str): "same" or "valid", see tf.keras.layers.Conv2DTranspose
            preprocess_fc_layer_params (tuple[int]): a list of fc layer units.
                These fc layers are used for preprocessing the latent vector before
                transposed convolutions.
            deconv_layer_params (list[tuple]): a list of elements (num_filters,
                kernel_size, strides)
            activation_fn (tf.nn.activation): activation for hidden layers
            output_activation_fn (tf.nn.activation): activation for the output
                layer. Usually our image inputs are normalized to [0, 1] or [-1, 1],
                so this function should be tf.nn.sigmoid or tf.nn.tanh.

        The user is responsible for calculating the output size given
        `start_decoding_size` and `deconv_layer_params`, and make sure that the
        size will match the expectation. How to calculate the output size:
            if padding=="same", then H = H1 * strides
            if padding=="valid", then H = (H1-1) * strides + HF
        where H = output size, H1 = input size, HF = height of kernel
        """
        super().__init__()

        self._preprocess_fc_layers = []
        if preprocess_fc_layer_params:
            for size in preprocess_fc_layer_params:
                self._preprocess_fc_layers.append(
                    tf.keras.layers.Dense(size, activation=activation_fn))

        # We always assume "channels_last" !
        self._start_decoding_shape = [
            start_decoding_size, start_decoding_size, start_decoding_filters
        ]

        self._preprocess_fc_layers.append(
            tf.keras.layers.Dense(
                tf.reduce_prod(self._start_decoding_shape),
                activation=activation_fn
                if deconv_layer_params else output_activation_fn))

        self._deconv_layers = []
        if deconv_layer_params:
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

    def call(self, inputs):
        z = inputs
        for fc_l in self._preprocess_fc_layers:
            z = fc_l(z)
        z = tf.reshape(z, [-1] + self._start_decoding_shape)
        for deconv_l in self._deconv_layers:
            z = deconv_l(z)
        return z
