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
from tf_agents.networks.encoding_network import EncodingNetwork as TFAEncodingNetwork


class EncodingNetwork(TFAEncodingNetwork):
    """Feed Forward network with CNN and FNN layers.."""

    def __init__(self,
                 input_tensor_spec,
                 last_layer_size,
                 last_activation_fn=None,
                 dtype=tf.float32,
                 last_kernel_initializer=None,
                 last_bias_initializer=tf.initializers.Zeros(),
                 preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
                 **xargs):
        """Create an EncodingNetwork

        This EncodingNetwork allows the last layer to have different setting
        from the other layers.

        Args:
            last_layer_size (int): size of the last layer
            last_activation_fn: Activation function of the last layer.
            last_kernel_initializer: Initializer for the kernel of the last
                layer. If none is provided a default
                tf.initializers.VarianceScaling is used.
            last_bias_initializer: initializer for the bias of the last layer.
            preprocessing_combiner: (Optional.) A keras layer that takes a flat
                list of tensors and combines them. Good options include
                `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
                This layer must not be already built. For more details see
                the documentation of `networks.EncodingNetwork`. If there is only
                one input, this will be ignored.
            xargs (dict): See tf_agents.networks.encoding_network.EncodingNetwork
              for detail
        """
        if len(tf.nest.flatten(input_tensor_spec)) == 1:
            preprocessing_combiner = None
        super(EncodingNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            preprocessing_combiner=preprocessing_combiner,
            dtype=dtype,
            **xargs)

        if not last_kernel_initializer:
            last_kernel_initializer = tf.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal')
        self._last_layer = tf.keras.layers.Dense(
            last_layer_size,
            activation=last_activation_fn,
            kernel_initializer=last_kernel_initializer,
            bias_initializer=last_bias_initializer,
            dtype=dtype)

    def call(self, observation, step_type=None, network_state=()):
        state, network_state = super(EncodingNetwork, self).call(
            observation, step_type=step_type, network_state=network_state)
        return self._last_layer(state), network_state
