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
import numpy as np
from tf_agents.networks import network, utils
from tf_agents.utils import nest_utils
from tf_agents.specs import tensor_spec

import gin.tf


@gin.configurable
class InverseNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 joint_fc_layer_params=None,
                 joint_dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 name='InverseNetwork'):
        """Creates an instance of `InverseNetwork`.

        Args:
            input_tensor_spec: A tuple of (feature, next_feature) each a nest of
                `tensor_spec.TensorSpec` representing the inputs.
            output_tensor_spec: A nest of `tensor_spec.TensorSpec` representing
                the outputs.
            joint_fc_layer_params: Optional list of fully connected parameters after
                merging observations and actions, where each item is the number of units
                in the layer.
            joint_dropout_layer_params: Optional list of dropout layer parameters
            activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
            name: A string representing name of the network.
        """

        super(InverseNetwork, self).__init__(input_tensor_spec, (), name)
        feature_spec, next_feature_spec = input_tensor_spec
        assert feature_spec == next_feature_spec
        flat_action_spec = tf.nest.flatten(output_tensor_spec)
        if len(flat_action_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network')

        self._joint_layers = utils.mlp_layers(
            None,
            joint_fc_layer_params,
            joint_dropout_layer_params,
            activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='joint_mlp')

        output_spec = flat_action_spec[0]
        if tensor_spec.is_discrete(output_spec):
            unique_num_actions = np.unique(
                output_spec.maximum - output_spec.minimum + 1)
            output_shape = output_spec.shape.concatenate(
                [unique_num_actions])
            self._output_shape = output_shape
        else:
            self._output_shape = output_spec.shape

        self._projection_layer = tf.keras.layers.Dense(
            self._output_shape.num_elements(),
            kernel_initializer=tf.compat.v1.keras.initializers.
                VarianceScaling(scale=0.1),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='logits')
        self._output_tensor_spec = output_tensor_spec

    def call(self, inputs, step_type=(), network_state=()):
        features, next_features = inputs
        outer_rank = nest_utils.get_outer_rank(
            features,
            self.input_tensor_spec[0])
        batch_squash = utils.BatchSquash(outer_rank)
        features = tf.cast(tf.nest.flatten(features)[0], tf.float32)
        next_features = tf.cast(tf.nest.flatten(next_features)[0], tf.float32)
        features = batch_squash.flatten(features)
        next_features = batch_squash.flatten(next_features)
        joint = tf.concat([features, next_features], 1)
        for layer in self._joint_layers:
            joint = layer(joint)

        logits = self._projection_layer(joint)
        logits = tf.reshape(logits, [-1] + self._output_shape.as_list())
        logits = batch_squash.unflatten(logits)

        output_spec = tf.nest.flatten(self._output_tensor_spec)[0]
        if tensor_spec.is_discrete(output_spec):
            actions = tf.nn.softmax(logits)
        else:
            actions = tf.nn.tanh(logits)

        output_actions = tf.nest.pack_sequence_as(
            self._output_tensor_spec,
            [actions])
        return output_actions, network_state
