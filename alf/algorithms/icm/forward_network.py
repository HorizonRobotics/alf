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
from tf_agents.networks import network, utils
from tf_agents.utils import nest_utils

import gin.tf


@gin.configurable
class ForwardNetwork(network.Network):
    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 action_fc_layer_params=None,
                 action_dropout_layer_params=None,
                 joint_fc_layer_params=None,
                 joint_dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 name='ForwardNetwork'):
        """Creates an instance of `ForwardNetwork`.

        Args:
            input_tensor_spec: A tuple of (feature, action) each a nest of
                `tensor_spec.TensorSpec` representing the inputs.
            output_tensor_spec: A nest of `tensor_spec.TensorSpec` representing
                the outputs.
            action_fc_layer_params: Optional list of fully connected parameters for
                actions, where each item is the number of units in the layer.
            action_dropout_layer_params: Optional list of dropout layer parameters
            joint_fc_layer_params: Optional list of fully connected parameters after
                merging observations and actions, where each item is the number of units
                in the layer.
            joint_dropout_layer_params: Optional list of dropout layer parameters
            activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
            name: A string representing name of the network.
        """

        super(ForwardNetwork, self).__init__(input_tensor_spec, (), name)
        feature_spec, action_spec = input_tensor_spec
        assert feature_spec == output_tensor_spec
        if len(tf.nest.flatten(feature_spec)) > 1:
            raise ValueError(
                'Only a single feature is supported by this network')
        if len(tf.nest.flatten(action_spec)) > 1:
            raise ValueError(
                'Only a single action is supported by this network')
        self._action_layers = utils.mlp_layers(
            None,
            action_fc_layer_params,
            action_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='action_encoding')
        self._joint_layers = utils.mlp_layers(
            None,
            joint_fc_layer_params,
            joint_dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='joint_mlp')
        self._joint_layers.append(
            tf.keras.layers.Dense(
                tf.nest.flatten(output_tensor_spec)[0].shape.num_elements(),
                activation=activation_fn,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.003, maxval=0.003),
                name='feature'))

        self._output_tensor_spec = output_tensor_spec

    def call(self, inputs, step_type=(), network_state=()):
        features, actions = inputs
        outer_rank = nest_utils.get_outer_rank(
            features,
            self.input_tensor_spec[0])
        if outer_rank == len(actions.shape):  #
            actions = tf.expand_dims(actions, -1)
        batch_squash = utils.BatchSquash(outer_rank)

        features = tf.cast(tf.nest.flatten(features)[0], tf.float32)
        features = batch_squash.flatten(features)
        actions = tf.cast(tf.nest.flatten(actions)[0], tf.float32)
        actions = batch_squash.flatten(actions)
        for layer in self._action_layers:
            actions = layer(actions)
        joint = tf.concat([features, actions], -1)
        for layer in self._joint_layers:
            joint = layer(joint)
        joint = batch_squash.unflatten(joint)
        return joint, network_state
