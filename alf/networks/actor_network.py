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
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common
from tf_agents.networks import encoding_network
from tf_agents.utils import nest_utils


@gin.configurable
class ActorNetwork(network.Network):
    """Creates an actor network."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 fc_layer_params=None,
                 dropout_layer_params=None,
                 conv_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 name='ActorNetwork'):
        """Creates an instance of `ActorNetwork`.

         This ActorNetwork supports handling complex observations with preprocessing_layer
         and preprocessing_combiner.

        Args:
            input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
                inputs.
            output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
                the outputs.
            preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
                representing preprocessing for the different observations.
                All of these layers must not be already built. For more details see
                the documentation of `networks.EncodingNetwork`.
            preprocessing_combiner: (Optional.) A keras layer that takes a flat list
                of tensors and combines them. Good options include
                `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
                This layer must not be already built. For more details see
                the documentation of `networks.EncodingNetwork`.
            fc_layer_params: Optional list of fully_connected parameters, where each
                item is the number of units in the layer.
            dropout_layer_params: Optional list of dropout layer parameters, each item
                is the fraction of input units to drop or a dictionary of parameters
                according to the keras.Dropout documentation. The additional parameter
                `permanent', if set to True, allows to apply dropout at inference for
                approximated Bayesian inference. The dropout layers are interleaved with
                the fully connected layers; there is a dropout layer after each fully
                connected layer, except if the entry in the list is None. This list must
                have the same length of fc_layer_params, or be None.
            conv_layer_params: Optional list of convolution layers parameters, where
                each item is a length-three tuple indicating (filters, kernel_size,
                stride).
            activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
            kernel_initializer: Initializer to use for the kernels of the conv and
                dense layers. If none is provided a default glorot_uniform
            name: A string representing name of the network.
        """
        super(ActorNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        self._encoder = encoding_network.EncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer)

        flat_action_spec = tf.nest.flatten(output_tensor_spec)
        action_layers = [
            tf.keras.layers.Dense(
                single_action_spec.shape.num_elements(),
                activation=tf.keras.activations.tanh,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.003, maxval=0.003),
                name='action') for single_action_spec in flat_action_spec
        ]

        self._output_tensor_spec = output_tensor_spec
        self._flat_action_spec = flat_action_spec
        self._action_layers = action_layers

    def call(self, observations, step_type=(), network_state=()):
        outer_rank = nest_utils.get_outer_rank(observations,
                                               self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)

        state, network_state = self._encoder(
            observations, step_type=step_type, network_state=network_state)

        state = batch_squash.flatten(state)

        actions = []
        for layer, spec in zip(self._action_layers, self._flat_action_spec):
            action = layer(state)
            action = common.scale_to_spec(action, spec)
            action = batch_squash.unflatten(action)
            actions.append(action)

        output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec,
                                                  actions)
        return output_actions, network_state
