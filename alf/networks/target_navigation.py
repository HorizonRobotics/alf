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
"""Actor and Value networks for target navigation task."""

import collections
import gin
import tensorflow as tf
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork

from alf.utils import common


def get_ac_networks(conv_layer_params=None,
                    num_embedding_dims=None,
                    fc_layer_params=None,
                    num_state_tiles=None,
                    num_sentence_tiles=None):
    """
    Generate the actor and value networks
    """
    observation_spec = common.get_observation_spec()
    states_shape = common.get_states_shape()
    action_spec = common.get_action_spec()
    vocab_size = common.get_vocab_size(
    )  # must have sentence as part of observation

    conv_layers = common.get_conv_layers(conv_layer_params)
    state_layers = common.get_identity_layer()
    sentence_layers = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, num_embedding_dims),
        tf.keras.layers.GlobalAveragePooling1D()
    ])
    # [image: (1, 12800), sentence: (1, 16 * 800), states: (1, 16 * 800)]
    # Here, we tile along the last dimension of the input.
    if num_state_tiles:
        state_layers = tf.keras.Sequential([
            state_layers,
            tf.keras.layers.Lambda(lambda x: tf.tile(
                x, multiples=[1, num_state_tiles]))
        ])
    if num_sentence_tiles:
        sentence_layers.add(
            tf.keras.layers.Lambda(lambda x: tf.tile(
                x, multiples=[1, num_sentence_tiles])))

    preprocessing_layers = {
        'image': conv_layers,
        'states': state_layers,
        'sentence': sentence_layers
    }

    preprocessing_combiner = tf.keras.layers.Concatenate()

    actor = ActorDistributionRnnNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        input_fc_layer_params=fc_layer_params)

    value = ValueRnnNetwork(
        input_tensor_spec=observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        input_fc_layer_params=fc_layer_params)

    return actor, value


@gin.configurable
def get_actor_network(conv_layer_params=None,
                      num_embedding_dims=None,
                      fc_layer_params=None,
                      num_state_tiles=None,
                      num_sentence_tiles=None):
    """
    Generate the actor network
    """
    a, _ = get_ac_networks(
        conv_layer_params=conv_layer_params,
        num_embedding_dims=num_embedding_dims,
        fc_layer_params=fc_layer_params,
        num_state_tiles=num_state_tiles,
        num_sentence_tiles=num_sentence_tiles)
    return a


@gin.configurable
def get_value_network(conv_layer_params=None,
                      num_embedding_dims=None,
                      fc_layer_params=None,
                      num_state_tiles=None,
                      num_sentence_tiles=None):
    """
    Generate the value network
    """
    _, c = get_ac_networks(
        conv_layer_params=conv_layer_params,
        num_embedding_dims=num_embedding_dims,
        fc_layer_params=fc_layer_params,
        num_state_tiles=num_state_tiles,
        num_sentence_tiles=num_sentence_tiles)
    return c
