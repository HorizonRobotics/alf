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
import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork

from alf.utils import common
from alf.layers import get_identity_layer, get_first_element_layer


def attend_layer(input):
    """
    Generate the attention layer output

    Args:
        query: sentence input tensor of shape (batch, seq_length -- 20 from
            PlayGround.vocab_sequence_length), which will be transformed
            into an embedding of shape (batch, seq_length, num_embedding_dim), the same
            dimensionality as that of position embedding (value).
        key: the output of the conv_layers, which will be flattened into
            tensor of shape (batch, h * w, channels).
    """
    query = input[1]  # 'sentence'
    key = input[0]  # 'image'
    # reshape [batches, height, width, channels] tensor
    # into [batches, height * width, channels] tensor
    # Assumes channels_last.
    (b, h, w, c) = key.shape
    flatten_shape = (b, h * w, c)
    key_embeddings = tf.reshape(key, flatten_shape)

    # create position input tensor
    # np.asarray is to avoid ValueError: Argument must be a dense tensor: range(0, 400) - got shape [400], but wanted [].
    p = tf.constant(np.asarray(range(h * w)))
    p = tf.reshape(p, (h * w, 1))
    p = tf.keras.backend.repeat(p, n=b)
    p = tf.transpose(p)
    p = tf.reshape(p, flatten_shape[0:-1])

    # create position embedding (value) tensor
    pos_embeddings = tf.keras.layers.Embedding(flatten_shape[1], c)(
        p)  # p: b, h * w; output: b, h * w, c

    # inner product (Luong style) attention

    # c will be the number of embedding dimensions for the sentence input,
    # and num_embedding_dims parameter is ignored.  This is because we do
    # inner product of image and sentence embedding vectors to compute attention.
    query_embeddings = tf.keras.layers.Embedding(query.shape[1],
                                                 c)(query)  # seq_len, c

    # generates the position attention of shape (batch, h*w, c)
    pos_attention_seq = tf.keras.layers.Attention()(
        [query_embeddings, pos_embeddings, key_embeddings])

    img_attention_seq = tf.keras.layers.Attention()(
        [query_embeddings, key_embeddings])

    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(query_embeddings)
    pos_attention = tf.keras.layers.GlobalAveragePooling1D()(pos_attention_seq)
    img_attention = tf.keras.layers.GlobalAveragePooling1D()(img_attention_seq)
    return tf.keras.layers.Concatenate()(
        [query_encoding, pos_attention, img_attention])


@gin.configurable
def get_ac_networks(conv_layer_params=None,
                    attention=False,
                    activation_fn=tf.keras.activations.softsign,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    num_embedding_dims=None,
                    fc_layer_params=None,
                    num_state_tiles=None,
                    num_sentence_tiles=None,
                    name=None):
    """
    Generate the actor and value networks

    Args:
        conv_layer_params (list[int 3 tuple]): optional convolution layers
            parameters, where each item is a length-three tuple indicating
            (filters, kernel_size, stride).
        attention (bool): if true, use attention after conv layer.
            Assumes that sentence is part of the input dictionary.
        num_embedding_dims (int): optional number of dimensions of the
            vocabulary embedding space.
        fc_layer_params (list[int]): optional fully_connected parameters, where
            each item is the number of units in the layer.
        num_state_tiles (int): optional number of times to repeat the
            internal state tensor before concatenation with other inputs.
            The rationale is to match the number of dimentions of the image
            input, so that the final concatenation will have roughly equal
            representation from different sources of input.  Without this,
            typically image input, due to its large input size, will take over
            and trump all other small dimensional inputs.
        num_sentence_tiles (int): optional number of times to repeat the
            sentence embedding tensor before concatenation with other inputs,
            so that sentence input won't be trumped by other high dimensional
            inputs like image observation.
    """
    observation_spec = common.get_observation_spec()
    action_spec = common.get_action_spec()

    if attention:
        num_embedding_dims = conv_layer_params[-1][0]

    vocab_size = common.get_vocab_size()
    if vocab_size:
        sentence_layers = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, num_embedding_dims),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
        if num_sentence_tiles:
            sentence_layers.add(
                tf.keras.layers.Lambda(lambda x: tf.tile(
                    x, multiples=[1, num_sentence_tiles])))

    # image:
    conv_layers = []
    conv_layers.extend([
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation_fn,
            kernel_initializer=kernel_initializer,
            name='/'.join([name, 'conv2d']) if name else None)
        for (filters, kernel_size, strides) in conv_layer_params
    ])

    image_layers = conv_layers
    if not attention:
        image_layers.append(tf.keras.layers.Flatten())
    image_layers = tf.keras.Sequential(image_layers)

    preprocessing_layers = {}
    obs_spec = common.get_observation_spec()
    if isinstance(obs_spec, dict) and 'image' in obs_spec:
        preprocessing_layers['image'] = image_layers

    if isinstance(obs_spec, dict) and 'states' in obs_spec:
        state_layers = get_identity_layer()
        # [image: (1, 12800), sentence: (1, 16 * 800), states: (1, 16 * 800)]
        # Here, we tile along the last dimension of the input.
        if num_state_tiles:
            state_layers = tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: tf.tile(
                    x, multiples=[1, num_state_tiles]))
            ])
        preprocessing_layers['states'] = state_layers

    if isinstance(obs_spec, dict) and 'sentence' in obs_spec:
        preprocessing_layers['sentence'] = sentence_layers

    # This makes the code work with internal states input alone as well.
    if not preprocessing_layers:
        preprocessing_layers = get_identity_layer()

    preprocessing_combiner = tf.keras.layers.Concatenate()

    if attention:
        attention_combiner = tf.keras.layers.Lambda(lambda input: attend_layer(
            input))
        preprocessing_combiner = attention_combiner

    if not isinstance(preprocessing_layers, dict):
        preprocessing_combiner = None

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
                      activation_fn=tf.keras.activations.softsign,
                      kernel_initializer=tf.keras.initializers.GlorotUniform(),
                      num_embedding_dims=None,
                      fc_layer_params=None,
                      num_state_tiles=None,
                      num_sentence_tiles=None):
    """
    Generate the actor network
    """
    a, _ = get_ac_networks(
        conv_layer_params=conv_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        num_embedding_dims=num_embedding_dims,
        fc_layer_params=fc_layer_params,
        num_state_tiles=num_state_tiles,
        num_sentence_tiles=num_sentence_tiles,
        name="actor")
    return a


@gin.configurable
def get_value_network(conv_layer_params=None,
                      activation_fn=tf.keras.activations.softsign,
                      kernel_initializer=tf.keras.initializers.GlorotUniform(),
                      num_embedding_dims=None,
                      fc_layer_params=None,
                      num_state_tiles=None,
                      num_sentence_tiles=None):
    """
    Generate the value network
    """
    _, c = get_ac_networks(
        conv_layer_params=conv_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        num_embedding_dims=num_embedding_dims,
        fc_layer_params=fc_layer_params,
        num_state_tiles=num_state_tiles,
        num_sentence_tiles=num_sentence_tiles,
        name="value")
    return c
