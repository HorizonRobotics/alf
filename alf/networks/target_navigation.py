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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
import tensorflow as tf
import tf_agents
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork

from alf.algorithms.agent import get_obs
from alf.utils import common
from alf.layers import get_identity_layer, get_first_element_layer


# followed https://www.tensorflow.org/guide/keras/custom_layers_and_models
class ImageLanguageAttentionCombiner(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size,
                 n_heads=1,
                 network_to_debug=None,
                 output_attention_max=False,
                 use_attn_residual=False,
                 **kwargs):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self._n_heads = n_heads
        self._network_to_debug = network_to_debug
        self._output_attention_max = output_attention_max
        self._use_attn_residual = use_attn_residual
        self.fig = None

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        key_shape = input_shape[0]
        query_shape = input_shape[1]
        states_shape = 0
        if len(input_shape) > 2:
            states_shape = input_shape[2]
        (b, h, w, c) = key_shape  # channels last
        self._embedding = tf.keras.layers.Embedding(self._vocab_size, c)
        self._embedding.build(input_shape[1])

        self._attention = tf.keras.layers.Attention(use_scale=True)
        query_emb_shape = self._embedding.compute_output_shape(query_shape)
        key_value_shape = (b, h * w, c + c)
        key_emb_shape = (b, h * w, c)
        self._attention.build(
            input_shape=[query_emb_shape, key_value_shape, key_emb_shape])
        super().build(input_shape)

    # def compute_output_shape(self, input_shape):
    #     assert isinstance(input_shape, list)
    #     key_shape = input_shape[0]
    #     states_shape = 0
    #     if len(input_shape) > 2:
    #         states_shape = input_shape[2]
    #     b = key_shape[0]
    #     c = key_shape[-1]  # channels last
    #     # Output per batch contains query: c num_embedding_dims, image: c channels, position: 2 dims
    #     output_channels = 3 * c  # 2 * c + 2 if not tiling position tensor
    #     if self._output_attention_max:
    #         output_channels += c
    #     return (b, output_channels)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self._vocab_size,
            'n_heads': self._n_heads,
            'output_attention_max': self._output_attention_max,
            'use_attn_residual': self._use_attn_residual,
            'network_to_debug': self._network_to_debug
        })
        return config

    def call(self, inputs):
        """
        Generate the attention layer output

        Args:
            input: a list of two tensors:
                    image: the output of the conv_layers, which will be flattened
                        into tensor of shape (batch, h * w, channels),
                    sentence: integer encoded raw sentence input of shape (batch,
                        seq_length -- 20 from PlayGround.vocab_sequence_length),
                        which will be embedded into shape (batch, seq_length,
                        num_embedding_dim), where num_embedding_dim == channels
                        from the image input (key) in order to do inner product in
                        attention layer.
        """
        assert isinstance(inputs, list)
        key = inputs[0]
        query = inputs[1]
        states = None
        if len(inputs) > 2:
            states = inputs[2]

        # flatten image tensor [batches, height, width, channels]
        # into [batches, height * width, channels].
        # Assumes channels_last.
        (b, h, w, c) = key.shape
        flatten_shape = (b, h * w, c)
        key_embeddings = tf.reshape(key, flatten_shape)

        # create position input tensor
        x = tf.reshape(
            tf.constant(np.asarray(range(w)), dtype=tf.float32), (1, w))
        x = tf.reshape(tf.keras.backend.repeat(x, n=h), (h * w, 1))
        y = tf.reshape(
            tf.constant(np.asarray(range(h)), dtype=tf.float32), (1, h))
        y = tf.reshape(
            tf.transpose(tf.keras.backend.repeat(y, n=w)), (h * w, 1))
        p = tf.concat([x, y], -1)  # (h * w, 2)
        p = tf.keras.backend.repeat(tf.reshape(p, (h * w * 2, 1)), n=b)
        p = tf.transpose(p)
        p = tf.reshape(p,
                       (b, h * w, 2)) / w  # divide by w to normalize input pos

        # value: (b, h*w, 2 * c/2)
        pos_embeddings = tf.tile(p, multiples=[1, 1, c // 2])
        # inner product (Luong style) attention

        # c will be the number of embedding dimensions for the sentence input,
        # and num_embedding_dims parameter is ignored.  This is because we do
        # inner product of image and sentence embedding vectors to compute attention.
        # query_embeddings shape: (seq_len, c)
        query_embeddings = self._embedding(query)

        key_value = tf.concat([key_embeddings, pos_embeddings], axis=-1)
        n_heads = self._n_heads
        nbatch = b

        # Section of code adapted from https://github.com/google/trax/blob/master/trax/layers/attention.py
        # under http://www.apache.org/licenses/LICENSE-2.0

        # nbatch, seqlen, d_feature --> nbatch, n_heads, seqlen, d_head
        def SplitHeads(x, d_feature):
            assert d_feature % n_heads == 0
            d_head = d_feature // n_heads
            return tf.transpose(
                tf.reshape(x, (nbatch, -1, n_heads, d_head)), (0, 2, 1, 3))

        # nbatch, n_heads, seqlen, d_head --> nbatch, seqlen, d_feature
        def JoinHeads(x, d_feature):  # pylint: disable=invalid-name
            assert d_feature % n_heads == 0
            d_head = d_feature // n_heads
            return tf.reshape(
                tf.transpose(x, (0, 2, 1, 3)), (nbatch, -1, n_heads * d_head))

        # End section of code

        orig_query_embeddings = query_embeddings
        if n_heads > 1:
            query_embeddings = SplitHeads(query_embeddings, c)
            key_value = SplitHeads(key_value, 2 * c)
            key_embeddings = SplitHeads(key_embeddings, c)

        # print('query ', query_embeddings, '\nkey_val ', key_value, '\nkey ',
        #       key_embeddings)

        # generates the position attention of shape (batch, seq_len, c+c)  or c+2 if not tiling
        pos_attention_seq = self._attention(
            [query_embeddings, key_value, key_embeddings])

        if n_heads > 1:
            pos_attention_seq = JoinHeads(pos_attention_seq, 2 * c)

        attn_scores = tf.matmul(
            query_embeddings, key_embeddings, transpose_b=True)
        # print('attn_scores ', attn_scores)
        if n_heads > 1:
            attn_scores = tf.reduce_mean(attn_scores, axis=2)
        else:
            attn_scores = tf.keras.layers.GlobalAveragePooling1D()(
                attn_scores)  # across seq_len
        attn_softmax = tf.nn.softmax(attn_scores)
        attn_score_max = tf.reduce_max(attn_softmax, axis=-1)  # across h*w
        attn_residual = tf.multiply(
            tf.expand_dims(attn_softmax, -1), key_value)
        if n_heads > 1:
            attn_residual = JoinHeads(attn_residual, 2 * c)
        if tf.executing_eagerly():
            # shape (batch, )
            tf.summary.histogram(
                name=self.name + "/attention/value-max", data=attn_score_max)

            if self._network_to_debug == self.name:
                # take first batch
                attn = tf.reshape(attn_scores, (b, n_heads, h, w))[0]

                def tensor_to_image(attn, img):
                    # Default color palette: viridis (purple to yellow):
                    # https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html
                    if 0:  # Smaller image:
                        img = imresize(img, attn.shape, interp='bilinear')
                        attn = np.repeat(
                            np.reshape(attn, (h, w, 1)), 3, axis=-1)
                    else:  # Larger image:
                        attn = np.repeat(
                            np.reshape(attn, (h, w, 1)), 3, axis=-1)
                        attn = imresize(
                            attn * 255, img.shape, interp='nearest') / 255.
                        attn = np.clip(attn, 0, 1)
                    attn_img = (img * attn).astype(np.uint8)
                    return attn_img

                img = get_obs()['image'].numpy()  # will fail in graph mode
                img = img[
                    0, :, :,
                    -3:] * 255.  # take first batch, take last 3 channels to revert FrameStack, multiply by 255 to revert image scale wrapper
                img = np.clip(img, 0, 255)
                if n_heads > 1:
                    num_rows = 2
                else:
                    num_rows = 1
                for i in range(n_heads):
                    attn_img = tensor_to_image(attn[i].numpy(), img)
                    if self.fig is None:
                        plt.interactive(True)
                        _, axs = plt.subplots(num_rows, n_heads // num_rows)
                        self.fig = axs
                        plt.show()
                    if num_rows > 1:
                        self.fig[i % num_rows, i // num_rows].imshow(attn_img)
                    else:
                        self.fig.imshow(attn_img)
                # plt.pause(.00001)
                input()

        query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
            orig_query_embeddings)
        pos_attention = tf.keras.layers.GlobalAveragePooling1D()(
            pos_attention_seq)
        # tf.print(self.name, " pos: ", pos_attention[0][-4:], " attn_max", attn_score_max)

        outputs = [query_encoding, pos_attention]
        if self._output_attention_max:
            outputs.append(
                tf.reshape(
                    tf.keras.backend.repeat(
                        tf.reshape(attn_score_max, (b, n_heads)), n=c),
                    (b, c * n_heads)))
        if states is not None:
            outputs.append(states)
        if self._use_attn_residual:
            outputs.append(tf.reshape(attn_residual, [b, -1]))

        with tf.init_scope():
            if self.name == 'actor' and b > 1:
                print('attention_combiner outputs {} tensors'.format(
                    len(outputs)))
                shape_str = '  '
                for t in outputs:
                    shape_str += '  ' + str(t.shape)
                print(shape_str)
        return tf.keras.layers.Concatenate()(outputs)


class StateLanguageAttentionCombiner(tf.keras.layers.Layer):
    def __init__(self,
                 network_to_debug=None,
                 use_attention=False,
                 num_obj_dims=2,
                 **kwargs):
        super().__init__(**kwargs)
        self._network_to_debug = network_to_debug
        self._use_attention = use_attention
        self._num_obj_dims = num_obj_dims

    def get_config(self):
        config = super().get_config()
        config.update({
            'network_to_debug': self._network_to_debug,
            'use_attention': self._use_attention,
            'num_obj_dims': self._num_obj_dims,
        })
        return config

    def build(self, input_shape):
        if self._use_attention:
            assert isinstance(input_shape, list)
            query_shape = input_shape[0]
            states_shape = input_shape[1]
            (b, c) = query_shape
            (b, d) = states_shape
            n_objs = d // (1 + self._num_obj_dims)
            if self._num_obj_dims == 2:
                skip = 3  # first 3 objects are fake, encoding agent internal states
            else:  # == 3:
                skip = 2  # first 2 are internal states
            n_objs_real = n_objs - skip
            self._embedding = tf.keras.layers.Embedding(
                input_dim=n_objs,  # vocab_size
                output_dim=c)
            self._embedding.build((b, n_objs_real))
            self._attention = tf.keras.layers.Attention(use_scale=True)
            key_emb_shape = self._embedding.compute_output_shape((b,
                                                                  n_objs_real))
            value_shape = (b, n_objs_real, self._num_obj_dims)
            query_emb_shape = (b, 1, c)
            self._attention.build(
                input_shape=[query_emb_shape, value_shape, key_emb_shape])
        super().build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        lang = inputs[0]
        states = inputs[1]
        (b, d) = states.shape
        n_objs = d // (1 + self._num_obj_dims)
        states = tf.reshape(states, (b, n_objs, (1 + self._num_obj_dims)))
        if self._num_obj_dims == 2:
            skip = 3  # first 3 objects are fake, encoding agent internal states
        else:  # == 3:
            skip = 2  # first 2 objects are fake, encoding agent internal states
        keys = states[:, skip:, 0]  # object_ids
        key_embeddings = self._embedding(keys)
        values = states[:, skip:, 1:]  # skip fake objects and object_ids
        if self._use_attention:
            # Perform attention to extract coordinates based on self._num_obj_dims
            query_embeddings = tf.reshape(lang, (b, 1, -1))

            pos_attention = self._attention(
                [query_embeddings, values, key_embeddings])

            outputs = [
                lang,  # flattened query_embeddings
                tf.reshape(pos_attention, (b, -1)),
                tf.reshape(states[:, :, 1:], (b, -1))
            ]
            with tf.init_scope():
                if self.name == 'actor' and b > 1:
                    debug_str = (
                        'attention_combiner gets lang: {}, states: {}, ' +
                        'and outputs concat of {} tensors:\n').format(
                            inputs[0].shape, inputs[1].shape, len(outputs))
                    for t in outputs:
                        debug_str += " {}".format(t.shape)
                    print(debug_str)
                    print('key_embed: {}, query_embed: {}, values: {}'.format(
                        key_embeddings.shape, query_embeddings.shape,
                        values.shape))
            return tf.keras.layers.Concatenate()(outputs)
        else:  # use transformation
            lang = tf.reshape(lang, (b, d, d))
            states = tf.reshape(states, (b, d, 1))
            outputs = tf.reshape(tf.matmul(lang, states), (b, d))
            with tf.init_scope():
                if self.name == 'actor' and b > 1:
                    print(
                        'attention_combiner gets lang: {}, states: {}, and outputs {} tensor'
                        .format(inputs[0].shape, inputs[1].shape,
                                outputs.shape))
            return outputs


@gin.configurable
def get_ac_networks(conv_layer_params=None,
                    attention=False,
                    has_obj_id=False,
                    angle_limit=0,
                    activation_fn=tf.keras.activations.softsign,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    num_embedding_dims=None,
                    fc_layer_params=None,
                    num_state_tiles=None,
                    num_sentence_tiles=None,
                    name=None,
                    n_heads=1,
                    network_to_debug=None,
                    output_attention_max=False,
                    use_attn_residual=False,
                    rnn=True):
    """
    Generate the actor and value networks

    Args:
        conv_layer_params (list[int 3 tuple]): optional convolution layers
            parameters, where each item is a length-three tuple indicating
            (filters, kernel_size, stride).
        attention (bool): if true, use attention after conv layer.
            Assumes that sentence is part of the input dictionary.
        has_obj_id (bool): whether input has object id for id embedding.
        angle_limit (int): 0 if not restricting agent's viewing angle, and input
            coordinates per object will be 2-dimensional, otherwise, if positive
            inputs will be 3-dim per object.
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
    state_attn = (attention and isinstance(observation_spec, dict)
                  and 'image' not in observation_spec
                  and 'sentence' in observation_spec
                  and 'states' in observation_spec)
    if attention:
        if not state_attn:
            num_embedding_dims = conv_layer_params[-1][0]
        else:
            if has_obj_id:  # objects ordered left to right, indexed by id
                num_embedding_dims = observation_spec['sentence'].shape[-1]
            else:  # objects have fixed positions in the input list
                d = observation_spec['states'].shape[-1]
                num_embedding_dims = d * d

    vocab_size = common.get_vocab_size()
    if vocab_size:
        if not attention or state_attn:
            sentence_layers = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, num_embedding_dims),
                tf.keras.layers.GlobalAveragePooling1D()
            ])
            if num_sentence_tiles:
                sentence_layers.add(
                    tf.keras.layers.Lambda(lambda x: tf.tile(
                        x, multiples=[1, num_sentence_tiles])))
        else:  # image attention
            sentence_layers = get_identity_layer()

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
    obs_spec = observation_spec
    if isinstance(obs_spec, dict) and 'image' in obs_spec:
        preprocessing_layers['image'] = image_layers

    if isinstance(obs_spec, dict) and 'sentence' in obs_spec:
        preprocessing_layers['sentence'] = sentence_layers

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

    # This makes the code work with internal states input alone as well.
    if not preprocessing_layers:
        preprocessing_layers = get_identity_layer()

    preprocessing_combiner = tf.keras.layers.Concatenate()

    if attention:
        if state_attn:
            num_obj_dims = 2
            if angle_limit > 0:
                num_obj_dims = 3
            attention_combiner = StateLanguageAttentionCombiner(
                name=name,
                network_to_debug=network_to_debug,
                use_attention=has_obj_id,
                num_obj_dims=num_obj_dims)
        else:
            attention_combiner = ImageLanguageAttentionCombiner(
                vocab_size=vocab_size,
                n_heads=n_heads,
                name=name,
                output_attention_max=output_attention_max,
                use_attn_residual=use_attn_residual,
                network_to_debug=network_to_debug)
        preprocessing_combiner = attention_combiner

    if not isinstance(preprocessing_layers, dict):
        preprocessing_combiner = None

    if name == 'actor':
        print('==========================================')
        print('observation_spec: ', observation_spec)
        print('action_spec: ', action_spec)

    if rnn:
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
    else:
        actor = ActorDistributionNetwork(
            input_tensor_spec=observation_spec,
            output_tensor_spec=action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params)

        value = ValueNetwork(
            input_tensor_spec=observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params)

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
