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
import functools
import gin
import torch

import alf
from alf.initializers import variance_scaling_init
from alf.networks.actor_distribution_networks import ActorDistributionRNNNetwork
from alf.networks.value_networks import ValueRNNNetwork
from alf.utils import common


class StateLanguageAttentionCombiner(torch.nn.Module):
    def __init__(self,
                 network_to_debug=None,
                 use_attention=False,
                 num_obj_dims=2,
                 **kwargs):
        super().__init__(**kwargs)
        self._network_to_debug = network_to_debug
        self._num_obj_dims = num_obj_dims
        self._first_call = True

    def forward(self, inputs):
        assert isinstance(inputs, list)
        lang = inputs[0]
        states = inputs[1]
        (b, d) = states.shape
        n_objs = d // (1 + self._num_obj_dims)
        states = states.reshape(b, n_objs, (1 + self._num_obj_dims))

        if self._num_obj_dims == 3:
            # use transformation
            lang = lang.reshape(b, d, d)
            states = states.reshape(b, d, 1)
            outputs = torch.matmul(lang, states).reshape(b, d)
        else:
            # first object is always the goal, no need to use language
            outputs = states
        if self._first_call:
            if self.name == 'actor':
                print(
                    'attention_combiner gets lang: {}, states: {}, and outputs {} tensor'
                    .format(inputs[0].shape, inputs[1].shape, outputs.shape))
            self._first_call = False
        return outputs


@gin.configurable
def get_ac_networks(conv_layer_params=None,
                    attention=False,
                    has_obj_id=False,
                    angle_limit=0,
                    activation_fn=torch.nn.functional.softsign,
                    kernel_initializer=None,
                    num_embedding_dims=None,
                    fc_layer_params=None,
                    name=None,
                    n_heads=1,
                    gft=0,
                    network_to_debug=None,
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
    """
    observation_spec = common.get_observation_spec()
    action_spec = common.get_action_spec()
    if kernel_initializer is None:
        kernel_initializer = functools.partial(
            variance_scaling_init,
            mode='fan_in',
            distribution='truncated_normal',
            nonlinearity=activation_fn)

    if attention:
        assert (isinstance(observation_spec, dict)
                and 'image' not in observation_spec
                and 'sentence' in observation_spec
                and 'states' in observation_spec
                ), "state and sentence obs are required for attention."
        if has_obj_id:  # objects ordered left to right, indexed by id
            num_embedding_dims = observation_spec['sentence'].shape[-1]
        else:  # objects have fixed positions in the input list
            d = observation_spec['states'].shape[-1]
            num_embedding_dims = d * d

    vocab_size = common.get_vocab_size()
    if vocab_size:
        sentence_layers = torch.nn.Sequential([
            torch.nn.Embedding(vocab_size, num_embedding_dims),
            torch.nn.AvgPool1d()
        ])

    # image:
    if attention:
        assert False
    else:
        image_layers = alf.networks.ImageEncodingNetwork(flatten_output=True)

    preprocessing_layers = {}
    obs_spec = observation_spec
    if isinstance(obs_spec, dict) and 'image' in obs_spec:
        preprocessing_layers['image'] = image_layers

    if isinstance(obs_spec, dict) and 'sentence' in obs_spec:
        preprocessing_layers['sentence'] = sentence_layers

    if isinstance(obs_spec, dict) and 'states' in obs_spec:
        preprocessing_layers['states'] = torch.nn.Identity()

    # This makes the code work with internal states input alone as well.
    if not preprocessing_layers:
        preprocessing_layers = torch.nn.Identity()

    preprocessing_combiner = torch.nn.Concat(1)

    if attention:
        num_obj_dims = 2
        if angle_limit > 0:
            num_obj_dims = 3
        attention_combiner = StateLanguageAttentionCombiner(
            name=name,
            network_to_debug=network_to_debug,
            use_attention=has_obj_id,
            num_obj_dims=num_obj_dims)
        preprocessing_combiner = attention_combiner

    if not isinstance(preprocessing_layers, dict):
        preprocessing_combiner = None

    if name == 'actor':
        print('==========================================')
        print('observation_spec: ', observation_spec)
        print('action_spec: ', action_spec)

    if rnn:
        actor = ActorDistributionRNNNetwork(
            input_tensor_spec=observation_spec,
            output_tensor_spec=action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            input_fc_layer_params=fc_layer_params)

        value = ValueRNNNetwork(
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
                      kernel_initializer=None,
                      num_embedding_dims=None,
                      fc_layer_params=None):
    """
    Generate the actor network
    """
    a, _ = get_ac_networks(
        conv_layer_params=conv_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        num_embedding_dims=num_embedding_dims,
        fc_layer_params=fc_layer_params,
        name="actor")
    return a


@gin.configurable
def get_value_network(conv_layer_params=None,
                      activation_fn=tf.keras.activations.softsign,
                      kernel_initializer=None,
                      num_embedding_dims=None,
                      fc_layer_params=None):
    """
    Generate the value network
    """
    _, c = get_ac_networks(
        conv_layer_params=conv_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        num_embedding_dims=num_embedding_dims,
        fc_layer_params=fc_layer_params,
        name="value")
    return c
