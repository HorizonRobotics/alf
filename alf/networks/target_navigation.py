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

from absl import logging
import collections
import functools
import gin
import torch

import alf
from alf.initializers import variance_scaling_init
from alf.networks.actor_distribution_networks import ActorDistributionNetwork, ActorDistributionRNNNetwork
from alf.networks.value_networks import ValueNetwork, ValueRNNNetwork
from alf.utils import common


class StateLanguageAttentionCombiner(torch.nn.Module):
    def __init__(self,
                 network_to_debug=None,
                 use_attention=False,
                 num_obj_dims=2,
                 name="",
                 **kwargs):
        """A combiner for modulating states input with language.
        """
        super().__init__(**kwargs)
        self._network_to_debug = network_to_debug
        self._num_obj_dims = num_obj_dims
        self._use_attention = use_attention
        self.name = name

    def forward(self, inputs):
        flat = alf.nest.flatten(inputs)
        if isinstance(flat[0], alf.TensorSpec):
            tensors = alf.nest.map_structure(
                lambda spec: spec.zeros(outer_dims=(1, )), inputs)
        else:
            tensors = inputs
        lang = tensors["sentence"]
        states = tensors["states"]
        (b, d) = states.shape

        if self._use_attention:
            # use transformation
            lang = lang.reshape(b, d, d)
            states = states.reshape(b, d, 1)
            outputs = torch.matmul(lang, states).reshape(b, d)
        else:
            # first object is always the goal, no need to use language
            outputs = states
        if isinstance(flat[0], alf.TensorSpec):
            return alf.TensorSpec.from_tensor(outputs, from_dim=1)
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
    sentence_layers = None
    if vocab_size:
        sentence_layers = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, num_embedding_dims),
            torch.nn.AdaptiveAvgPool1d(1))

    # image:
    if attention:
        image_layers = None
    elif isinstance(observation_spec, dict) and 'image' in observation_spec:
        image_layers = alf.networks.ImageEncodingNetwork(
            input_channels=None,
            input_size=None,
            conv_layer_params=conv_layer_params,
            flatten_output=True)

    preprocessing_layers = collections.OrderedDict()
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

    preprocessing_combiner = None

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
    elif (isinstance(preprocessing_layers, dict)
          and len(preprocessing_layers) > 1):
        preprocessing_combiner = alf.nest.utils.NestConcat()
    else:
        preprocessing_combiner = None

    if rnn:
        actor = ActorDistributionRNNNetwork(
            input_tensor_spec=observation_spec,
            action_spec=action_spec,
            input_preprocessors=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params)

        value = ValueRNNNetwork(
            input_tensor_spec=observation_spec,
            input_preprocessors=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params)
    else:
        actor = ActorDistributionNetwork(
            input_tensor_spec=observation_spec,
            action_spec=action_spec,
            input_preprocessors=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params)

        value = ValueNetwork(
            input_tensor_spec=observation_spec,
            input_preprocessors=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=fc_layer_params)

    return actor, value


@gin.configurable
def get_actor_network(conv_layer_params=None,
                      activation_fn=torch.nn.functional.softsign,
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
                      activation_fn=torch.nn.functional.softsign,
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
