# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from rlbench.observation_config import ObservationConfig, CameraConfig
import numpy as np
import gym
from functools import partial

from itertools import groupby

import torch

import alf
from alf.examples.networks import impala_cnn_encoder
from alf.algorithms.merlin_algorithm import ResnetEncodingNetwork
from alf.algorithms.encoding_algorithm import EncodingAlgorithm
from alf.networks import EncodingNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.environments import suite_rlbench


def _get_obs_config():
    off_camera_config = CameraConfig()
    off_camera_config.set_all(False)
    on_camera_config = CameraConfig(image_size=(64, 64))
    on_camera_config.set_all(False)
    on_camera_config.rgb = True
    on_camera_config.depth = False
    #on_camera_config.depth_in_meters = False
    on_camera_config.point_cloud = True
    # We use the front and wrist cameras for image inputs
    obs_config = ObservationConfig(
        left_shoulder_camera=on_camera_config,
        right_shoulder_camera=on_camera_config,
        overhead_camera=off_camera_config,
        wrist_camera=off_camera_config,
        front_camera=on_camera_config)
    obs_config.set_all_low_dim(True)
    obs_config.task_low_dim_state = False
    return obs_config


alf.config(
    "suite_rlbench.load",
    max_episode_steps=500,
    task_variation=False,
    observation_config=_get_obs_config())


def get_sensors(name):
    obs_spec = alf.get_raw_observation_spec()
    return [k for k in obs_spec.keys() if name in k]


def _create_input_preprocessors(encoding_dim, input_spec):
    def _make_simple_preprocessor(spec):
        return EncodingNetwork(
            input_tensor_spec=spec,
            last_layer_size=encoding_dim,
            last_activation=alf.math.identity)

    def _make_cnn_preprocessor(spec, type='cnn'):
        if type == 'resnet':
            return ResnetEncodingNetwork(
                input_tensor_spec=spec,
                output_size=encoding_dim,
                output_activation=alf.math.identity)
        elif type == 'impala_cnn':
            return impala_cnn_encoder.create(
                input_tensor_spec=spec,
                cnn_channel_list=(16, 32, 32),
                num_blocks_per_stack=2,
                output_size=encoding_dim)
        else:
            return EncodingNetwork(
                input_tensor_spec=spec,
                conv_layer_params=((32, 8, 4), (64, 4, 2), (64, 3, 1)),
                last_layer_size=encoding_dim,
                last_activation=alf.math.identity)

    def _make_preprocessor(sensor, spec):
        if "command" in sensor:
            assert spec.is_discrete
            # We use a stateful LSTMCell to encode a char at every step
            return alf.nn.Sequential(
                EmbeddingPreprocessor(spec, encoding_dim),
                alf.networks.LSTMCell(encoding_dim, encoding_dim))
        elif spec.ndim >= 3:
            return _make_cnn_preprocessor(spec)
        else:
            return _make_simple_preprocessor(spec)

    preprocessors = alf.nest.py_map_structure_with_path(
        _make_preprocessor, input_spec)
    return preprocessors


def _group_image_tensors(observation):
    """Group together all image tensors of each camera. Tensors from the same
    camera have the same prefix in the their names. For example, if the observation
    contains keys ``["wrist_rgb", "wrist_depth", "wrist_point_cloud", "state"]``, then
    after grouping the output will be a dict
    ``{"wrist": ("wrist_rgb", "wrist_depth", "wrist_point_cloud"),
       "state": ("state",)}``, with each group being a tuple.
    """

    def _find_camera(k):
        cameras = [
            'wrist', 'left_shoulder', 'right_shoulder', 'front', 'overhead'
        ]
        for c in cameras:
            if k.startswith(c):
                return c
        return k

    # ``observation`` can be either a dict of arrays or a ``gym.spaces.Dict``
    keys = observation.keys()
    # group together sensors that have the same prefix
    grouped_keys = groupby(keys, _find_camera)
    obs = dict()
    for name, g in grouped_keys:
        obs[name] = tuple([observation[k] for k in g])
    return obs


def create_vision_encoder(input_tensor_spec, preproc_encoding_dim,
                          fc_layer_params, activation):

    group_net = alf.networks.NetworkWrapper(_group_image_tensors,
                                            input_tensor_spec)

    concat_channels = alf.nest.utils.NestConcat(dim=0)
    modules = {k: concat_channels for k in group_net.output_spec.keys()}
    # Because depth/point cloud maps are float while rgb images are uint8, we can't
    # concat their channels in a gym wrapper. So we do it on the network side.
    concat_net = alf.nn.Parallel(modules, group_net.output_spec)

    return alf.nn.Sequential(
        group_net, concat_net,
        EncodingNetwork(
            input_tensor_spec=concat_net.output_spec,
            input_preprocessors=_create_input_preprocessors(
                preproc_encoding_dim, concat_net.output_spec),
            preprocessing_combiner=alf.layers.NestSum(
                activation=activation, average=True),
            fc_layer_params=fc_layer_params,
            activation=activation))


def create_encoding_algorithm_ctor(preproc_encoding_dim, fc_layer_params,
                                   activation):
    return partial(
        EncodingAlgorithm,
        encoder_cls=partial(
            create_vision_encoder,
            preproc_encoding_dim=preproc_encoding_dim,
            fc_layer_params=fc_layer_params,
            activation=activation))
