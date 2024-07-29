# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from absl import logging
import torch

import alf
from alf.algorithms.merlin_algorithm import ResnetEncodingNetwork
from alf.environments import suite_carla
from alf.environments.alf_wrappers import ActionObservationWrapper, ScalarRewardWrapper
from alf.networks import SocialAttentionNetwork

alf.config(
    'suite_carla.load',
    wrappers=[ActionObservationWrapper, ScalarRewardWrapper])

alf.config('CameraSensor', image_size_x=192, image_size_y=96, fov=135)

alf.config('CollisionSensor', include_collision_location=True)

history_idx = [-16, -11, -6, -1]
alf.config(
    'DynamicObjectSensor',
    with_ego_history=True,
    history_idx=history_idx,
    max_object_number=6)

alf.config(
    'create_environment',
    env_name='Town01',
    env_load_fn=suite_carla.load,
    num_parallel_environments=4)


def create_input_preprocessors(encoding_dim, use_bn=False, preproc_bn=False):
    alf.config('BottleneckBlock', with_batch_normalization=use_bn)

    observation_spec = alf.get_observation_spec()
    prev_action_spec = observation_spec['prev_action']
    observation_preprocessors = {}

    def _make_simple_preproc(spec, use_bias=False):
        return torch.nn.Sequential(
            alf.layers.Reshape([-1]),
            alf.layers.FC(
                spec.numel, encoding_dim, use_bias=use_bias,
                use_bn=preproc_bn))

    for sensor, spec in observation_spec['observation'].items():
        if sensor == 'camera':
            observation_preprocessors[sensor] = ResnetEncodingNetwork(
                input_tensor_spec=spec,
                output_size=encoding_dim,
                output_activation=alf.math.identity)
        elif sensor == "dynamic_object":
            observation_preprocessors[sensor] = SocialAttentionNetwork(
                input_tensor_spec=spec,
                input_preprocessors=torch.nn.Sequential(
                    alf.layers.Permute(1, 0, 2, 3),
                    alf.layers.Reshape([len(history_idx), -1]),
                ),
                fc_layer_params=(encoding_dim, ),
                activation=alf.math.identity,
                num_of_heads=4,
                last_layer_size=encoding_dim,
                last_activation=alf.math.identity,
                use_fc_bn=preproc_bn)

        else:
            observation_preprocessors[sensor] = _make_simple_preproc(spec)

    return {
        'observation':
            observation_preprocessors,
        'prev_action':
            _make_simple_preproc(
                prev_action_spec,
                use_bias='camera' not in observation_spec['observation']),
    }


def create_input_preprocessor_masks():
    # create input mask to flexibly mask out some sensors
    observation_spec = alf.get_observation_spec()
    observation_preprocessor_masks = {}
    for sensor, spec in observation_spec['observation'].items():
        if sensor in [
                'gnss',
                'goal',
                'radar',
        ]:  # mask selected seneors out of observation for training
            logging.info("-----skip CARLA input: {}-----".format(sensor))
            observation_preprocessor_masks[sensor] = 0
        else:
            observation_preprocessor_masks[sensor] = 1

    return {'observation': observation_preprocessor_masks, 'prev_action': 1}
