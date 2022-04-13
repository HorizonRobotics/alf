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

import torch

import alf
from alf.algorithms.merlin_algorithm import ResnetEncodingNetwork
from alf.environments import suite_carla
from alf.environments.alf_wrappers import ActionObservationWrapper, ScalarRewardWrapper

alf.config(
    'suite_carla.load',
    wrappers=[ActionObservationWrapper, ScalarRewardWrapper])

alf.config('CameraSensor', image_size_x=192, image_size_y=96, fov=135)

alf.config('CollisionSensor', include_collision_location=True)

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
                output_activation=alf.math.identity,
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
