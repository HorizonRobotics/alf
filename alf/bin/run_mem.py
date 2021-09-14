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

import alf
import torch
import gym3
import time
from procgen import ProcgenGym3Env
from alf.examples import procgen_conf
from alf.algorithms.data_transformer import create_data_transformer
from alf.algorithms.data_transformer import ImageScaleTransformer

from alf.algorithms.ppg import DisjointPolicyValueNetwork
from alf.examples.networks import impala_cnn_encoder


def encoding_network_ctor(input_tensor_spec, kernel_initializer):
    encoder_output_size = 256
    return impala_cnn_encoder.create(
        input_tensor_spec=input_tensor_spec,
        cnn_channel_list=(16, 32, 32),
        num_blocks_per_stack=2,
        output_size=encoder_output_size,
        kernel_initializer=kernel_initializer)


def run_wrapped(num_envs=64,
                unroll_length=256,
                run_network=True,
                batch_size=2048):
    alf.config(
        'create_environment',
        env_name='bossfight',
        num_parallel_environments=num_envs)
    env = alf.get_env()
    data_transformer = create_data_transformer([ImageScaleTransformer],
                                               env.observation_spec())

    observation_spec = data_transformer.transformed_observation_spec
    print(f'Transformed observation spec: {observation_spec}')

    if run_network:
        dual_actor_value_network = DisjointPolicyValueNetwork(
            observation_spec=observation_spec,
            action_spec=env.action_spec(),
            encoding_network_ctor=encoding_network_ctor,
            is_sharing_encoder=False)

    step = 0
    replay_buffer = []
    while True:
        actions = env.action_spec().ones(outer_dims=(num_envs, ))
        time_step = env.step(actions)
        time_step, _ = data_transformer.transform_timestep(time_step, ())
        replay_buffer.append(time_step)
        step += 1

        if step % unroll_length == 0:
            print(
                f'step: {step}, replay_buffer size: {len(replay_buffer)}, shape: {replay_buffer[0].observation.shape}, dtype={replay_buffer[0].observation.dtype}'
            )

            result_buffer = []
            if run_network:
                concat_length = batch_size // num_envs
                observation = torch.cat([
                    sample.observation
                    for sample in replay_buffer[:concat_length]
                ],
                                        dim=0)
                (action_distribution, value,
                 aux), state = dual_actor_value_network(
                     observation, state=())
                result_buffer.append((action_distribution, value, aux))

            replay_buffer = []
            time.sleep(1)


if __name__ == '__main__':
    if torch.cuda.is_available():
        alf.set_default_device('cuda')
    run_wrapped()
