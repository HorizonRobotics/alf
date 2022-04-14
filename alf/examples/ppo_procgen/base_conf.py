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
"""This is the configuration to train Procgen environment with PPO

Specifically, this configuration serves as an example that trains on
bossfight, but can be adapted to other Procgen games with slgiht
modification.

Note that the default mini batch size = 4096 currently consumes 23.5
GB of GPU memory. You will need at least an RTX 3090 for training.

Most of the hyper parameters are set with courtesy of OpenAI's Phasic
Policy Gradient implementation:
https://github.com/openai/phasic-policy-gradient

"""

import alf
from alf.examples import ppo_conf
from alf.examples import procgen_conf
from alf.examples.networks import impala_cnn_encoder

# Environment Configuration
alf.config('create_environment', num_parallel_environments=96)

# Construct the networks


def policy_network_ctor(input_tensor_spec, action_spec):
    encoder_output_size = 256
    return alf.nn.Sequential(
        impala_cnn_encoder.create(
            input_tensor_spec=input_tensor_spec,
            cnn_channel_list=(16, 32, 32),
            num_blocks_per_stack=2,
            flatten_output_size=encoder_output_size),
        alf.networks.CategoricalProjectionNetwork(
            input_size=encoder_output_size, action_spec=action_spec))


def value_network_ctor(input_tensor_spec):
    encoder_output_size = 256
    return alf.nn.Sequential(
        impala_cnn_encoder.create(
            input_tensor_spec=input_tensor_spec,
            cnn_channel_list=(16, 32, 32),
            num_blocks_per_stack=2,
            flatten_output_size=encoder_output_size),
        alf.layers.FC(input_size=encoder_output_size, output_size=1),
        alf.layers.Reshape(shape=()))


# Construct the algorithm

alf.config(
    'ActorCriticAlgorithm',
    actor_network_ctor=policy_network_ctor,
    value_network_ctor=value_network_ctor,
    optimizer=alf.optimizers.AdamTF(lr=5e-4))

alf.config(
    'PPOLoss',
    entropy_regularization=0.01,
    gamma=0.999,
    td_lambda=0.95,
    td_loss_weight=0.5)

# training config
alf.config(
    'TrainerConfig',
    unroll_length=256,
    # Setting mini_batch_length to None so that the algorithm will
    # implicitly make it equal to the length from the replay buffer.
    mini_batch_length=None,
    mini_batch_size=16,
    num_updates_per_train_iter=3,
    num_iterations=7500,
    num_checkpoints=5,
    evaluate=True,
    eval_interval=50,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=10)
