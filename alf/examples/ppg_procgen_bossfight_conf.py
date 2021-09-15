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
from alf.examples import ppg_conf
from alf.examples import procgen_conf
from alf.algorithms.ppg_algorithm import PPGAlgorithm
from alf.networks.encoding_networks import EncodingNetwork
from alf.utils.losses import element_wise_huber_loss
from alf.examples.networks import impala_cnn_encoder

# Environment Configuration
alf.config(
    'create_environment', env_name='bossfight', num_parallel_environments=64)

# Reward Scailing
alf.config('PPGAuxPhaseLoss', policy_kl_loss_weight=0.005)


def encoding_network_ctor(input_tensor_spec, kernel_initializer):
    encoder_output_size = 256
    return impala_cnn_encoder.create(
        input_tensor_spec=input_tensor_spec,
        cnn_channel_list=(16, 32, 32),
        num_blocks_per_stack=2,
        output_size=encoder_output_size,
        kernel_initializer=kernel_initializer)


alf.config(
    'PPGAlgorithm',
    aux_phase_interval=None,
    encoding_network_ctor=encoding_network_ctor,
    policy_optimizer=alf.optimizers.AdamTF(lr=2e-4),
    aux_optimizer=alf.optimizers.AdamTF(lr=2e-4))

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
    mini_batch_length=1,
    mini_batch_size=4096,
    num_updates_per_train_iter=3,
    num_iterations=6000,
    num_checkpoints=5,
    evaluate=True,
    eval_interval=50,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=10)
