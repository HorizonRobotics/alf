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

import functools

import alf
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.networks import ActorNetwork, CriticNetwork
from alf.optimizers import Adam
from alf.utils.losses import element_wise_huber_loss

# include default ddpg config
from alf.examples import ddpg_conf

# environment config
alf.config(
    'create_environment', env_name='Pendulum-v0', num_parallel_environments=1)

hidden_layers = (100, 100)
actor_network_cls = functools.partial(
    ActorNetwork, fc_layer_params=hidden_layers)

critic_network_cls = functools.partial(
    CriticNetwork, joint_fc_layer_params=hidden_layers)

critic_optimizer = Adam(lr=1e-3)
actor_optimizer = Adam(lr=1e-4)

alf.config(
    'DdpgAlgorithm',
    actor_network_ctor=actor_network_cls,
    critic_network_ctor=critic_network_cls,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    target_update_period=5)

alf.config('OneStepTDLoss', td_error_loss_fn=element_wise_huber_loss)

# training config
alf.config(
    'TrainerConfig',
    initial_collect_steps=1000,
    mini_batch_length=2,
    mini_batch_size=64,
    unroll_length=1,
    num_updates_per_train_iter=1,
    num_iterations=10000,
    num_checkpoints=5,
    evaluate=True,
    eval_interval=1000,
    debug_summaries=True,
    summarize_grads_and_vars=1,
    summarize_gradient_noise_scale=True,
    summary_interval=100,
    replay_buffer_length=100000)
