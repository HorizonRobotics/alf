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

from functools import partial

import torch

import alf
from alf.examples import sac_conf
from alf.algorithms.data_transformer import RewardNormalizer

alf.config(
    "create_environment",
    env_name="BipedalWalker-v2",
    num_parallel_environments=32)

hidden_layers = (256, ) * 2
proj_net = partial(
    alf.networks.NormalProjectionNetwork,
    state_dependent_std=True,
    scale_distribution=True,
    std_transform=alf.math.clipped_exp)
actor_distribution_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    fc_layer_params=hidden_layers,
    continuous_projection_net_ctor=proj_net)
critic_network_cls = partial(
    alf.networks.CriticNetwork, joint_fc_layer_params=hidden_layers)

optimizer = alf.optimizers.AdamTF(lr=5e-4)
alf.config("Agent", optimizer=optimizer)

alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_distribution_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.005)

alf.config('calc_default_target_entropy', min_prob=0.1)

reward_normalizer = partial(RewardNormalizer, clip_value=5.)

# training config
alf.config(
    "TrainerConfig",
    initial_collect_steps=3000,
    data_transformer_ctor=[reward_normalizer],
    mini_batch_length=2,
    unroll_length=4,
    mini_batch_size=4096,
    num_updates_per_train_iter=1,
    num_iterations=200000,
    num_checkpoints=5,
    evaluate=False,
    debug_summaries=True,
    summarize_gradient_noise_scale=True,
    summarize_grads_and_vars=True,
    num_summaries=100,
    replay_buffer_length=100000)
