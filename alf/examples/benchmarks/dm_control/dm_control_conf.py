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
"""A common configuration for DM control tasks independent of algorithms.
This file defines some basic experiment protocol (e.g., parallel envs, hidden
layers, learning rate, etc) to be shared by different algorithms to be evaluated.
"""
import torch

from functools import partial

import alf
from alf.environments import suite_dmc
from alf.environments.gym_wrappers import FrameSkip
from alf.utils.math_ops import clipped_exp
from alf.algorithms.data_transformer import RewardNormalizer, ObservationNormalizer
from alf.optimizers import AdamTF

alf.config(
    "create_environment",
    num_parallel_environments=10,
    env_load_fn=suite_dmc.load)

alf.config(
    "suite_dmc.load",
    from_pixels=False,
    gym_env_wrappers=(partial(FrameSkip, skip=2), ),
    max_episode_steps=500)

hidden_layers = (256, 256)

actor_distribution_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    fc_layer_params=hidden_layers,
    continuous_projection_net_ctor=partial(
        alf.networks.NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=clipped_exp))

critic_network_cls = partial(
    alf.networks.CriticNetwork, joint_fc_layer_params=hidden_layers)

optimizer = AdamTF(lr=3e-4)

alf.config(
    "TrainerConfig",
    temporally_independent_train_step=True,
    use_rollout_state=True,
    initial_collect_steps=10000,
    unroll_length=1,
    mini_batch_length=6,
    mini_batch_size=256,
    num_updates_per_train_iter=1,
    num_env_steps=int(3e6),
    num_iterations=0,
    num_checkpoints=1,
    evaluate=True,
    num_evals=100,
    debug_summaries=True,
    summarize_grads_and_vars=0,
    num_summaries=100,
    replay_buffer_length=int(1e5))
