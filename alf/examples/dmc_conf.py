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
"""A common configuration for DM control tasks independent of algortihms.
This file defines some basic experiment protocol (e.g., parallel envs, hidden
layers, learning rate, etc) to be shared by different algorithms to be evaluted.
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
    env_name="cheetah:run",
    num_parallel_environments=1,
    env_load_fn=suite_dmc.load)

alf.config(
    "suite_dmc.load",
    from_pixels=False,
    gym_env_wrappers=(partial(FrameSkip, skip=1), ),
    max_episode_steps=1000)

hidden_layers = (256, 256)

alf.config(
    "NormalProjectionParamNetwork",
    std_transform=partial(clipped_exp, clip_value_min=-4, clip_value_max=2))

alf.config("StableNormalProjectionParamNetwork", max_std=100)

actor_distribution_network_cls = partial(
    alf.networks.ActorDistributionNetwork, fc_layer_params=hidden_layers)

optimizer = AdamTF(lr=3e-4)

# alf.config('calc_default_target_entropy', min_prob=0.184)

alf.config(
    "TrainerConfig",
    temporally_independent_train_step=True,
    use_rollout_state=True,
    use_wandb=True,
    async_eval=True,
    entity="runjerry",
    project="Actor-Bayes-Critic",
    initial_collect_steps=10000,
    unroll_length=1,
    mini_batch_length=2,
    mini_batch_size=256,
    num_updates_per_train_iter=1,
    num_env_steps=int(1e6),
    num_iterations=0,
    num_checkpoints=1,
    evaluate=True,
    eval_interval=5000,
    num_eval_episodes=5,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    num_summaries=1000,
    replay_buffer_length=int(1e6))
