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
"""A common Locomotion task configuration independent of algorithms. This file
defines some basic experiment protocol (e.g., parallel envs, hidden layers,
learning rate, etc) to be shared by different algorithms to be evaluated.
"""

from functools import partial

import alf
from alf.utils.math_ops import clipped_exp
from alf.optimizers import AdamTF

alf.config(
    "create_environment", num_parallel_environments=1, env_name="Ant-v3")

hidden_layers = (256, ) * 2

alf.config(
    "NormalProjectionNetwork",
    state_dependent_std=True,
    scale_distribution=True,
    std_transform=partial(clipped_exp, clip_value_min=-10, clip_value_max=2))

actor_distribution_network_cls = partial(
    alf.networks.ActorDistributionNetwork, fc_layer_params=hidden_layers)

critic_network_cls = partial(
    alf.networks.CriticNetwork, joint_fc_layer_params=hidden_layers)

optimizer = AdamTF(lr=3e-4)

alf.config(
    "TrainerConfig",
    temporally_independent_train_step=True,
    use_rollout_state=True,
    initial_collect_steps=50000,
    unroll_length=1,
    mini_batch_length=2,
    mini_batch_size=256,
    num_updates_per_train_iter=1,
    num_env_steps=int(3e6),
    num_iterations=0,
    num_checkpoints=5,
    evaluate=True,
    eval_interval=20000,
    num_eval_episodes=20,
    debug_summaries=True,
    summarize_grads_and_vars=0,
    num_summaries=1000,
    replay_buffer_length=int(1e6))
