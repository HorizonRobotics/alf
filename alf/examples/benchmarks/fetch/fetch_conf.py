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
"""A common Fetch task configuration independent of algorithms. This file
defines some basic experiment protocol (e.g., parallel envs, hidden layers,
learning rate, etc) to be shared by different algorithms to be evaluated.
"""

from functools import partial

import alf
from alf.algorithms.data_transformer import RewardNormalizer
from alf.environments import suite_robotics
from alf.optimizers import AdamTF
from alf.utils.math_ops import clipped_exp

alf.config(
    "create_environment",
    env_name="FetchPush-v1",
    env_load_fn=suite_robotics.load,
    num_parallel_environments=38)

hidden_layers = (256, ) * 3

actor_distribution_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    fc_layer_params=hidden_layers,
    continuous_projection_net_ctor=partial(
        alf.networks.NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=clipped_exp))

actor_network_cls = partial(
    alf.networks.ActorNetwork, fc_layer_params=hidden_layers)

critic_network_cls = partial(
    alf.networks.CriticNetwork, joint_fc_layer_params=hidden_layers)

alf.config("OneStepTDLoss", gamma=0.98)
alf.config("TDLoss", gamma=0.98)

optimizer = AdamTF(lr=1e-3)

alf.config(
    "TrainerConfig",
    temporally_independent_train_step=True,
    use_rollout_state=True,
    data_transformer_ctor=[partial(RewardNormalizer, clip_value=1.0)],
    initial_collect_steps=10000,
    unroll_length=50,
    mini_batch_length=2,
    mini_batch_size=4864,
    num_updates_per_train_iter=40,
    num_env_steps=int(1e7),
    num_iterations=0,
    num_checkpoints=5,
    evaluate=True,
    eval_interval=60,
    debug_summaries=True,
    summarize_grads_and_vars=0,
    summary_interval=30,
    replay_buffer_length=20000)
