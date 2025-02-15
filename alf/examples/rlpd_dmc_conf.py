# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.algorithms.agent import Agent
from alf.algorithms.rlpd_algorithm import RlpdAlgorithm
from alf.examples.benchmarks.dm_control import dmc_conf

actor_network_cls = dmc_conf.actor_distribution_network_cls

critic_network_cls = partial(
    alf.networks.CriticNetwork,
    joint_fc_layer_params=dmc_conf.hidden_layers,
    use_fc_ln=True)  # turning on critic layernorm is crucial for high utd

alf.config(
    'Agent', optimizer=dmc_conf.optimizer, rl_algorithm_cls=RlpdAlgorithm)

alf.config(
    'RlpdAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    num_critic_replicas=10,
    num_sampled_critic_targets=1,  # should be 1 or 2 depending on tasks
    use_bootstrap_critics=False,  # turning to True might lead to larger variance
    bootstrap_mask_prob=0.8,
    actor_utd=3,
    critic_utd=
    10,  # the suggesting utd ratio between critic and actor is [2, 10]
    use_entropy_reward=True,
    target_update_tau=0.005)

alf.config('calc_default_target_entropy', min_prob=0.184)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    num_updates_per_train_iter=13,  # actor_utd + critic_utd
    summarize_gradient_noise_scale=False,
    summarize_action_distributions=False,
    random_seed=0)
