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

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.oaec_algorithm import OaecAlgorithm
from alf.examples import dmc_conf
from alf.optimizers import AdamTF

actor_network_cls = dmc_conf.actor_distribution_network_cls

critic_network_cls = partial(
    alf.networks.CriticNetwork, joint_fc_layer_params=dmc_conf.hidden_layers)

alf.config('Agent', rl_algorithm_cls=OaecAlgorithm)

alf.config(
    'OaecAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    beta_ub=1.,
    beta_lb=.5,
    conservative_actor_training=False,
    conservative_critic_training=True,
    output_target_critic=True,
    std_for_explore='epi',
    std_for_overestimate='tot',
    opt_ptb_single_data=True,
    reward_noise_scale=None,
    num_rollout_sampled_actions=10,
    num_sampled_target_q_actions=0,
    use_common_target_q=True,
    target_q_from_sampled_actions='max',
    greedy_target_q_action=False,
    initial_uniform_rollout=False,
    num_bootstrap_critics=2,
    bootstrap_mask_prob=0.8,
    use_target_actor=True,
    actor_optimizer=AdamTF(lr=3e-4),
    critic_optimizer=AdamTF(lr=3e-4),
    target_update_tau=0.005)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    summarize_gradient_noise_scale=True,
    summarize_action_distributions=True,
    random_seed=0)
