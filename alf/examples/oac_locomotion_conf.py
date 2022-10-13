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

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.oac_algorithm import OacAlgorithm
from alf.examples.benchmarks.locomotion import locomotion_conf
from alf.optimizers import AdamTF

alf.config('Agent', rl_algorithm_cls=OacAlgorithm)
# optimizer=locomotion_conf.optimizer)

alf.config(
    'OacAlgorithm',
    actor_network_cls=locomotion_conf.actor_distribution_network_cls,
    critic_network_cls=locomotion_conf.critic_network_cls,
    num_critic_replicas=2,
    explore=True,
    beta_ub=4.66,
    beta_lb=1.,
    explore_delta=6.86,
    reproduce_locomotion=True,
    actor_optimizer=AdamTF(lr=3e-4),
    critic_optimizer=AdamTF(lr=3e-4),
    alpha_optimizer=AdamTF(lr=3e-4),
    target_update_tau=0.005)

alf.config('calc_default_target_entropy', min_prob=0.184)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    summarize_gradient_noise_scale=True,
    summarize_action_distributions=True)
