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
from alf.algorithms.agent import Agent
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.networks import QNetwork
from alf.optimizers import AdamTF
from alf.utils.dist_utils import calc_default_target_entropy

from alf.examples import ac_stochastic_with_risky_branch_conf
from alf.examples import sac_conf

CONV_LAYER_PARAMS = None
alf.config(
    'QNetwork', conv_layer_params=CONV_LAYER_PARAMS, fc_layer_params=(10, ))

alf.config(
    'SacAlgorithm',
    actor_network_cls=None,
    critic_network_cls=None,
    q_network_cls=QNetwork,
    actor_optimizer=AdamTF(lr=1e-3),
    critic_optimizer=AdamTF(lr=1e-3),
    alpha_optimizer=AdamTF(lr=1e-3),
    target_entropy=calc_default_target_entropy,
    target_update_tau=0.05,
    target_update_period=10)
alf.config('calc_default_target_entropy', min_prob=0.001)

gamma = 0.99
alf.config('OneStepTDLoss', gamma=gamma)

alf.config('Agent', rl_algorithm_cls=SacAlgorithm)

# training config
alf.config(
    'TrainerConfig',
    mini_batch_size=512,
    mini_batch_length=2,
    num_updates_per_train_iter=10,
    unroll_length=10,
    epsilon_greedy=0.05,
    initial_collect_steps=500,
    num_iterations=8000,
    num_env_steps=0,
    evaluate=True,
    num_eval_episodes=100,
    eval_interval=1000,
    num_checkpoints=1,
    debug_summaries=True,
    replay_buffer_length=200)
