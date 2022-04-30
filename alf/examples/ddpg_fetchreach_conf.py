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
from alf.algorithms.agent import Agent
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.environments import suite_robotics
from alf.networks import ActorNetwork, CriticNetwork
from alf.optimizers import AdamTF

from alf.examples import ddpg_conf


def define_config(name, default_value):
    alf.define_config(name, default_value)
    return alf.get_config_value('_CONFIG._USER.' + name)


alf.config(
    'create_environment',
    env_load_fn=suite_robotics.load,
    env_name='FetchReach-v1',
    num_parallel_environments=38)

gamma = 0.98
alf.config('OneStepTDLoss', gamma=gamma)
# The gamma in ReplayBuffer is only used to compute future discounted return,
# to be used e.g. as lower bounds for value training.
alf.config('ReplayBuffer', gamma=gamma)

hidden_layers = (256, 256, 256)
actor_network_cls = functools.partial(
    ActorNetwork, fc_layer_params=hidden_layers)

critic_network_cls = functools.partial(
    CriticNetwork, joint_fc_layer_params=hidden_layers)

lr = define_config('lr', 1e-3)
critic_optimizer = AdamTF(lr=lr)
actor_optimizer = AdamTF(lr=lr)

alf.config(
    'DdpgAlgorithm',
    actor_network_ctor=actor_network_cls,
    critic_network_ctor=critic_network_cls,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    rollout_random_action=0.3,
    target_update_period=40)

alf.config('Agent', rl_algorithm_cls=DdpgAlgorithm)
alf.config('ObservationNormalizer', clipping=5.)

alf.config(
    'TrainerConfig',
    data_transformer_ctor=ObservationNormalizer,
    algorithm_ctor=Agent,
    initial_collect_steps=10000,
    mini_batch_length=2,
    mini_batch_size=5000,
    unroll_length=50,
    num_updates_per_train_iter=40,
    num_iterations=0,
    num_env_steps=5000000,
    num_checkpoints=10,
    evaluate=True,
    eval_interval=40,
    num_eval_episodes=200,
    debug_summaries=True,
    summarize_grads_and_vars=1,
    summary_interval=20,
    use_rollout_state=True,
    replay_buffer_length=50000)
