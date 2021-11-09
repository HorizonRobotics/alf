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
import functools
from alf.environments import suite_simple
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.agent import Agent
from alf.networks import ActorDistributionNetwork, CategoricalProjectionNetwork, ValueNetwork

alf.config(
    'create_environment',
    env_name='StochasticWithRiskyBranch',
    env_load_fn=suite_simple.load,
    num_parallel_environments=32)

hidden_layers = (10, )

# Neural Network Configuration
actor_network_cls = functools.partial(
    ActorDistributionNetwork, fc_layer_params=hidden_layers)
value_network_cls = functools.partial(
    ValueNetwork, fc_layer_params=hidden_layers)

alf.config('CategoricalProjectionNetwork', logits_init_output_factor=1e-10)

# Algorithm Configuration
alf.config(
    'ActorCriticLoss',
    entropy_regularization=0.01,
    use_gae=True,
    use_td_lambda_return=True,
    td_lambda=0.95,
    td_loss_weight=0.5,
    advantage_clip=None)

alf.config(
    'ActorCriticAlgorithm',
    actor_network_ctor=actor_network_cls,
    value_network_ctor=value_network_cls,
    optimizer=alf.optimizers.AdamTF(lr=1e-3))
alf.config('Agent', rl_algorithm_cls=ActorCriticAlgorithm)

alf.config(
    'TrainerConfig',
    data_transformer_ctor=None,
    mini_batch_size=512,
    mini_batch_length=2,
    num_updates_per_train_iter=10,
    unroll_length=10,
    algorithm_ctor=Agent,
    epsilon_greedy=0.05,
    num_iterations=8000,
    num_env_steps=0,
    evaluate=True,
    num_eval_episodes=100,
    eval_interval=1000,
    debug_summaries=1,
    summarize_grads_and_vars=1,
    summary_interval=10)
