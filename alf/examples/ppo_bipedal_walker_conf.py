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

from functools import partial

import alf
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.algorithms.data_transformer import RewardNormalizer
from alf.algorithms.agent import Agent
from alf.algorithms.entropy_target_algorithm import SGDEntropyTargetAlgorithm
from alf.networks.projection_networks import BetaProjectionNetwork

import torch
# algorithm config
fc_layer_params = (128, ) * 4
env_name = "BipedalWalker-v2"
num_iterations = 200000

alf.config(
    'create_environment', env_name=env_name, num_parallel_environments=64)

alf.config(
    'ActorDistributionNetwork',
    fc_layer_params=fc_layer_params,
    continuous_projection_net_ctor=partial(
        BetaProjectionNetwork, min_concentration=1.))
alf.config('ValueNetwork', fc_layer_params=fc_layer_params)

# reward scaling
alf.config(
    'TrainerConfig',
    data_transformer_ctor=partial(
        RewardNormalizer, update_mode="rollout", clip_value=5.),
    algorithm_ctor=Agent,
    epsilon_greedy=0.1,
    evaluate=True,
    num_evals=200,
    unroll_length=8,
    mini_batch_length=1,
    mini_batch_size=256,
    num_updates_per_train_iter=4,
    num_iterations=num_iterations,
    num_checkpoints=1,
    debug_summaries=True,
    summarize_grads_and_vars=False,
    num_summaries=200)

alf.config("calc_default_target_entropy", min_prob=0.1)

alf.config(
    'Agent',
    optimizer=alf.optimizers.Adam(lr=3e-4),
    rl_algorithm_cls=partial(PPOAlgorithm, loss_class=PPOLoss),
    enforce_entropy_target=True,
    entropy_target_cls=SGDEntropyTargetAlgorithm)
