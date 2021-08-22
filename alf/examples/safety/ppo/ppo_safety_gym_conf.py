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

from functools import partial

import torch

import alf
from alf.algorithms.data_transformer import FrameStacker, RewardNormalizer
# Needs to install safety gym first:
# https://github.com/openai/safety-gym
from alf.environments import suite_safety_gym
from alf.algorithms.lagrangian_reward_weight_algorithm import LagrangianRewardWeightAlgorithm
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.networks import BetaProjectionNetwork, NormalProjectionNetwork
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.utils.math_ops import clipped_exp

from alf.examples import ppo_conf

env_name = "Safexp-PointGoal2-v0"  # natural lidar

# environment config
alf.config(
    'create_environment',
    env_name=env_name,
    num_parallel_environments=32,
    env_load_fn=suite_safety_gym.load)

alf.config('suite_safety_gym.load', episodic=True)

hidden_layers = (256, 256, 256)
alf.config(
    "TrainerConfig",
    data_transformer_ctor=[
        partial(FrameStacker, stack_size=4),
        partial(
            RewardNormalizer,
            clip_value=10.,
            # old ALF won't call transform_experience during replay for one-time replayer
            update_mode="rollout")
    ])

#### these two choices are the key ####
proj_net = partial(BetaProjectionNetwork, min_concentration=1.)
activation = torch.tanh
#######################################

actor_network_cls = partial(
    ActorDistributionNetwork,
    input_preprocessors=alf.layers.Detach(),
    activation=activation,
    continuous_projection_net_ctor=proj_net,
    fc_layer_params=hidden_layers)
#alf.config("NormalProjectionNetwork", std_transform=clipped_exp)
value_network_cls = partial(
    ValueNetwork, activation=activation, fc_layer_params=hidden_layers)

# search range: [1e-3, 0.01, 0]
alf.config(
    'PPOLoss', entropy_regularization=1e-3,
    normalize_advantages=False)  # [False, True]
alf.config(
    'PPOAlgorithm',
    actor_network_ctor=actor_network_cls,
    value_network_ctor=value_network_cls)

alf.config(
    'LagrangianRewardWeightAlgorithm',
    reward_thresholds=[None, -5e-4],
    optimizer=alf.optimizers.AdamTF(lr=1e-2))

alf.config(
    'Agent',
    # [3e-4, 1e-4]
    optimizer=alf.optimizers.AdamTF(lr=1e-4),
    reward_weight_algorithm_cls=LagrangianRewardWeightAlgorithm)

# training config
alf.config(
    'TrainerConfig',
    unroll_length=8,  # (8,16)
    mini_batch_length=1,
    mini_batch_size=256,  # (256,512)
    num_updates_per_train_iter=10,  # (5,10)
    num_iterations=0,
    num_env_steps=int(5e6),
    num_checkpoints=1,
    evaluate=False,
    debug_summaries=True,
    summarize_first_interval=False,
    num_summaries=100)
