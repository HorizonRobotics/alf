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
import math

import torch

import alf
from alf.algorithms.data_transformer import (RewardNormalizer, FrameStacker,
                                             UntransformedTimeStep)
# Needs to install safety gym first:
# https://github.com/hnyu/safety-gym
from alf.environments import suite_safety_gym
from alf.algorithms.lagrangian_reward_weight_algorithm import LagrangianRewardWeightAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.networks import BetaProjectionNetwork
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.utils import math_ops
from alf.nest.utils import NestSum, NestConcat
from alf.algorithms.td_loss import TDLoss

from alf.examples import sac_conf

env_name = "Safexp-PointGoal2-v0"  # natural lidar

DEBUG = True

if DEBUG:
    num_envs = 8
    initial_collect_steps = 100
else:
    num_envs = 32
    initial_collect_steps = 10000

# environment config
alf.config(
    'create_environment',
    env_name=env_name,
    num_parallel_environments=num_envs,
    env_load_fn=suite_safety_gym.load)

alf.config('suite_safety_gym.load', episodic=True)

proj_net = partial(BetaProjectionNetwork, min_concentration=1.)

hidden_layers = (256, 256, 256)
alf.config(
    "TrainerConfig",
    data_transformer_ctor=[
        UntransformedTimeStep,
        partial(FrameStacker, stack_size=4),
        partial(RewardNormalizer, clip_value=10.)
    ])

activation = torch.tanh
actor_network_ctor = partial(
    ActorDistributionNetwork,
    fc_layer_params=hidden_layers,
    activation=activation,
    continuous_projection_net_ctor=proj_net)
actor_network_cls = partial(
    actor_network_ctor, input_preprocessors=alf.layers.Detach())
critic_network_cls = partial(
    CriticNetwork, activation=activation, joint_fc_layer_params=hidden_layers)

alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.005,
    target_update_period=1,
    critic_loss_ctor=TDLoss)

alf.config('calc_default_target_entropy', min_prob=0.1)

alf.config(
    'LagrangianRewardWeightAlgorithm',
    reward_thresholds=[None, -5e-4],
    optimizer=alf.optimizers.AdamTF(lr=0.01))

alf.config(
    'Agent',
    rl_algorithm_cls=SacAlgorithm,
    optimizer=alf.optimizers.AdamTF(lr=3e-4),
    reward_weight_algorithm_cls=LagrangianRewardWeightAlgorithm)

# training config
alf.config(
    'TrainerConfig',
    temporally_independent_train_step=True,
    initial_collect_steps=initial_collect_steps,
    mini_batch_length=8,
    unroll_length=5,
    mini_batch_size=1024,
    num_updates_per_train_iter=1,
    num_iterations=0,
    num_env_steps=int(5e6),
    num_checkpoints=1,
    evaluate=False,
    num_evals=100,
    debug_summaries=True,
    summarize_first_interval=False,
    num_summaries=100,
    replay_buffer_length=50000)
