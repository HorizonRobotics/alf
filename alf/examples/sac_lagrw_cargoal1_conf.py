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

import alf
from alf.algorithms.data_transformer import RewardNormalizer, UntransformedTimeStep
# Needs to install safety gym first:
# https://github.com/openai/safety-gym
from alf.environments import suite_safety_gym
from alf.algorithms.lagrangian_reward_weight_algorithm import LagrangianRewardWeightAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.networks import NormalProjectionNetwork, ActorDistributionNetwork, CriticNetwork
from alf.utils import math_ops
from alf.algorithms.td_loss import TDLoss
from alf.optimizers import AdamTF

from alf.examples import sac_conf

# environment config
alf.config(
    'create_environment',
    env_name="Safexp-CarGoal1-v0",
    num_parallel_environments=30,
    env_load_fn=suite_safety_gym.load)

# algorithm config
actor_network_cls = partial(
    ActorDistributionNetwork,
    fc_layer_params=(256, 256),
    continuous_projection_net_ctor=partial(
        NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=partial(
            math_ops.clipped_exp, clip_value_min=-10, clip_value_max=2)))

critic_network_cls = partial(CriticNetwork, joint_fc_layer_params=(256, 256))

alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.05,
    use_entropy_reward=False,
    critic_loss_ctor=TDLoss)

alf.config('calc_default_target_entropy', min_prob=0.05)

alf.config(
    'LagrangianRewardWeightAlgorithm',
    init_weights=1.,
    reward_thresholds=[None, -0.025, None],
    optimizer=AdamTF(lr=1e-3))

alf.config(
    'Agent',
    rl_algorithm_cls=SacAlgorithm,
    optimizer=AdamTF(lr=1e-3),
    reward_weight_algorithm_cls=LagrangianRewardWeightAlgorithm)

# training config
alf.config(
    'TrainerConfig',
    temporally_independent_train_step=True,
    data_transformer_ctor=[
        UntransformedTimeStep,
        partial(RewardNormalizer, clip_value=10.)
    ],
    initial_collect_steps=50000,
    mini_batch_length=20,
    unroll_length=10,
    mini_batch_size=2000,
    num_updates_per_train_iter=10,
    num_iterations=0,
    num_env_steps=1e7,
    num_checkpoints=5,
    evaluate=False,
    debug_summaries=True,
    summarize_grads_and_vars=False,
    summary_interval=10,
    replay_buffer_length=100000)
