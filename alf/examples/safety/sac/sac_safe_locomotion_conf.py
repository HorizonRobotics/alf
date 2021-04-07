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
from alf.examples import sac_conf
from alf.networks import BetaProjectionNetwork, NormalProjectionNetwork
from alf.algorithms.lagrangian_reward_weight_algorithm import LagrangianRewardWeightAlgorithm
from alf.algorithms.td_loss import TDLoss
from alf.utils.math_ops import clipped_exp

from alf.environments import suite_safe_locomotion

alf.config(
    "create_environment",
    num_parallel_environments=1,
    env_name="",
    env_load_fn=suite_safe_locomotion.load)

env_name = alf.get_config_value("create_environment.env_name")

MAX_SPEED = {
    "SafeAnt-v3": 4.,
    "SafeHalfCheetah-v3": 12.,
    "SafeHopper-v3": 3.,
    "SafeHumanoid-v3": 1.,
    "SafeSwimmer-v3": 0.1,
    "SafeWalker2d-v3": 4.,
}
CONSTRAINT_LEVEL = 0.8
MODE = "speed"

alf.config("VectorReward", constraint_mode=MODE)

if MODE == "speed":
    reward_thresholds = [None, CONSTRAINT_LEVEL * MAX_SPEED[env_name]]
else:
    reward_thresholds = [None, -CONSTRAINT_LEVEL]
    alf.config("TDLoss", gamma=[0.99, 0.])

hidden_layers = (256, 256)

proj_net = partial(BetaProjectionNetwork, min_concentration=1.)

actor_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    input_preprocessors=alf.layers.Detach(),
    fc_layer_params=hidden_layers,
    continuous_projection_net_ctor=proj_net)
critic_network_cls = partial(
    alf.networks.CriticNetwork, joint_fc_layer_params=hidden_layers)

alf.config(
    'LagrangianRewardWeightAlgorithm',
    init_weights=(1., 1.),
    reward_thresholds=reward_thresholds,
    max_weight=50.,
    optimizer=alf.optimizers.AdamTF(lr=1e-2))

alf.config(
    'Agent',
    optimizer=alf.optimizers.AdamTF(lr=3e-4),
    reward_weight_algorithm_cls=LagrangianRewardWeightAlgorithm)

alf.config('calc_default_target_entropy', min_prob=0.184)

alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.005,
    use_parallel_network=True,
    # only cares about constraint at the current step
    critic_loss_ctor=TDLoss,
    use_entropy_reward=False)

alf.config(
    'TrainerConfig',
    temporally_independent_train_step=True,
    initial_collect_steps=50000,
    mini_batch_length=2,
    unroll_length=100,
    mini_batch_size=256,
    epsilon_greedy=0,
    num_updates_per_train_iter=100,
    num_iterations=0,
    num_env_steps=3000000,
    num_checkpoints=5,
    evaluate=False,
    num_evals=100,
    debug_summaries=1,
    num_summaries=100,
    replay_buffer_length=1000000)
