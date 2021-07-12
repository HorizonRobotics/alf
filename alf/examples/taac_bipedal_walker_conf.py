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
from alf.algorithms.data_transformer import RewardNormalizer
from alf.algorithms.taac_algorithm import TaacAlgorithm, TaacLAlgorithm, TaacQAlgorithm
from alf.networks import NormalProjectionNetwork, ActorDistributionNetwork, CriticNetwork
from alf.optimizers import AdamTF
from alf.utils import dist_utils, math_ops

from alf.examples import sac_conf

# environment config
alf.config(
    'create_environment',
    env_name="BipedalWalker-v2",
    num_parallel_environments=32)

hidden_layers = (256, 256)

# algorithm config
actor_network_cls = partial(
    ActorDistributionNetwork,
    fc_layer_params=hidden_layers,
    continuous_projection_net_ctor=partial(
        NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=math_ops.clipped_exp))

critic_network_cls = partial(
    CriticNetwork, joint_fc_layer_params=hidden_layers)

alf.config(
    'TaacAlgorithmBase',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.005,
    target_entropy=(partial(
        dist_utils.calc_default_target_entropy, min_prob=0.05),
                    partial(
                        dist_utils.calc_default_target_entropy, min_prob=0.1)))

alg = TaacAlgorithm  # TaacLAlgorithm TaacQAlgorithm
alf.config('Agent', rl_algorithm_cls=alg, optimizer=AdamTF(lr=5e-4))

# training config
alf.config(
    'TrainerConfig',
    temporally_independent_train_step=True,
    data_transformer_ctor=[partial(RewardNormalizer, clip_value=1.)],
    use_rollout_state=True,
    initial_collect_steps=20000,
    mini_batch_length=6,
    unroll_length=5,
    mini_batch_size=4096,
    num_updates_per_train_iter=1,
    num_iterations=0,
    num_env_steps=int(5e6),
    num_checkpoints=5,
    evaluate=False,
    debug_summaries=True,
    summarize_grads_and_vars=False,
    summary_interval=100,
    replay_buffer_length=100000)
