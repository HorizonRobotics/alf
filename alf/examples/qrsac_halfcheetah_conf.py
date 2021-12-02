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
from alf.algorithms.qrsac_algorithm import QrsacAlgorithm
from alf.algorithms.one_step_loss import OneStepTDQRLoss
from alf.nest.utils import NestConcat
from alf.networks import ActorDistributionNetwork, CriticNetwork, NormalProjectionNetwork
from alf.optimizers import Adam, AdamTF
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import clipped_exp

from alf.examples import sac_conf

# environment config
alf.config(
    'create_environment',
    env_name="HalfCheetah-v2",
    num_parallel_environments=1)

# algorithm config
fc_layer_params = (256, 256)
num_quantiles = 50

actor_network_cls = partial(
    ActorDistributionNetwork,
    fc_layer_params=fc_layer_params,
    continuous_projection_net_ctor=partial(
        NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=clipped_exp))

critic_network_cls = partial(
    CriticNetwork,
    joint_fc_layer_params=fc_layer_params,
    output_tensor_spec=TensorSpec((num_quantiles, )))

# def critic_network_cls(input_tensor_spec):
#     return alf.nn.Sequential(
#         alf.nn.EncodingNetwork(
#             input_tensor_spec,
#             preprocessing_combiner=alf.layers.NestConcat(dim=-1),
#             fc_layer_params=fc_layer_params[:-1],
#             last_layer_size=fc_layer_params[-1],
#             last_activation=torch.relu_),
#         alf.nn.QuantileProjectionNetwork(
#             input_size=fc_layer_params[-1],
#             output_tensor_spec=TensorSpec((num_quantiles, ))))

alf.config('calc_default_target_entropy', min_prob=0.184)

alf.config(
    'QrsacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    critic_loss_ctor=OneStepTDQRLoss,
    target_update_tau=0.005,
    actor_optimizer=AdamTF(lr=3e-4),
    critic_optimizer=AdamTF(lr=3e-4),
    alpha_optimizer=AdamTF(lr=3e-4))

alf.config('OneStepTDQRLoss', num_quantiles=num_quantiles)

# training config
alf.config('Agent', rl_algorithm_cls=QrsacAlgorithm)

alf.config(
    'TrainerConfig',
    initial_collect_steps=10000,
    mini_batch_length=2,
    unroll_length=1,
    mini_batch_size=256,
    num_updates_per_train_iter=1,
    num_iterations=2500000,
    num_checkpoints=1,
    evaluate=True,
    eval_interval=10000,
    num_eval_episodes=5,
    debug_summaries=True,
    random_seed=0,
    summarize_grads_and_vars=True,
    summary_interval=1000,
    replay_buffer_length=1000000)
