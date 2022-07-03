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
import torch
import math
from functools import partial

import alf
from alf.initializers import variance_scaling_init
from alf.algorithms.stable_sac_algorithm import StableSacAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.networks import (
    NormalProjectionNetwork,
    ActorDistributionNetwork,
    CriticNetwork,
    EncodingNetwork,
)
from alf.optimizers import AdamTF
from alf.tensor_specs import TensorSpec
from alf.utils.math_ops import clipped_exp, identity

from alf.examples import sac_conf

# environment config
alf.config("create_environment",
           env_name="HalfCheetah-v2",
           num_parallel_environments=1)

# algorithm config
fc_layer_params = (256, 256)
feat_fc_layer_params = (256, )

actor_network_cls = partial(
    ActorDistributionNetwork,
    fc_layer_params=fc_layer_params,
    continuous_projection_net_ctor=partial(
        NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=clipped_exp,
    ),
)

kernel_initializer = partial(
    variance_scaling_init,
    gain=math.sqrt(1.0 / 3),
    mode="fan_in",
    distribution="uniform",
)

feat_network_cls = partial(
    CriticNetwork,
    joint_fc_layer_params=feat_fc_layer_params,
    output_tensor_spec=TensorSpec((256, )),
    last_kernel_initializer=kernel_initializer,
    last_activation=torch.relu_,
)

critic_network_cls = partial(
    EncodingNetwork,
    input_tensor_spec=TensorSpec((256, )),
    last_layer_size=1,
    last_activation=identity,
    last_kernel_initializer=partial(torch.nn.init.uniform_, a=-0.003, b=0.003),
)

alf.config("calc_default_target_entropy", min_prob=0.184)

alf.config(
    "StableSacAlgorithm",
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    feat_network_cls=feat_network_cls,
    critic_loss_ctor=OneStepTDLoss,
    use_entropy_reward=True,
    target_update_tau=0.005,
    inferf_reg_coef=1e-1,
    actor_optimizer=AdamTF(lr=3e-4),
    critic_optimizer=AdamTF(lr=3e-4),
    feat_optimizer=AdamTF(lr=3e-4),
    alpha_optimizer=AdamTF(lr=3e-4),
)

# training config
alf.config("Agent", rl_algorithm_cls=StableSacAlgorithm)

alf.config(
    "TrainerConfig",
    version='debug',
    use_wandb=True,
    entity="jiachenli",
    project="stable-rl",
    initial_collect_steps=10000,
    mini_batch_length=2,
    unroll_length=1000,
    mini_batch_size=256,
    num_updates_per_train_iter=1000,
    num_iterations=3000,
    num_checkpoints=1,
    evaluate=True,
    eval_interval=1,
    num_eval_episodes=5,
    debug_summaries=True,
    random_seed=0,
    summarize_grads_and_vars=False,
    summary_interval=1,
    replay_buffer_length=1000000,
)
