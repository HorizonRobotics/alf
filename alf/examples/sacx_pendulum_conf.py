# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import alf
from alf.environments import suite_gym, suite_dmc
from alf.utils.losses import element_wise_squared_loss
from alf.algorithms.sacx_algorithm import SacXAlgorithm
from alf.algorithms.td_loss import TDLoss
from alf.algorithms.agent import Agent
from alf.utils.losses import element_wise_huber_loss

import alf.utils.math_ops

alf.config("suite_gym.load", randomize_first_episode_length=True)

# environment config
alf.config(
    'create_environment',
    env_load_fn=suite_gym.load,
    env_name="Pendulum-v0",
    num_parallel_environments=32)

# algorithm config
alf.config("ActorDistributionNetwork", fc_layer_params=(256, 256, 256))
alf.config("CriticNetwork", joint_fc_layer_params=(256, 256, 256))
alf.config("ValueNetwork", fc_layer_params=(256, 256, 256))

alf.config(
    "SacAlgorithm",
    use_entropy_reward=False,
    actor_optimizer=alf.optimizers.AdamTF(lr=5e-4),
    critic_optimizer=alf.optimizers.Adam(
        lr=5e-4, gradient_clipping=200, clip_by_global_norm=True),
    alpha_optimizer=alf.optimizers.Adam(lr=5e-4),
    num_critic_replicas=2,
    initial_log_alpha=math.log(1e-30),
    value_network_ctor=alf.networks.ValueNetwork,
    target_update_tau=0.005,
    critic_loss_ctor=partial(TDLoss, gamma=0.99),
)

alf.config(
    "SacXAlgorithm",
    num_critic_steps=16,
    kld_weight=1.0,
    mode='dqda',
)

alf.config(
    'PPOLoss',
    entropy_regularization=1e-4,
    gamma=0.99,
    td_error_loss_fn=element_wise_huber_loss,
    advantage_clip=0.2,
    normalize_advantages=True)

alf.config('Agent', rl_algorithm_cls=SacXAlgorithm)

# training config
alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    mini_batch_length=1,
    unroll_length=32,
    mini_batch_size=512,
    num_updates_per_train_iter=12,
    num_iterations=400,
    num_checkpoints=1,
    evaluate=True,
    whole_replay_buffer_training=True,
    clear_replay_buffer=True,
    eval_interval=50,
    random_seed=1,
    summarize_action_distributions=True,
    summarize_grads_and_vars=True,
    debug_summaries=True,
    summary_interval=5)

alf.config('summarize_gradients', with_histogram=False)
alf.config('summarize_variables', with_histogram=False)
