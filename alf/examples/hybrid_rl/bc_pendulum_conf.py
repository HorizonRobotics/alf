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
import torch

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.bc_algorithm import BcAlgorithm

# default params
lr = 5e-4
encoding_dim = 256
fc_layers_params = (encoding_dim, ) * 2
activation = torch.relu_

offline_buffer_length = None
offline_buffer_dir = [
    "./hybrid_rl/replay_buffer_data/pendulum_replay_buffer_from_sac_10k"
]

alf.config(
    "create_environment", env_name="Pendulum-v0", num_parallel_environments=1)

alf.config(
    'Agent',
    rl_algorithm_cls=BcAlgorithm,
    optimizer=alf.optimizers.Adam(lr=1e-4),
)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False)

proj_net = partial(
    alf.networks.StableNormalProjectionNetwork,
    state_dependent_std=True,
    squash_mean=False,
    scale_distribution=True,
    min_std=1e-3,
    max_std=10)

actor_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    fc_layer_params=fc_layers_params,
    activation=activation,
    continuous_projection_net_ctor=proj_net)

use_action_delta = False
alf.config('BcAlgorithm', actor_network_cls=actor_network_cls)

num_iterations = 10000000

# training config
alf.config(
    "TrainerConfig",
    initial_collect_steps=0,
    num_updates_per_train_iter=1,
    num_iterations=num_iterations,
    rl_train_after_update_steps=num_iterations + 1000,
    rl_train_every_update_steps=1,
    mini_batch_size=64,
    mini_batch_length=2,
    offline_buffer_dir=offline_buffer_dir,
    offline_buffer_length=offline_buffer_length,
    num_checkpoints=1,
    debug_summaries=True,
    evaluate=True,
    eval_interval=1000,
    num_eval_episodes=3,
)
