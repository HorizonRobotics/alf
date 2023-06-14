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
from alf.algorithms.vfbc_algorithm import VfbcAlgorithm
from alf.algorithms.dynamics_learning_algorithm import DeterministicDynamicsAlgorithm, StochasticDynamicsAlgorithm
from alf.algorithms.dynamics_learning_algorithm import DeterministicDynamicsDeltaAlgorithm
from alf.initializers import variance_scaling_init
from alf.networks.dynamics_networks import DynamicsNetwork
from alf.utils.math_ops import swish

# default params
lr = 1e-4
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
    rl_algorithm_cls=VfbcAlgorithm,
    optimizer=alf.optimizers.Adam(lr=lr),
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

alf.config(
    "DynamicsNetwork",
    activation=swish,
    joint_fc_layer_params=(500, 500, 500),
    kernel_initializer=variance_scaling_init)

alf.config("DynamicsNetwork", prob=0)
dynamics_module_ctor = partial(
    # DeterministicDynamicsAlgorithm,
    DeterministicDynamicsDeltaAlgorithm,
    num_replicas=1,
    dynamics_network_ctor=DynamicsNetwork)

alf.config('VfbcAlgorithm', 
           actor_network_cls=actor_network_cls,
           dynamics_module_ctor=dynamics_module_ctor)

num_iterations = 100000

# training config
alf.config(
    "TrainerConfig",
    initial_collect_steps=0,
    num_updates_per_train_iter=1,
    num_iterations=num_iterations,
    # use_rollout_state=True,
    # disable rl training by setting rl_train_after_update_steps
    # to be larger than num_iterations
    rl_train_after_update_steps=num_iterations + 1000,
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
