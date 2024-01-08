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
from alf.algorithms.smodice_algorithm import SmodiceAlgorithm
from alf.utils import math_ops

from alf.examples import sac_bipedal_walker_conf

# default params
lr = 1e-4
encoding_dim = 256
fc_layers_params = (encoding_dim, ) * 2
activation = torch.relu_

offline_buffer_length = None
offline_buffer_dir = [
    "/home/haichaozhang/data/DATA/sac_bipedal_baseline/train/algorithm/ckpt-80000-replay_buffer"
]

alf.config('Agent', rl_algorithm_cls=SmodiceAlgorithm, optimizer=None)

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

v_network_cls = partial(
    alf.networks.ValueNetwork,
    fc_layer_params=fc_layers_params,
    activation=activation)

action_spec = alf.get_action_spec()
discriminator_network_cls = partial(
    alf.networks.CriticNetwork, joint_fc_layer_params=fc_layers_params)

alf.config(
    'SmodiceAlgorithm',
    actor_network_cls=actor_network_cls,
    v_network_cls=v_network_cls,
    discriminator_network_cls=discriminator_network_cls,
    actor_optimizer=alf.optimizers.Adam(lr=lr),
    # add weight decay to the v_net following smodice paper
    value_optimizer=alf.optimizers.Adam(lr=lr, weight_decay=1e-4),
    discriminator_optimizer=alf.optimizers.Adam(lr=lr),
    gradient_penalty_weight=0.1,
)

# training config
alf.config(
    "TrainerConfig",
    offline_buffer_dir=offline_buffer_dir,
    offline_buffer_length=offline_buffer_length)
