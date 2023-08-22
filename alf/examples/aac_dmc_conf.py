# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.algorithms.agent import Agent
from alf.algorithms.aac_algorithm import AacAlgorithm
from alf.examples import dmc_conf
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.networks import NormalProjectionNetwork, BetaProjectionNetwork
from alf.networks import CauchyProjectionNetwork
from alf.optimizers import AdamTF
from alf.examples import dmc_conf
from alf.utils.summary_utils import summarize_tensor_gradients

alf.config(
    'ActorDistributionNetwork',
    fc_layer_params=dmc_conf.hidden_layers,
    continuous_projection_net_ctor=partial(
        BetaProjectionNetwork, min_concentration=1.))

alf.config('ValueNetwork', fc_layer_params=dmc_conf.hidden_layers)

alf.config('Agent', rl_algorithm_cls=AacAlgorithm)

alf.config(
    'AacAlgorithm',
    actor_network_cls=ActorDistributionNetwork,
    value_network_cls=ValueNetwork,
    num_mc_samples=10,
    initial_log_alpha=0.0,
    optimizer=AdamTF(lr=3e-4))

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    summarize_gradient_noise_scale=True,
    summarize_action_distributions=True,
    random_seed=0)
