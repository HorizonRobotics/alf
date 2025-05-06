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
import torch

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.bafc_algorithm import BafcAlgorithm
from alf.examples.benchmarks.dm_control import dmc_conf

actor_network_cls = partial(
    alf.networks.ActorFCNetwork,
    fc_layer_params=dmc_conf.hidden_layers)

critic_network_cls = partial(
    alf.networks.FuncCriticNetwork,
    actor_encoding_dim=32,
    obs_action_encoding_dim=32,
    obs_action_joint_fc_layer_params=dmc_conf.hidden_layers,
    actor_obs_action_joint_fc_layer_params=dmc_conf.hidden_layers,
    use_fc_ln=True)  # turning on critic layernorm is crucial for high utd

alf.config('Agent',
           optimizer=dmc_conf.optimizer,
           rl_algorithm_cls=BafcAlgorithm)

alf.config(
    'BafcAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    num_bootstrapped_actors=5,
    actor_utd=1,
    critic_utd=5,
    target_update_tau=0.005)

alf.config(
    'RelationalTransformer',
    num_graph_eval_samples=50,
    use_cls_token=False,
    pooling_method="cat")

alf.config(
    'GraphConstructor',
    sin_emb=True,
    sin_emb_dim=128)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    enable_amp=True,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    num_updates_per_train_iter=6,
    summarize_gradient_noise_scale=False,
    summarize_action_distributions=False,
    random_seed=0)
