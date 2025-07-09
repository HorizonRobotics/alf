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
from alf.algorithms.bafc_algorithm_v0 import BafcAlgorithmV0
from alf.examples.benchmarks.dm_control import dmc_conf
from alf.optimizers import Adam

actor_hidden_layers = (256, 128)
joint_hidden_layers = (256, 256)
# actor_hidden_layers = (32, 32)
# joint_hidden_layers = (32, 32)
optimizer = Adam(lr=5e-4)

actor_network_cls = partial(
    alf.networks.ActorFCNetwork,
    fc_layer_params=actor_hidden_layers)

critic_network_cls = partial(
    alf.networks.FuncCriticNetwork,
    obs_action_joint_fc_layer_params=dmc_conf.hidden_layers,
    actor_obs_action_joint_fc_layer_params=joint_hidden_layers,
    use_fc_ln=True)  # turning on critic layernorm is crucial for high utd

alf.config('Agent',
           optimizer=optimizer,
           rl_algorithm_cls=BafcAlgorithmV0)

alf.config(
    'BafcAlgorithmV0',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    num_actors=10,
    use_target_actor=False,
    use_bootstrap_actors=True,
    bootstrap_mask_prob=0.8,
    num_actor_eval_samples=512,
    # num_actor_eval_samples=64,
    actor_graph_node_dim=64,
    actor_graph_edge_dim=32,
    actor_encoding_dim=256,
    obs_action_encoding_dim=128,
    actor_utd=1,
    critic_utd=2,
    target_critic_tau=0.005,
    target_critic_period=1,
    target_critic_use_ema=False,
    target_actor_tau=0.05,
    target_actor_period=1,
    target_actor_use_ema=False)

alf.config(
    'GraphNetwork',
    d_out_hid=256,
    dropout=0.0,
    disable_edge_updates=True,
    pooling_method="cat",
    pooling_layer_idx="last")

alf.config(
    'ActorGraph',
    sin_emb=True,
    sin_emb_dim=128)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    enable_amp=False,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    num_updates_per_train_iter=6,
    num_env_steps=int(1e7),
    evaluate=False,
    debug_summaries=True,
    summary_interval=100,
    summarize_grads_and_vars=True,
    summarize_gradient_noise_scale=False,
    summarize_action_distributions=False,
    summarize_train_every_mini_batch=True,
    random_seed=0)
