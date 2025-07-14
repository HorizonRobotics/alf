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
from alf.algorithms.bafc_algorithm_v1 import BafcAlgorithmV1
from alf.algorithms.data_transformer import ObservationNormalizer
from alf.examples.benchmarks.dm_control import dmc_conf
from alf.optimizers import Adam

actor_hidden_layers = (256, 256)
# actor_hidden_layers = (32, 32)
optimizer = Adam(lr=5e-4)
use_obs_normalizer = True
obs_normalizer_clipping = False
critic_use_memory = False

if use_obs_normalizer:
    data_transformer_ctor = ObservationNormalizer
else:
    data_transformer_ctor = None
if obs_normalizer_clipping:
    alf.config('ObservationNormalizer', clipping=1.)

actor_network_cls = partial(
    alf.networks.ActorFCNetwork,
    fc_layer_params=actor_hidden_layers)

if critic_use_memory:
    critic_network_cls = partial(
        alf.networks.FuncCriticMemTransformer,
        obs_action_joint_fc_layer_params=dmc_conf.hidden_layers,
        num_attention_heads=1,
        num_transformer_layers=4,
        num_memory_layers=0,
        memory_size=256,
        use_fc_ln=True)  # turning on critic layernorm is crucial for high utd
else:
    critic_network_cls = partial(
        alf.networks.FuncCriticTransformer,
        obs_action_joint_fc_layer_params=dmc_conf.hidden_layers,
        num_attention_heads=1,
        num_transformer_layers=4,
        use_fc_ln=True)  # turning on critic layernorm is crucial for high utd

alf.config('Agent',
           optimizer=optimizer,
           rl_algorithm_cls=BafcAlgorithmV1)

alf.config(
    'BafcAlgorithmV1',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    num_actors=10,
    use_target_actor=False,
    use_bootstrap_actors=True,
    bootstrap_mask_prob=0.8,
    num_actor_eval_samples=512,
    # num_actor_eval_samples=64,
    eval_samples_init_method='normal',
    eval_samples_clipping=obs_normalizer_clipping,
    actor_eval_type='last',
    actor_utd=1,
    critic_utd=5,
    target_critic_tau=0.005,
    target_critic_period=1,
    target_critic_use_ema=False,
    target_actor_tau=0.05,
    target_actor_period=1,
    target_actor_use_ema=False)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    data_transformer_ctor=data_transformer_ctor,
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
