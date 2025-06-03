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

import alf
from alf.algorithms.mdqn_algorithm import MDQNAlgorithm
from alf.algorithms.agent import Agent
from alf.networks import QNetwork
from alf.utils import schedulers
from alf.environments.alf_wrappers import AtariTerminalOnLifeLossWrapper
from alf.utils import dist_utils

import torch

alf.import_config("atari_conf.py")

alf.config(
    'suite_gym.load',
    max_episode_steps=10000,
    alf_env_wrappers=[AtariTerminalOnLifeLossWrapper])

# Environment Configuration
alf.config(
    'create_environment',
    env_name='BreakoutNoFrameskip-v4',
    num_parallel_environments=32,  # 1
)

# Neural Network Configuration
CONV_LAYER_PARAMS = ((32, 8, 4), (64, 4, 2), (64, 3, 1))

# algorithm config
alf.config(
    'QNetwork',
    fc_layer_params=(512, ),
    conv_layer_params=CONV_LAYER_PARAMS,
    activation=torch.relu_,
    dueling=False,
    use_fc_ln=False)
# last_layer_init_weight_range=0.0,
# last_layer_init_bias_value=0)

alf.config(
    'MDQNAlgorithm',
    num_replicas=1,
    epsilon_greedy=
    1.0,  # schedulers.LinearScheduler("iterations", [(0, 1.0), (250_000, 0.01)]),
    use_sac_style_target=False,
    gamma=0.99,
    td_lambda=0.0,
    entropy_regularization=1.0,
    target_entropy=partial(
        dist_utils.calc_default_target_entropy, min_prob=0.1),
    alpha=0.9,
    target_update_period=400,  # 8000,
    target_update_tau=1,
    epsilon_greedy_uniform=False,
    use_entropy_reward=False,
    log_pi_clip=-1,
    separate_q_m=True,
    q_network_ctor=QNetwork,
    optimizer=alf.optimizers.Adam(
        lr=5e-4,  # 5e-5
    ))

alf.config('Agent', rl_algorithm_cls=MDQNAlgorithm)

# training config
alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    random_seed=1,
    mini_batch_length=2,
    unroll_length=2,  # 4,
    mini_batch_size=512,  # 32,
    num_updates_per_train_iter=1,
    num_iterations=0,
    num_env_steps=12_000_000,  #5000000,
    initial_collect_steps=1e5,  # 50000,
    num_checkpoints=1,
    replay_buffer_length=33334,  # 1000_000,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    evaluate=True,
    num_evals=20,
    num_eval_episodes=100,
    num_eval_environments=10,
    confirm_checkpoint_upon_crash=False,
    debug_summaries=True,
    summarize_first_interval=False,
    summarize_grads_and_vars=True,
    update_counter_every_mini_batch=False,
    clear_replay_buffer_but_keep_one_step=True,
    num_summaries=100)

alf.config('summarize_gradients', with_histogram=False)
alf.config('summarize_variables', with_histogram=False)
