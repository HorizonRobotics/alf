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

import alf
from alf.algorithms.data_transformer import RewardScaling
from alf.algorithms.dqnx_algorithm import DQNXAlgorithm
from alf.algorithms.agent import Agent
from alf.networks import QNetwork

# Environment Configuration
alf.config(
    'create_environment',
    env_name='CartPole-v0',
    num_parallel_environments=8,
    start_serially=False)
alf.config('FastParallelEnvironment', start_method="spawn")

# Reward Scailing
alf.config('TrainerConfig', data_transformer_ctor=RewardScaling)
alf.config('RewardScaling', scale=0.01)

# algorithm config
alf.config(
    'QNetwork',
    fc_layer_params=(100, ),
    use_fc_ln=True,
    last_layer_init_bias_value=0)

alf.config(
    'DQNXAlgorithm',
    epsilon_greedy=1.0,
    entropy_regularization=0.01,
    alpha=0.95,
    target_update_period=1,
    use_entropy_reward=False,
    log_pi_clip=0,
    delta_log_pi_clip=0.2,
    q_network_ctor=QNetwork,
    optimizer=alf.optimizers.Adam(lr=1e-3))

alf.config('Agent', rl_algorithm_cls=DQNXAlgorithm)

# training config
alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    random_seed=5,
    mini_batch_length=1,
    unroll_length=32,
    mini_batch_size=128,
    num_updates_per_train_iter=4,
    num_iterations=200,
    num_checkpoints=5,
    whole_replay_buffer_training=True,
    clear_replay_buffer=True,
    evaluate=True,
    eval_interval=50,
    confirm_checkpoint_upon_crash=False,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    update_counter_every_mini_batch=True,
    clear_replay_buffer_but_keep_one_step=True,
    summary_interval=1)

alf.config('summarize_gradients', with_histogram=False)
alf.config('summarize_variables', with_histogram=False)
