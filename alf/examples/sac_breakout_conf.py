# Copyright (c) 2022 Horizon Robotics. All Rights Reserved.
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

import functools

import alf
from alf.algorithms.td_loss import TDLoss
from alf.environments.alf_wrappers import AtariTerminalOnLifeLossWrapper
from alf.networks import QNetwork
from alf.optimizers import AdamTF

from alf.examples import ac_breakout_conf, sac_conf


def define_config(name, default_value):
    alf.define_config(name, default_value)
    return alf.get_config_value('_CONFIG._USER.' + name)


alf.config(
    'create_environment',
    env_name='BreakoutNoFrameskip-v4',
    num_parallel_environments=30)

FC_LAYER_PARAMS = define_config('FC_LAYER_PARAMS', (512, ))
CONV_LAYER_PARAMS = define_config('CONV_LAYER_PARAMS',
                                  ac_breakout_conf.CONV_LAYER_PARAMS)

q_network_cls = functools.partial(
    QNetwork,
    fc_layer_params=FC_LAYER_PARAMS,
    conv_layer_params=CONV_LAYER_PARAMS)

critic_loss_ctor = functools.partial(TDLoss, td_lambda=0.95)

lr = define_config('lr', 5e-4)
critic_optimizer = AdamTF(lr=lr)
alpha_optimizer = AdamTF(lr=lr)
alf.config('calc_default_target_entropy', min_prob=0.1)

alf.config(
    'SacAlgorithm',
    actor_network_cls=None,
    critic_network_cls=None,
    q_network_cls=q_network_cls,
    critic_loss_ctor=critic_loss_ctor,
    critic_optimizer=critic_optimizer,
    alpha_optimizer=alpha_optimizer,
    target_update_tau=0.05,
    target_update_period=20)

gamma = define_config('gamma', 0.99)
alf.config('OneStepTDLoss', gamma=gamma)
alf.config('ReplayBuffer', gamma=gamma, reward_clip=(-1, 1))

# training config
alf.config(
    'suite_gym.load',
    max_episode_steps=10000,
    alf_env_wrappers=[AtariTerminalOnLifeLossWrapper])

alf.config(
    'TrainerConfig',
    epsilon_greedy=0.05,
    initial_collect_steps=1e5,
    mini_batch_length=2,
    num_updates_per_train_iter=
    4,  # 8GB GPU memory: 40 updates, 250x2 timesteps, 84x84, 3 convs, 4 frames stacked
    mini_batch_size=500,
    unroll_length=8,
    num_env_steps=12000000,
    evaluate=True,
    num_eval_episodes=100,
    num_evals=10,
    num_checkpoints=5,
    num_summaries=100,
    debug_summaries=True,
    use_rollout_state=True,
    replay_buffer_length=33334)  # 20GB CPU memory: 30 envs, 84x84
