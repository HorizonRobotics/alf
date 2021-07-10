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
import functools

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.agent import Agent
from alf.networks import ActorDistributionNetwork, CategoricalProjectionNetwork, ValueNetwork

from alf.examples import atari_conf

# From OpenAI gym wiki:
# "v0 vs v4: v0 has repeat_action_probability of 0.25
#  (meaning 25% of the time the previous action will be used instead of the new action),
#   while v4 has 0 (always follow your issued action)
# Because we already implements frame_skip in AtariPreprocessing, we should always
# use 'NoFrameSkip' Atari environments from OpenAI gym
alf.config(
    'create_environment',
    env_name='BreakoutNoFrameskip-v4',
    num_parallel_environments=64)

# Neural Network Configuration
CONV_LAYER_PARAMS = ((32, 8, 4), (64, 4, 2), (64, 3, 1))
actor_network_cls = functools.partial(
    ActorDistributionNetwork,
    fc_layer_params=(512, ),
    conv_layer_params=CONV_LAYER_PARAMS)
value_network_cls = functools.partial(
    ValueNetwork, fc_layer_params=(512, ), conv_layer_params=CONV_LAYER_PARAMS)

alf.config('CategoricalProjectionNetwork', logits_init_output_factor=1e-10)

# Algorithm Configuration
alf.config(
    'ActorCriticLoss',
    entropy_regularization=0.01,
    use_gae=True,
    use_td_lambda_return=True,
    td_lambda=0.95,
    td_loss_weight=0.5,
    advantage_clip=None)

alf.config(
    'ActorCriticAlgorithm',
    actor_network_ctor=actor_network_cls,
    value_network_ctor=value_network_cls,
    optimizer=alf.optimizers.Adam(lr=1e-3))
alf.config('Agent', rl_algorithm_cls=ActorCriticAlgorithm)

alf.config(
    'TrainerConfig',
    unroll_length=8,
    algorithm_ctor=Agent,
    num_iterations=0,
    num_env_steps=5000000,
    evaluate=False,
    debug_summaries=1,
    summarize_grads_and_vars=1,
    summary_interval=10)
