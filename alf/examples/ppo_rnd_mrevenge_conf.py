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
"""Configuration to train on Atari Game Montezuma's Revenge with PPO + RND exploration.

NOTE: Currently this configuration only achieves around 3000 (reward) instead of
around 6000 as it used to be.

TODO: Tune the parameters to make it achieving 6000 or better reward again.

"""
import alf
import torch

import functools

# from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.rnd_algorithm import RNDAlgorithm
from alf.networks import EncodingNetwork, ActorDistributionNetwork, CategoricalProjectionNetwork, ValueNetwork
from alf.tensor_specs import TensorSpec

from alf.examples import ppo_conf
from alf.examples import atari_conf

alf.config('DMAtariPreprocessing', noop_max=0)

# From OpenAI gym wiki:
#
# "v0 vs v4: v0 has repeat_action_probability of 0.25
#  (meaning 25% of the time the previous action will be used instead of the new action),
#   while v4 has 0 (always follow your issued action)
# Because we already implements frame_skip in AtariPreprocessing, we should always
# use 'NoFrameSkip' Atari environments from OpenAI gym
alf.config(
    'create_environment',
    env_name='MontezumaRevengeNoFrameskip-v0',
    num_parallel_environments=128)

# RND config

KEEP_STACKED_FRAMES = 1
EMBEDDING_DIM = 1000

alf.config(
    'RNDAlgorithm',
    encoder_net=EncodingNetwork(
        activation=torch.tanh,
        input_tensor_spec=TensorSpec(shape=(KEEP_STACKED_FRAMES, 84, 84)),
        conv_layer_params=((64, 5, 5), (64, 2, 2), (64, 2, 2))),
    target_net=EncodingNetwork(
        activation=torch.tanh,
        input_tensor_spec=TensorSpec(shape=(1024, )),
        fc_layer_params=(300, 400, 500, EMBEDDING_DIM)),
    predictor_net=EncodingNetwork(
        activation=torch.tanh,
        input_tensor_spec=TensorSpec(shape=(1024, )),
        fc_layer_params=(300, 400, 500, EMBEDDING_DIM)),
    optimizer=alf.optimizers.AdamTF(lr=4e-5),
    keep_stacked_frames=KEEP_STACKED_FRAMES)

alf.config(
    'Agent',
    extrinsic_reward_coef=1.0,
    intrinsic_reward_module=RNDAlgorithm(),
    intrinsic_reward_coef=1e-3,
    optimizer=alf.optimizers.AdamTF(lr=1e-4))

alf.config('PPOLoss', entropy_regularization=0.01)

# Neural Network Configuration
CONV_LAYER_PARAMS = ((32, 8, 4), (64, 4, 2), (64, 3, 1))
FC_LAYER_PARAMS = (512, 512)

actor_network_cls = functools.partial(
    ActorDistributionNetwork,
    fc_layer_params=FC_LAYER_PARAMS,
    conv_layer_params=CONV_LAYER_PARAMS,
    discrete_projection_net_ctor=CategoricalProjectionNetwork)

alf.config('CategoricalProjectionNetwork', logits_init_output_factor=1e-10)

value_network_cls = functools.partial(
    ValueNetwork,
    fc_layer_params=FC_LAYER_PARAMS,
    conv_layer_params=CONV_LAYER_PARAMS)

alf.config(
    'ActorCriticAlgorithm',
    actor_network_ctor=actor_network_cls,
    value_network_ctor=value_network_cls)

alf.config(
    'TrainerConfig',
    num_updates_per_train_iter=6,
    unroll_length=32,
    mini_batch_length=1,
    mini_batch_size=1024,
    num_iterations=0,
    num_env_steps=50000000,  # = 200M frames / 4 (frame_skip)
    debug_summaries=True,
    summarize_grads_and_vars=False,
    summary_interval=100,
    num_checkpoints=10,
    use_rollout_state=True,
    update_counter_every_mini_batch=True)
