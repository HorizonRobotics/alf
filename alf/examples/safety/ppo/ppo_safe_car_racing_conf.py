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
import math

import torch

import gym
import numpy as np

import alf
from alf.algorithms.data_transformer import (
    RewardNormalizer, ObservationNormalizer, ImageScaleTransformer,
    FrameStacker)

import alf.environments.safe_car_racing
from alf.environments.gym_wrappers import FrameGrayScale, FrameSkip
from alf.environments.alf_wrappers import RewardObservationWrapper
from alf.algorithms.lagrangian_reward_weight_algorithm import LagrangianRewardWeightAlgorithm
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.networks import BetaProjectionNetwork
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.algorithms.encoding_algorithm import EncodingAlgorithm
from alf.utils import math_ops

from alf.examples import ppo_conf


class VectorReward(gym.Wrapper):
    """This wrapper makes the env returns a reward vector of length 2.
    """

    REWARD_DIMENSION = 2

    def __init__(self, env):
        super().__init__(env)
        self.reward_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=[self.REWARD_DIMENSION])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Get the second and third reward from ``info`` and use the minimum
        constraint_reward = min(info["obstacles_reward"],
                                info["out_of_track_reward"])
        return obs, np.array([reward, constraint_reward],
                             dtype=np.float32), done, info


env_name = "SafeCarRacing1-v0"

DEBUG = False

if DEBUG:
    num_envs = 4
else:
    num_envs = 16

# environment config
alf.config(
    'create_environment',
    env_name=env_name,
    num_parallel_environments=num_envs)

alf.config("ImageChannelFirst", fields=["rgb"])

alf.config(
    "suite_gym.load",
    gym_env_wrappers=(
        partial(FrameGrayScale, fields=['rgb']),
        VectorReward,
    ),
    max_episode_steps=1000)

proj_net = partial(BetaProjectionNetwork, min_concentration=1.)

latent_size = 256
alf.config(
    "TrainerConfig",
    data_transformer_ctor=[
        #partial(FrameStacker, stack_size=4, fields=['observation']),
        partial(ImageScaleTransformer, min=0., max=1., fields=['rgb']),
        partial(RewardNormalizer, clip_value=10., update_mode='rollout')
    ])

activation = torch.tanh
actor_network_ctor = partial(
    ActorDistributionNetwork,
    fc_layer_params=(latent_size, ) * 2,
    activation=activation,
    continuous_projection_net_ctor=proj_net)
actor_network_cls = partial(
    actor_network_ctor, input_preprocessors=alf.layers.Detach())
value_network_cls = partial(
    ValueNetwork, activation=activation, fc_layer_params=(latent_size, ) * 2)

alf.config("PPOLoss", entropy_regularization=1e-2, normalize_advantages=True)

alf.config(
    'PPOAlgorithm',
    actor_network_ctor=actor_network_cls,
    value_network_ctor=value_network_cls)

alf.config(
    'LagrangianRewardWeightAlgorithm',
    #init_weights=[1., 0.],
    reward_thresholds=[None, -5e-4],
    optimizer=alf.optimizers.AdamTF(lr=0.01))

obs_spec = alf.get_observation_spec()

encoder_cls = partial(
    alf.networks.EncodingNetwork,
    input_preprocessors={
        'rgb':
            alf.networks.EncodingNetwork(
                input_tensor_spec=obs_spec['rgb'],
                conv_layer_params=((32, 8, 4), (64, 4, 2), (64, 3, 1)),
                activation=torch.relu_,
                last_activation=math_ops.identity,
                last_layer_size=latent_size),
        #'rgb': ResnetEncodingNetwork(
        #          input_tensor_spec=obs_spec['rgb'],
        #          output_size=latent_size,
        #          output_activation=math_ops.identity,
        #          use_fc_bn=False),
        #'rgb': impala_cnn_encoder.create(
        #      input_tensor_spec=obs_spec['rgb'],
        #      cnn_channel_list=(16, 32, 32),
        #      num_blocks_per_stack=2,
        #      output_activation=math_ops.identity,
        #      output_size=latent_size
        #),
        'car':
            torch.nn.Sequential(
                alf.layers.FC(obs_spec['car'].numel, latent_size))
    },
    preprocessing_combiner=alf.layers.NestSum(
        activation=activation, average=True))

from alf.algorithms.encoding_algorithm import EncodingAlgorithm
alf.config('EncodingAlgorithm', encoder_cls=encoder_cls)

learning_rate = 3e-4
alf.config(
    'Agent',
    representation_learner_cls=EncodingAlgorithm,
    optimizer=alf.optimizers.AdamTF(lr=learning_rate),
    reward_weight_algorithm_cls=LagrangianRewardWeightAlgorithm)

alf.config("summarize_variables", with_histogram=False)
alf.config("summarize_gradients", with_histogram=False)

# training config
alf.config(
    'TrainerConfig',
    unroll_length=8,
    mini_batch_length=1,
    mini_batch_size=128,
    num_updates_per_train_iter=10,
    num_iterations=0,
    num_env_steps=int(1e7),
    num_checkpoints=1,
    evaluate=False,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summarize_first_interval=False,
    num_summaries=100)
