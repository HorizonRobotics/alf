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
from alf.algorithms.data_transformer import RewardNormalizer, ImageScaleTransformer

import alf.environments.safe_car_racing
from alf.environments.gym_wrappers import FrameGrayScale
from alf.algorithms.lagrangian_reward_weight_algorithm import LagrangianRewardWeightAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.networks import BetaProjectionNetwork
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.algorithms.td_loss import TDLoss
from alf.algorithms.encoding_algorithm import EncodingAlgorithm
from alf.utils import math_ops

from alf.examples import sac_conf


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
    initial_collect_steps = 1000
else:
    num_envs = 16
    initial_collect_steps = 50000

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
#latent_size2x = 360
alf.config(
    "TrainerConfig",
    data_transformer_ctor=[
        partial(ImageScaleTransformer, min=0., max=1., fields=['rgb']),
        partial(RewardNormalizer, clip_value=10.)
    ])

activation = torch.tanh
actor_network_ctor = partial(
    ActorDistributionNetwork,
    fc_layer_params=(latent_size, ) * 2,
    activation=activation,
    continuous_projection_net_ctor=proj_net)
actor_network_cls = partial(
    actor_network_ctor, input_preprocessors=alf.layers.Detach())
critic_network_cls = partial(
    CriticNetwork,
    activation=activation,
    joint_fc_layer_params=(latent_size, ) * 2)

alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.005,
    target_update_period=1,
    critic_loss_ctor=TDLoss)

alf.config('calc_default_target_entropy', min_prob=0.1)

alf.config(
    'LagrangianRewardWeightAlgorithm',
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
    rl_algorithm_cls=SacAlgorithm,
    optimizer=alf.optimizers.AdamTF(lr=learning_rate),
    reward_weight_algorithm_cls=LagrangianRewardWeightAlgorithm)

alf.config("summarize_variables", with_histogram=False)
alf.config("summarize_gradients", with_histogram=False)

# training config
alf.config(
    'TrainerConfig',
    temporally_independent_train_step=True,
    initial_collect_steps=initial_collect_steps,
    mini_batch_length=8,  # must use a large length
    unroll_length=5,
    mini_batch_size=256,
    num_updates_per_train_iter=1,
    num_iterations=0,
    num_env_steps=int(5e6),
    num_checkpoints=1,
    evaluate=False,
    num_evals=100,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summarize_first_interval=False,
    num_summaries=100,
    replay_buffer_length=100000)
