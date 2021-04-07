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

import alf
from alf.algorithms.data_transformer import (
    RewardNormalizer, ObservationNormalizer, FrameStacker,
    ImageScaleTransformer)
# Needs to install safety gym first:
# https://github.com/openai/safety-gym
from alf.environments import suite_safety_gym
from alf.environments.gym_wrappers import FrameGrayScale
from alf.algorithms.lagrangian_reward_weight_algorithm import LagrangianRewardWeightAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.merlin_algorithm import ResnetEncodingNetwork
from alf.environments.alf_wrappers import ActionObservationWrapper, RewardObservationWrapper
from alf.networks import NormalProjectionNetwork, BetaProjectionNetwork
from alf.networks import ActorDistributionNetwork, CriticNetwork, EncodingNetwork
from alf.utils import math_ops
from alf.nest.utils import NestSum, NestConcat
from alf.algorithms.td_loss import TDLoss
from alf.algorithms.encoding_algorithm import EncodingAlgorithm
from alf.algorithms.merlin_algorithm import ResnetEncodingNetwork

from alf.examples import sac_conf

env_name = "Safexp-CarButton2-v0"  # natural lidar
#env_name = "Safexp-CarPushVision1-v0" # two cameras

DEBUG = False

if DEBUG:
    num_envs = 8
    initial_collect_steps = 100
else:
    num_envs = 32
    initial_collect_steps = 10000

# environment config
alf.config(
    'create_environment',
    env_name=env_name,
    num_parallel_environments=num_envs,
    env_load_fn=suite_safety_gym.load)

alf.config(
    'suite_safety_gym.load',
    episodic=True,
    sparse_reward=False,
    #gym_env_wrappers=[partial(FrameGrayScale, fields=["vision"])],
)

n_frames = 8
use_bn = False

#alf.config("Beta", eps=1e-3)
proj_net = partial(
    BetaProjectionNetwork, min_concentration=1., max_concentration=None)

if not "Vision" in env_name:
    hidden_layers = (256, 256, 256)
    alf.config(
        "TrainerConfig",
        data_transformer_ctor=[partial(FrameStacker, stack_size=n_frames)])
else:  # RGB image
    from alf.algorithms.encoding_algorithm import EncodingAlgorithm
    hidden_layers = (256, 256)
    encoding_dim = 256
    # must be configured before first calling ``alf.env()``
    alf.config('ImageChannelFirst', fields=['vision'])
    # need to config the data transformer first to get the transformed obs_spec
    alf.config(
        "TrainerConfig",
        data_transformer_ctor=[
            # 8 frames is better than 4 frames with gremlins moving
            partial(FrameStacker, stack_size=n_frames, fields=['vision']),
            partial(ImageScaleTransformer, fields=['vision']),
            #partial(ObservationNormalizer, clipping=5., fields=['vision']),
        ])
    """
    trans_obs_spec
    {
        'observation': {
            'observation': {'vision': , 'robot': },
            'prev_action':
        },
        'prev_reward':
    }
    """
    trans_obs_spec = alf.get_observation_spec()
    #conv_layer_params = ((32, 8, 4), (32, 6, 3), (32, 3, 1)) # 160x160
    conv_layer_params = ((32, 8, 4), (64, 4, 2), (64, 3, 1))  # 84x84
    #conv_layer_params = ((32, 6, 3), (64, 4, 2), (64, 3, 1))  # 64x64
    obs_proc_fn = partial(
        alf.networks.EncodingNetwork,
        conv_layer_params=conv_layer_params,
        last_layer_size=encoding_dim,
        use_fc_bn=use_bn,
        last_activation=math_ops.identity)
    #obs_proc_fn = partial(
    #    ResnetEncodingNetwork,
    #    output_size=encoding_dim,
    #    output_activation=torch.relu_)
    fc_ctor = lambda in_size: torch.nn.Sequential(
        alf.layers.FC(in_size, encoding_dim, use_bn=use_bn))

    input_preprocessors = alf.nest.py_map_structure_with_path(
        lambda path, spec: (obs_proc_fn(input_tensor_spec=spec) if 'vision' in
                            path else fc_ctor(spec.numel)), trans_obs_spec)

    encoder_cls = partial(
        alf.networks.EncodingNetwork,
        input_preprocessors=input_preprocessors,
        preprocessing_combiner=NestSum(),
        fc_layer_params=(encoding_dim, ))
    alf.config('EncodingAlgorithm', encoder_cls=encoder_cls)
    alf.config('Agent', representation_learner_cls=EncodingAlgorithm)

actor_network_ctor = partial(
    ActorDistributionNetwork,
    fc_layer_params=hidden_layers,
    continuous_projection_net_ctor=proj_net)
actor_network_cls = partial(
    actor_network_ctor, input_preprocessors=alf.layers.Detach())
critic_network_cls = partial(
    CriticNetwork, joint_fc_layer_params=hidden_layers)

alf.config('TDLoss', gamma=[0.99, 0.95])
alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.005,
    target_update_period=1,
    critic_loss_ctor=TDLoss,
    use_parallel_network=True,
    use_entropy_reward=False)

alf.config('calc_default_target_entropy', min_prob=0.1)

alf.config(
    'LagrangianRewardWeightAlgorithm',
    init_weights=(1., 1.),
    reward_thresholds=[None, -0.002],
    max_weight=None,
    optimizer=alf.optimizers.AdamTF(lr=1e-2))

alf.config(
    'Agent',
    rl_algorithm_cls=SacAlgorithm,
    optimizer=alf.optimizers.AdamTF(lr=3e-4),
    reward_weight_algorithm_cls=LagrangianRewardWeightAlgorithm)

# training config
alf.config(
    'TrainerConfig',
    temporally_independent_train_step=True,
    initial_collect_steps=initial_collect_steps,
    mini_batch_length=8,
    unroll_length=5,
    mini_batch_size=1024,
    num_updates_per_train_iter=1,
    num_iterations=0,
    num_env_steps=int(1e7),
    num_checkpoints=5,
    evaluate=False,
    num_evals=100,
    debug_summaries=True,
    num_summaries=100,
    replay_buffer_length=50000)
