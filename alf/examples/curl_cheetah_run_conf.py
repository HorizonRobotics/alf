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
from functools import partial
from alf.algorithms.agent import Agent
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.curl_encoder import curl_encoder
from alf.networks import (NormalProjectionNetwork, ActorDistributionNetwork,
                          CriticNetwork)
from alf.algorithms.data_transformer import FrameStacker, ImageScaleTransformer, RewardClipping
from alf.utils.math_ops import clipped_exp
from alf.optimizers import Adam
from alf.utils.dist_utils import calc_default_target_entropy
from alf.utils.losses import element_wise_squared_loss
from alf.environments.suite_dmc2gym import dmc2gym_loader
import math

seed = 1

alf.config(
    'create_environment',
    env_name='dmc2gym',
    env_load_fn=dmc2gym_loader,
    num_parallel_environments=1,
    nonparallel=True,
    seed=seed)

actor_network_cls = partial(
    ActorDistributionNetwork,
    fc_layer_params=(256, 256),
    continuous_projection_net_ctor=partial(
        NormalProjectionNetwork,
        state_dependent_std=True,
        squash_mean=False,
        scale_distribution=True,
        std_transform=clipped_exp))

critic_network_cls = partial(
    CriticNetwork,
    use_naive_parallel_network=True,
    joint_fc_layer_params=(256, 256))

alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    actor_optimizer=Adam(lr=2e-4),
    critic_optimizer=Adam(lr=2e-4),
    alpha_optimizer=Adam(lr=1e-4, betas=(0.5, 0.999)),  ##change as test 3
    target_entropy=partial(calc_default_target_entropy, min_prob=0.1),
    target_update_tau=0.01,
    initial_log_alpha=math.log(0.1),
    target_update_period=2)
alf.config(
    'OneStepTDLoss', gamma=0.99, td_error_loss_fn=element_wise_squared_loss)

alf.config(
    'curl_encoder',
    feature_dim=50,
    crop_size=84,
    optimizer=Adam(lr=2e-4),
    save_image=False,
    use_pytorch_randcrop=True)

alf.config(
    'Agent',
    rl_algorithm_cls=SacAlgorithm,
    representation_learner_cls=curl_encoder)

alf.config('FrameStacker', stack_size=3)

alf.config(
    'TrainerConfig',
    initial_collect_steps=1000,
    mini_batch_length=2,
    unroll_length=1,
    mini_batch_size=512,
    num_updates_per_train_iter=1,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    num_iterations=1000000,
    num_checkpoints=20,
    evaluate=False,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=100,
    replay_buffer_length=100000,
    algorithm_ctor=Agent,
    profiling=False,
    data_transformer_ctor=[FrameStacker, ImageScaleTransformer],
)
