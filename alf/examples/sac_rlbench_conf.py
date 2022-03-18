# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.algorithms.data_transformer import (
    RewardNormalizer, ObservationNormalizer, ImageScaleTransformer)
from alf.algorithms.td_loss import TDLoss
from alf.examples import sac_conf, rlbench_conf
from alf.environments import suite_rlbench
from alf.optimizers import AdamTF
from alf.utils.math_ops import clipped_exp

alf.config(
    "create_environment",
    env_name="reach_target_dense-v0",
    env_load_fn=suite_rlbench.load,
    num_parallel_environments=1)

hidden_layers = (256, ) * 2

actor_distribution_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    input_preprocessors=alf.layers.Detach(),
    fc_layer_params=hidden_layers,
    continuous_projection_net_ctor=partial(
        alf.networks.BetaProjectionNetwork, min_concentration=1.))

critic_network_cls = partial(
    alf.networks.CriticNetwork, joint_fc_layer_params=hidden_layers)

alf.config(
    'SacAlgorithm',
    actor_network_cls=actor_distribution_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.005,
    target_update_period=1,
    use_entropy_reward=False,
    critic_loss_ctor=TDLoss)

alf.config('calc_default_target_entropy', min_prob=0.2)

# Need to configure this before alf.get_observation_spec()
alf.config(
    "TrainerConfig",
    data_transformer_ctor=[
        partial(ImageScaleTransformer, fields=rlbench_conf.get_sensors('rgb')),
        partial(RewardNormalizer, clip_value=5.0),
        partial(
            ObservationNormalizer,
            clipping=5.,
            fields=(rlbench_conf.get_sensors('state')
                    ))  # This is needed for training
    ])

alf.config(
    'Agent',
    representation_learner_cls=rlbench_conf.create_encoding_algorithm_ctor(
        preproc_encoding_dim=512,
        fc_layer_params=(256, ),
        activation=torch.relu_),
    optimizer=AdamTF(lr=3e-4))

alf.config(
    "TrainerConfig",
    temporally_independent_train_step=True,
    initial_collect_steps=10000,
    unroll_length=1,
    mini_batch_length=2,
    mini_batch_size=128,
    num_updates_per_train_iter=1,
    num_env_steps=int(5e6),
    num_iterations=0,
    num_checkpoints=5,
    debug_summaries=True,
    summarize_grads_and_vars=1,
    num_summaries=200,
    replay_buffer_length=60000)
