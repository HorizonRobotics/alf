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

import alf.examples.metadrive.base_conf
from alf.examples import ppg_conf

from alf.examples.networks import impala_cnn_encoder
from alf.utils.losses import element_wise_squared_loss
from alf.algorithms.ppg_algorithm import PPGAuxOptions, PPGAlgorithm
from alf.environments import suite_metadrive
from alf.networks import StableNormalProjectionNetwork, TruncatedProjectionNetwork, BetaProjectionNetwork

# Environment Configuration
alf.config('create_environment', num_parallel_environments=36)


def encoding_network_ctor(input_tensor_spec):
    encoder_output_size = 256
    return impala_cnn_encoder.create(
        input_tensor_spec=input_tensor_spec,
        cnn_channel_list=(16, 32, 32),
        num_blocks_per_stack=2,
        output_size=encoder_output_size)


alf.config('ReplayBuffer.gather_all', convert_to_default_device=False)

stable_normal_proj_net = partial(
    StableNormalProjectionNetwork,
    state_dependent_std=True,
    squash_mean=False,
    scale_distribution=True,
    min_std=1e-3,
    max_std=10.0)

# NOTE: replace stable_normal_proj_net with the other projection
alf.config(
    'DisjointPolicyValueNetwork',
    continuous_projection_net_ctor=stable_normal_proj_net,
    is_sharing_encoder=True)

alf.config(
    'PPGAlgorithm',
    encoding_network_ctor=encoding_network_ctor,
    policy_optimizer=alf.optimizers.AdamTF(lr=8e-5),
    aux_optimizer=alf.optimizers.AdamTF(lr=8e-5),
    aux_options=PPGAuxOptions(
        enabled=True,
        interval=32,
        mini_batch_length=None,  # None means use unroll_length as
        # mini_batch_length for aux phase
        mini_batch_size=18,
        num_updates_per_train_iter=6,
    ))

alf.config(
    'PPOLoss',
    compute_advantages_internally=True,
    entropy_regularization=0.01,
    gamma=0.999,
    td_lambda=0.95,
    td_loss_weight=0.5)

alf.config(
    'PPGAuxPhaseLoss',
    td_error_loss_fn=element_wise_squared_loss,
    policy_kl_loss_weight=1.0,
    gamma=0.999,
    td_lambda=0.95)

# training config
alf.config(
    'TrainerConfig',
    unroll_length=64,
    # This means that mini_batch_length will set to equal to the
    # length of the batches taken from the replay buffer, and in this
    # case it will be adjusted unroll_length.
    mini_batch_length=None,
    mini_batch_size=18,
    num_updates_per_train_iter=3,
    num_iterations=4000,
    num_checkpoints=20,
    evaluate=False,
    eval_interval=50,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summarize_action_distributions=True,
    summary_interval=40)
