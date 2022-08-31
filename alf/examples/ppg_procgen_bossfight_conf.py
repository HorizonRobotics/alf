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
from alf.examples import ppg_conf
from alf.examples import procgen_conf
from alf.examples.networks import impala_cnn_encoder
from alf.utils.losses import element_wise_squared_loss
from alf.algorithms.ppg_algorithm import PPGAuxOptions

# Environment Configuration
alf.config(
    'create_environment', env_name='bossfight', num_parallel_environments=96)


def encoding_network_ctor(input_tensor_spec):
    encoder_output_size = 256
    return impala_cnn_encoder.create(
        input_tensor_spec=input_tensor_spec,
        cnn_channel_list=(16, 32, 32),
        num_blocks_per_stack=2,
        flatten_output_size=encoder_output_size)


# The PPG auxiliary replay buffer is typically large and does not fit in the GPU
# memory. As a result, for ``gather all()`` we set ``convert to default device``
# to ``False`` so that it does not have to put everything directly into GPU
# memory. Because of this, all data transformers should be created on "cpu" as
# they will be used while the experience is still in CPU memory.
alf.config('ReplayBuffer.gather_all', convert_to_default_device=False)
alf.config('data_transformer.create_data_transformer', device="cpu")

# The policy network and aux network is going to share the same
# encoder to save GPU memory. See
# https://github.com/HorizonRobotics/alf/issues/965#issuecomment-897950209
alf.config('DisjointPolicyValueNetwork', is_sharing_encoder=True)

alf.config(
    'PPGAlgorithm',
    encoding_network_ctor=encoding_network_ctor,
    policy_optimizer=alf.optimizers.AdamTF(lr=2e-4),
    aux_optimizer=alf.optimizers.AdamTF(lr=2e-4),
    aux_options=PPGAuxOptions(
        enabled=True,
        interval=32,
        mini_batch_length=None,  # None means use unroll_length as
        # mini_batch_length for aux phase
        mini_batch_size=8,
        num_updates_per_train_iter=6,
    ))

alf.config(
    'PPOLoss',
    compute_advantages_internally=True,
    entropy_regularization=0.01,
    gamma=0.999,
    td_lambda=0.95,
    td_loss_weight=0.5)

# Sample loss components from OpenAI's training:
#
# aux loss component: [pol_distance], weight: 1.0, unscaled: 0.0007583469850942492
# aux loss component: [vf_aux], weight: 1, unscaled: 0.44967320561408997
# aux loss component: [vf_true], weight: 1.0, unscaled: 0.46082180738449097
alf.config(
    'PPGAuxPhaseLoss',
    td_error_loss_fn=element_wise_squared_loss,
    policy_kl_loss_weight=1.0,
    gamma=0.999,
    td_lambda=0.95)

# training config
alf.config(
    'TrainerConfig',
    unroll_length=256,
    # This means that mini_batch_length will set to equal to the
    # length of the batches taken from the replay buffer, and in this
    # case it will be adjusted unroll_length.
    mini_batch_length=None,
    mini_batch_size=16,
    num_updates_per_train_iter=3,
    # Note that here 1000 iterations should already have a good
    # performance (reward = 10), while 6000 iterations brings it to
    # 12.
    num_iterations=6000,
    num_checkpoints=5,
    evaluate=True,
    eval_interval=50,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summary_interval=10)
