# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import math
import alf
from alf.environments import suite_gym, suite_dmc
import alf.algorithms.mpo_algorithm
import alf.utils.math_ops
from alf.utils import losses

# environment config
alf.config(
    'create_environment',
    # env_load_fn=suite_gym.load,
    # env_name="Pendulum-v0",
    env_load_fn=suite_dmc.load,
    env_name='cheetah:run',
    num_parallel_environments=1)

# algorithm config
alf.config(
    "BetaProjectionNetwork",
    min_concentration=0.5,
    bias_init_value=math.log(math.exp(0.5) - 1))

alf.config(
    "ActorDistributionNetwork",
    fc_layer_params=(256, 256, 256),
    continuous_projection_net_ctor=alf.networks.BetaProjectionNetwork)

num_quantiles = 255
alf.config(
    "CriticNetwork",
    joint_fc_layer_params=(256, 256, 256),
    output_tensor_spec=alf.TensorSpec((num_quantiles, )))

alf.config(
    "MPOAlgorithm",
    actor_optimizer=alf.optimizers.Adam(lr=5e-4, fused=True),
    critic_optimizer=alf.optimizers.Adam(lr=5e-4, fused=True),
    target_update_tau=0.01,
)

alf.config(
    "MPOLoss",
    action_weight_regulization=1.0,
    # value_loss=losses.SquareLoss())
    # value_loss=losses.QuantileRegressionLoss(
    #     transform=alf.math.Sqrt1pTransform(), inverse_after_mean=False))
    value_loss=losses.OrderedDiscreteRegressionLoss(
        transform=alf.math.Sqrt1pTransform(), inverse_after_mean=False))

# training config
alf.config(
    "TrainerConfig",
    algorithm_ctor=alf.algorithms.mpo_algorithm.MPOAlgorithm,
    initial_collect_steps=10000,
    mini_batch_length=2,
    unroll_length=1,
    mini_batch_size=256,
    num_updates_per_train_iter=1,
    num_iterations=1000000,
    num_checkpoints=1,
    evaluate=True,
    profiling=False,
    eval_interval=0,
    num_evals=20,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    confirm_checkpoint_upon_crash=False,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    summarize_action_distributions=True,
    summary_interval=1000,
    replay_buffer_length=1000000)

alf.config("ReplayBuffer", enable_checkpoint=True, device='cuda')
alf.config('summarize_gradients', with_histogram=False)
alf.config('summarize_variables', with_histogram=False)
