# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.environments import suite_gym
from alf.utils.losses import element_wise_squared_loss
from alf.algorithms.agent import Agent
from alf.algorithms.traj_ppo_algorithm import TrajectoryPPOAlgorithm, PPOLoss
from alf.algorithms.ppo_algorithm import PPOAlgorithm
import alf.utils.math_ops
from alf.environments.alf_wrappers import TrajectoryActionWrapper, RandomFirstEpisodeLength

trajectory_length = 10

alf.config("RandomFirstEpisodeLength", random_length_range=200)
alf.config("suite_gym.load", alf_env_wrappers=(RandomFirstEpisodeLength, ))

# environment config
alf.config(
    'create_environment',
    env_load_fn=suite_gym.load,
    env_name="Pendulum-v0",
    num_parallel_environments=64,
    batch_size_per_env=4,
    eval_batch_size_per_env=1,
    batched_wrappers=[
        partial(TrajectoryActionWrapper, trajectory_length=trajectory_length)
    ])

alf.config(
    'PPOLoss',
    check_numerics=True,
    entropy_regularization=0.01,
    normalize_scalar_advantages=True,
    gamma=0.99,
    advantage_norm_momentum=0.9,
    importance_ratio_clipping=0.2,
    td_lambda=0.95,
    td_loss_weight=0.5)

alf.config(
    "NormalProjectionNetwork",
    state_dependent_std=True,
    scale_distribution=True,
    std_transform=alf.utils.math_ops.clipped_exp)

alf.config("BetaProjectionNetwork", min_concentration=1.0)

optimizer = alf.optimizers.Adam(
    lr=5e-4,
    betas=(0.9, 0.999),
    clip_by_global_norm=True,
    gradient_clipping=100.0)

alf.config("ActorDistributionNetwork", fc_layer_params=(100, 100))
alf.config("CriticNetwork", joint_fc_layer_params=(100, 100))
alf.config("ValueNetwork", fc_layer_params=(100, 100))
alf.config(
    "RNNARModel",
    cell_ctor=alf.nn.GRUCell,
    hidden_sizes=[100, 100],
    projection_net_ctor=alf.nn.BetaProjectionNetwork)
alf.config(
    "TrajectoryPPOAlgorithm",
    trajectory_length=trajectory_length,
    target_switch_steps=10)

alf.config(
    'Agent', rl_algorithm_cls=TrajectoryPPOAlgorithm, optimizer=optimizer)

# alf.config("PPOAlgorithm", loss_class=PPOLoss)
# alf.config(
#     'Agent', rl_algorithm_cls=PPOAlgorithm, optimizer=optimizer)

# training config
alf.config(
    "TrainerConfig",
    algorithm_ctor=Agent,
    whole_replay_buffer_training=True,
    clear_replay_buffer=True,
    mini_batch_length=1,
    unroll_length=16,
    mini_batch_size=1024,
    num_updates_per_train_iter=8,
    num_iterations=1000,
    num_checkpoints=1,
    use_rollout_state=True,
    evaluate=True,
    profiling=False,
    eval_interval=0,
    num_evals=20,
    confirm_checkpoint_upon_crash=False,
    random_seed=1,
    summarize_grads_and_vars=True,
    summarize_action_distributions=True,
    debug_summaries=True,
    summary_interval=10,
    replay_buffer_length=100)

alf.config("ReplayBuffer", enable_checkpoint=False)
alf.config('summarize_gradients', with_histogram=False)
alf.config('summarize_variables', with_histogram=False)
