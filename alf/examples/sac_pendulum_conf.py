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

import alf
alf.import_config("sac_conf.py")
from alf.environments import suite_gym
from alf.utils.losses import element_wise_squared_loss

import alf.utils.math_ops

# environment config
alf.config(
    'create_environment',
    env_load_fn=suite_gym.load,
    env_name="Pendulum-v0",
    num_parallel_environments=1)

# algorithm config
alf.config("ActorDistributionNetwork", fc_layer_params=(100, 100))

alf.config(
    "NormalProjectionNetwork",
    state_dependent_std=True,
    scale_distribution=True,
    std_transform=alf.utils.math_ops.clipped_exp)

alf.config("CriticNetwork", joint_fc_layer_params=(100, 100))

alf.config(
    "SacAlgorithm",
    actor_optimizer=alf.optimizers.Adam(lr=5e-4),
    critic_optimizer=alf.optimizers.Adam(lr=5e-4),
    alpha_optimizer=alf.optimizers.Adam(lr=5e-4),
    target_update_tau=0.005,
)

alf.config("OneStepTDLoss", td_error_loss_fn=element_wise_squared_loss)

# training config
alf.config(
    "TrainerConfig",
    initial_collect_steps=1000,
    mini_batch_length=2,
    unroll_length=1,
    mini_batch_size=256,
    num_updates_per_train_iter=1,
    num_iterations=10000,
    num_checkpoints=1,
    evaluate=True,
    profiling=False,
    eval_interval=0,
    num_evals=20,
    save_checkpoint_for_best_eval=alf.trainers.evaluator.BestEvalChecker(),
    summarize_grads_and_vars=True,
    summarize_action_distributions=True,
    debug_summaries=True,
    summary_interval=100,
    replay_buffer_length=100000)

alf.config("ReplayBuffer", enable_checkpoint=True)
alf.config('summarize_gradients', with_histogram=False)
alf.config('summarize_variables', with_histogram=False)
