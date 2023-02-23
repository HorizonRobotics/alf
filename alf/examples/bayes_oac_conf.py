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

import alf
from alf.algorithms.bayes_oac_algorithm import BayesOacAlgorithm
from alf.algorithms.multi_bootstrap_ensemble import MultiBootstrapEnsemble
from alf.algorithms.multiswag_algorithm import MultiSwagAlgorithm
from alf.networks import ActorNetwork, ActorDistributionNetwork
from alf.networks import NormalProjectionNetwork
from alf.optimizers import Adam, AdamTF
from alf.utils.losses import element_wise_squared_loss
from alf.utils.math_ops import clipped_exp

from alf.examples import sac_conf

# experiment settings
use_multibootstrap = False
deterministic_actor = False
deterministic_critic = False

# environment config
alf.config(
    'create_environment',
    env_name="HalfCheetah-v2",
    num_parallel_environments=1)

# algorithm config
fc_layer_params = (256, 256)
joint_fc_layer_params = (256, 256)
deterministic_actor = False

if deterministic_actor:
    actor_network_cls = partial(ActorNetwork, fc_layer_params=fc_layer_params)
else:
    actor_network_cls = partial(
        ActorDistributionNetwork,
        fc_layer_params=fc_layer_params,
        continuous_projection_net_ctor=partial(
            NormalProjectionNetwork,
            state_dependent_std=True,
            scale_distribution=True,
            std_transform=clipped_exp))

alf.config('calc_default_target_entropy', min_prob=0.184)

alf.config(
    'CriticDistributionParamNetwork',
    joint_fc_layer_params=joint_fc_layer_params,
    state_dependent_std=True)

if use_multibootstrap:
    critic_module_cls = MultiBootstrapEnsemble
    alf.config(
        'MultiBootstrapEnsemble',
        num_basins=5,  # grid search
        num_particles_per_basin=4,  # grid search
        mask_sample_size=128,
        initial_train_steps=100)
    batch_size = 256
else:
    critic_module_cls = MultiSwagAlgorithm
    alf.config(
        'MultiSwagAlgorithm',
        num_particles=10,
        num_samples_per_model=1,
        subspace_max_rank=20,
        subspace_after_update_steps=10000)
    batch_size = 256

alf.config('CovarianceSpace', use_subspace_mean=True)

alf.config(
    'BayesOacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_module_cls=critic_module_cls,
    beta_ub=4.66,
    beta_lb=1.,
    explore_delta=6.86,
    initial_log_alpha=math.log(0.05),
    # entropy_regularization_weight=1.,
    deterministic_actor=deterministic_actor,
    deterministic_critic=False,
    deterministic_explorer=False,
    critic_training_weight=None,
    common_td_target=True,  # grid search
    use_q_mean_train_actor=True,
    use_entropy_reward=False,
    use_epistemic_alpha=False,
    use_basin_mean_for_target_critic=False,
    target_update_tau=0.005,
    actor_optimizer=AdamTF(lr=3e-4),
    explore_optimizer=None,
    critic_optimizer=Adam(lr=3e-4),  #, weight_decay=1e-4),
    alpha_optimizer=None,
    explore_alpha_optimizer=None)

alf.config('OneStepTDLoss', td_error_loss_fn=element_wise_squared_loss)

# training config
alf.config('Agent', rl_algorithm_cls=BayesOacAlgorithm)

alf.config(
    'TrainerConfig',
    version='normal',
    use_wandb=True,
    async_eval=True,
    entity="runjerry",
    project="Actor-Bayes-Critic",
    initial_collect_steps=10000,
    mini_batch_length=2,
    unroll_length=1,
    mini_batch_size=batch_size,
    num_updates_per_train_iter=1,
    num_iterations=2500000,
    num_checkpoints=1,
    evaluate=True,
    eval_interval=1000,
    num_eval_episodes=5,
    debug_summaries=True,
    random_seed=1,
    summarize_grads_and_vars=True,
    summary_interval=1000,
    replay_buffer_length=int(1e6))
