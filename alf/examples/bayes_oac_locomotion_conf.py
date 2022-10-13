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

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.bayes_oac_algorithm import BayesOacAlgorithm
from alf.algorithms.multi_bootstrap_ensemble import MultiBootstrapEnsemble
from alf.algorithms.multiswag_algorithm import MultiSwagAlgorithm
from alf.examples.benchmarks.locomotion import locomotion_conf
from alf.networks import ActorNetwork
from alf.optimizers import AdamTF

# experiment settings
use_multibootstrap = False
deterministic_actor = False
deterministic_critic = False

if deterministic_actor:
    actor_network_cls = partial(
        ActorNetwork, fc_layer_params=locomotion_conf.hidden_layers)
else:
    actor_network_cls = locomotion_conf.actor_distribution_network_cls

alf.config(
    'CriticDistributionParamNetwork',
    joint_fc_layer_params=locomotion_conf.hidden_layers,
    state_dependent_std=True)

if use_multibootstrap:
    critic_module_cls = MultiBootstrapEnsemble
    alf.config(
        'MultiBootstrapEnsemble',
        num_basins=5,  # grid search
        num_particles_per_basin=4,  # grid search
        mask_sample_ratio=0.5,
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

alf.config('Agent', rl_algorithm_cls=BayesOacAlgorithm)
# optimizer=locomotion_conf.optimizer)

alf.config(
    'BayesOacAlgorithm',
    actor_network_cls=actor_network_cls,
    critic_module_cls=critic_module_cls,
    beta_ub=4.66,
    beta_lb=1.,
    explore_delta=6.86,
    deterministic_actor=deterministic_actor,
    deterministic_critic=False,
    deterministic_explore=False,
    critic_training_weight=None,
    common_td_target=True,  # grid search
    use_q_mean_train_actor=True,
    use_entropy_reward=False,
    use_epistemic_alpha=False,
    use_basin_mean_for_target_critic=False,
    actor_optimizer=AdamTF(lr=3e-4),
    critic_optimizer=AdamTF(lr=3e-4),
    alpha_optimizer=AdamTF(lr=3e-4),
    target_update_tau=0.005)

alf.config('calc_default_target_entropy', min_prob=0.184)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    summarize_gradient_noise_scale=True,
    summarize_action_distributions=True)
