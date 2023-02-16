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
"""Optimistic Actor and Bayesian Critic Algorithm."""

import numpy as np
import functools
import torch
import torch.distributions as td
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.algorithms.actor_bayes_critic_algorithm import AbcAlgorithm
from alf.algorithms.actor_bayes_critic_algorithm import AbcActionState
from alf.algorithms.actor_bayes_critic_algorithm import AbcState
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep
from alf.optimizers import AdamTF
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork
from alf.networks.param_networks import CriticDistributionParamNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops, summary_utils


@alf.configurable
class OabcAlgorithm(AbcAlgorithm):
    r"""Soft Actor and Bayesian Critic Algorithm. """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 explore_network_cls=ActorDistributionNetwork,
                 critic_module_cls=FuncParVIAlgorithm,
                 deterministic_actor=False,
                 deterministic_critic=False,
                 reward_weights=None,
                 critic_training_weight=1.0,
                 epsilon_greedy=None,
                 use_entropy_reward=False,
                 use_q_mean_train_actor=True,
                 use_basin_mean_for_target_critic=True,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 beta_ub=1.,
                 beta_lb=1.,
                 common_td_target: bool = True,
                 entropy_regularization_weight=1.,
                 entropy_regularization=None,
                 target_entropy=None,
                 initial_log_alpha=0.0,
                 max_log_alpha=None,
                 epistemic_alpha_coeff=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 explore_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 explore_alpha_optimizer=None,
                 debug_summaries=False,
                 name="OabcAlgorithm"):
        """
        Args:
            explore_network_cls
            critic_module_cls
            deterministic_actor
            deterministic_critic
            beta_ub (float): parameter for computing the upperbound of Q value:
                :math:`Q_ub(s,a) = \mu_Q(s,a) + \beta_ub * \sigma_Q(s,a)`    
            beta_lb
            epistemic_alpha_coeff (float|None): if not None, use epistemic_std 
                to the power of epistemic_alpha_coeff as alpha weights.
            explore_optimizer
            explore_alpha_optimizer
        """
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            explore_network_cls=explore_network_cls,
            critic_module_cls=critic_module_cls,
            deterministic_actor=deterministic_actor,
            deterministic_critic=deterministic_critic,
            reward_weights=reward_weights,
            critic_training_weight=critic_training_weight,
            epsilon_greedy=epsilon_greedy,
            use_entropy_reward=use_entropy_reward,
            use_q_mean_train_actor=use_q_mean_train_actor,
            use_basin_mean_for_target_critic=use_basin_mean_for_target_critic,
            env=env,
            config=config,
            critic_loss_ctor=critic_loss_ctor,
            beta_ub=beta_ub,
            beta_lb=beta_lb,
            common_td_target=common_td_target,
            entropy_regularization_weight=entropy_regularization_weight,
            entropy_regularization=entropy_regularization,
            target_entropy=target_entropy,
            initial_log_alpha=initial_log_alpha,
            max_log_alpha=max_log_alpha,
            epistemic_alpha_coeff=epistemic_alpha_coeff,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            dqda_clipping=dqda_clipping,
            actor_optimizer=actor_optimizer,
            explore_optimizer=explore_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            explore_alpha_optimizer=explore_alpha_optimizer,
            debug_summaries=debug_summaries,
            name=name)

    def _predict_action(self,
                        observation,
                        state: AbcActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False,
                        train=False):
        del train
        new_state = AbcActionState()
        if explore:
            if self._training_started:
                # explore_network is deterministic
                action, explore_network_state = self._explore_network(
                    observation, state=state.explore_network)
                new_state = new_state._replace(
                    explore_network=explore_network_state)
            else:
                # This uniform sampling during initial collect stage is
                # important since current explore_network is deterministic
                action = alf.nest.map_structure(
                    lambda spec: spec.sample(outer_dims=observation.shape[:1]),
                    self._action_spec)
            action_dist = ()
        else:
            if self._deterministic_actor:
                action_dist = ()
                action, actor_network_state = self._actor_network(
                    observation, state=state.actor_network)
            else:
                action_dist, actor_network_state = self._actor_network(
                    observation, state=state.actor_network)

                if eps_greedy_sampling:
                    action = dist_utils.epsilon_greedy_sample(
                        action_dist, epsilon_greedy)
                else:
                    action = dist_utils.rsample_action_distribution(
                        action_dist)
            new_state = new_state._replace(actor_network=actor_network_state)

        return action_dist, action, new_state
