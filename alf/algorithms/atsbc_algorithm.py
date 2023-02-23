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
"""Actor and Thompson Sampling Bayesian Critic Algorithm."""

import numpy as np
import functools
import torch
import torch.distributions as td
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.algorithms.actor_bayes_critic_algorithm import AbcAlgorithm, ignore
from alf.algorithms.actor_bayes_critic_algorithm import AbcActionState
from alf.algorithms.actor_bayes_critic_algorithm import AbcState
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.optimizers import AdamTF
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork
from alf.networks.param_networks import CriticDistributionParamNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops, summary_utils


@alf.configurable
class AtsbcAlgorithm(AbcAlgorithm):
    r"""Actor and Thompson Sampling Bayesian Critic Algorithm. """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 explore_network_cls=None,
                 num_explore_action_samples=10,
                 critic_module_cls=FuncParVIAlgorithm,
                 deterministic_actor=False,
                 deterministic_critic=False,
                 basin_wise_ts_critic=False,
                 reward_weights=None,
                 critic_training_weight=1.0,
                 epsilon_greedy=None,
                 use_entropy_reward=False,
                 use_q_mean_train_actor=True,
                 use_basin_mean_for_target_critic=True,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
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
                 name="AtsbcAlgorithm"):
        """
        Args:
            critic_module_cls
            deterministic_actor
            deterministic_critic
            beta_lb
            epistemic_alpha_coeff (float|None): if not None, use epistemic_std 
                to the power of epistemic_alpha_coeff as alpha weights.
            explore_optimizer
            explore_alpha_optimizer
        """
        assert not deterministic_actor, "The target policy should be stochastic!"
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            explore_network_cls=None,
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

        self._idx = 0
        self._num_explore_action_samples = num_explore_action_samples
        self._basin_wise_ts_critic = basin_wise_ts_critic
        if basin_wise_ts_critic:
            self._num_ts_critics = self._critic_module.num_basins
        else:
            self._num_ts_critics = self._critic_module.num_particles

    def _predict_action(self,
                        observation,
                        state: AbcActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False,
                        train=False):
        if explore:
            assert not train, ("Explore_network is not maintained in Atsbc!")

        action_dist, actor_network_state = self._actor_network(
            observation, state=state.actor_network)
        assert isinstance(action_dist, td.TransformedDistribution), (
            "Squashed distribution is expected from actor_network.")
        new_state = AbcActionState(actor_network=actor_network_state)

        if explore:
            if self._training_started:
                # use optimistic Q-values to select action from
                # explore_networks's outputs
                action = action_dist.sample(
                    sample_shape=[self._num_explore_action_samples])
                critic_observation = observation.unsqueeze(
                    0).repeat_interleave(
                        self._num_explore_action_samples, dim=0)
                critic_action = action.reshape(
                    action.shape[0] * action.shape[1], *action.shape[2:])
                critic_observation = critic_observation.reshape(
                    critic_observation.shape[0] * critic_observation.shape[1],
                    *critic_observation.shape[2:])
                critic_step = self._critic_module.predict_step(
                    inputs=(critic_observation, critic_action),
                    state=state.critic)
                critics_state = critic_step.state
                if self._deterministic_critic:
                    critics = critic_step.output
                else:
                    critics_dist = critic_step.output
                    critics = critics_dist.mean  # [num_actions*bs, num_critics, 1]
                critics_info = critic_step.info

                if self._basin_wise_ts_critic and hasattr(
                        critics_info, "basin_means"):
                    if not ignore(critics_info.basin_means):
                        critics = critics_info.basin_means

                action_q = critics[:, self._idx, :]
                action_q = action_q.reshape(action.shape[0], action.shape[1],
                                            *action_q.shape[1:])
                action_idx = action_q.squeeze(-1).max(dim=0)[1]
                batch_idx = torch.arange(action.shape[1]).type_as(action_idx)
                action = action[action_idx, batch_idx, :]
            else:
                # This uniform sampling during initial collect stage is
                # important since current explore_network is deterministic
                action = alf.nest.map_structure(
                    lambda spec: spec.sample(outer_dims=observation.shape[:1]),
                    self._action_spec)
                action_dist = ()
        else:
            if eps_greedy_sampling:
                action = dist_utils.epsilon_greedy_sample(
                    action_dist, epsilon_greedy)
            else:
                action = dist_utils.rsample_action_distribution(action_dist)

        return action_dist, action, new_state

    def rollout_step(self, inputs: TimeStep, state: AbcState):
        if inputs.step_type == StepType.FIRST:
            self._idx = torch.randint(self._num_ts_critics, ())
        return super().rollout_step(inputs, state)
