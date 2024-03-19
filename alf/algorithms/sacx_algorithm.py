# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Soft Actor Critic Algorithm."""

from absl import logging
import numpy as np
import functools
from enum import Enum

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable, Optional, Union

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
from alf.nest.utils import convert_device
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import QNetwork, QRNNNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops, summary_utils
from alf.utils import value_ops, tensor_utils
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.utils.schedulers import Scheduler
from alf.algorithms.sac_algorithm import ActionType, SacAlgorithm, SacInfo, SacState, SacCriticState, SacCriticInfo
from alf.utils.summary_utils import safe_mean_hist_summary


@alf.configurable
class SacXAlgorithm(SacAlgorithm):
    def rollout_step(self, inputs: TimeStep, state: SacState):
        assert not self._is_eval
        observation, new_state, info = self._repr_step("rollout", inputs,
                                                       state)
        action_dist, action, _, action_state = self._predict_action(
            observation,
            state=state.action,
            epsilon_greedy=1.0,
            eps_greedy_sampling=True,
            rollout=True)

        if self._target_repr_alg is not None:
            tgt_repr_step = self._target_repr_alg.predict_step(
                inputs, state.target_repr)
            target_observation = tgt_repr_step.output
            new_state = new_state._replace(target_repr=tgt_repr_step.state)
        else:
            target_observation = observation

        target_critics, target_critics_state = self._compute_critics(
            self._target_critic_networks,
            observation,
            action,
            state.critic.critics,
            replica_min=False,
            apply_reward_weights=False,
            add_value=True)

        assert self._act_type == ActionType.Continuous

        critic_state = SacCriticState(target_critics=target_critics_state)

        new_state = new_state._replace(
            critic=critic_state, action=action_state)
        return AlgStep(
            output=action,
            state=new_state,
            info=info._replace(
                reward=inputs.reward,
                step_type=inputs.step_type,
                discount=inputs.discount,
                action=action,
                critic=SacCriticInfo(target_critic=target_critics),
                action_distribution=action_dist))

    def _critic_train_step(self, observation, target_observation,
                           state: SacCriticState, rollout_info: SacInfo,
                           action, action_distribution):
        critics, critics_state = self._compute_critics(
            self._critic_networks,
            observation,
            rollout_info.action,
            state.critics,
            replica_min=False,
            apply_reward_weights=False,
            add_value=True)

        state = SacCriticState(critics=critics_state)
        info = SacCriticInfo(critics=critics)

        return state, info

    def train_step(self, inputs: TimeStep, state: SacState,
                   rollout_info: SacInfo):
        assert not self._is_eval
        self._training_started = True
        observation, new_state, info = self._repr_step("train", inputs, state,
                                                       rollout_info.repr)
        (action_distribution, action, critics,
         action_state) = self._predict_action(
             observation, state=state.action)

        log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                    action_distribution, action)

        if self._act_type == ActionType.Mixed:
            # For mixed type, add log_pi separately
            log_pi = type(self._action_spec)((sum(nest.flatten(log_pi[0])),
                                              sum(nest.flatten(log_pi[1]))))
        else:
            log_pi = sum(nest.flatten(log_pi))

        if self._prior_actor is not None:
            prior_step = self._prior_actor.train_step(inputs, ())
            log_prior = dist_utils.compute_log_probability(
                prior_step.output, action)
            log_pi = log_pi - log_prior

        actor_state, actor_loss, alphas = self._actor_train_step(
            observation, state.actor, action, critics, log_pi,
            action_distribution)
        critic_state, critic_info = self._critic_train_step(
            observation, observation, state.critic, rollout_info, action,
            action_distribution)
        if self._alpha_uncertainty_ratio == 0:
            alpha_loss = self._alpha_train_step(log_pi)
        else:
            alpha_loss = ()
        new_state = new_state._replace(
            action=action_state, actor=actor_state, critic=critic_state)
        info = info._replace(
            returns=rollout_info.returns,
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            action_distribution=action_distribution,
            actor=actor_loss,
            critic=critic_info,
            alpha_loss=alpha_loss,
            alpha=alphas,
            log_pi=log_pi,
            discounted_return=rollout_info.discounted_return)
        return AlgStep(action, new_state, info)

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        """Compute advantages and put it into exp.rollout_info."""

        loss = self._critic_losses[0]

        # The device of rollout_info can be different from the default device
        # when ReplayBuffer.gather_all.convert_to_default_device is configured
        # to False to save gpu memory.
        step_type = convert_device(rollout_info.step_type)
        discount = convert_device(rollout_info.discount)
        reward = convert_device(rollout_info.reward)
        value = convert_device(rollout_info.critic.target_critic)

        if rollout_info.reward.ndim == 3:
            # [B, T, D] or [B, T, 1]
            discounts = discount.unsqueeze(-1) * loss.gamma
        else:
            # [B, T]
            discounts = discount * loss.gamma

        # expand the dimensions of the tensors to make them broadcastable
        # with value
        discounts = discounts[:, :, None, ...]
        reward = reward[:, :, None, ...]

        advantages = value_ops.generalized_advantage_estimation(
            rewards=reward,
            values=value,
            step_types=step_type,
            discounts=discounts,
            td_lambda=loss._lambda,
            time_major=False)
        advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)

        returns = value + advantages
        return root_inputs, rollout_info._replace(returns=returns)

    def _calc_critic_loss(self, info: SacInfo):
        assert not self._use_entropy_reward
        value = info.critic.critics
        returns = info.returns
        td_error = returns - value

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('values' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("td_error" + suffix, td, mask)

                for r in range(self._num_critic_replicas):
                    if value.ndim == 3:
                        _summarize(value[..., r], returns[..., r],
                                   td_error[..., r], f'/replica_{r}')
                    else:
                        for i in range(value.shape[3]):
                            suffix = f'/replica_{r}/{i}'
                            _summarize(value[..., r, i], returns[..., r, i],
                                       td_error[..., r, i], suffix)

        loss = self._critic_losses[0]._td_error_loss_fn(returns, value)
        loss = loss.reshape(*loss.shape[:2], -1).mean(-1)
        loss = tensor_utils.tensor_extend_zero(loss)
        td_error = tensor_utils.tensor_extend_zero(td_error)
        return LossInfo(loss=loss, extra=td_error)
