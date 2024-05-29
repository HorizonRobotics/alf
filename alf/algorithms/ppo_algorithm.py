# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""PPO algorithm."""

import functools
import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.algorithms.actor_critic_loss import normalize
from alf.data_structures import namedtuple, TimeStep
from alf.utils import value_ops, tensor_utils
from alf.nest.utils import convert_device

PPOInfo = namedtuple(
    "PPOInfo", [
        "step_type",
        "discount",
        "reward",
        "action",
        "rollout_log_prob",
        "rollout_action_distribution",
        "returns",
        "advantages",
        "action_distribution",
        "value",
        "reward_weights",
        "normalized_advantages",
    ],
    default_value=())


@alf.configurable
class PPOAlgorithm(ActorCriticAlgorithm):
    """PPO Algorithm.
    Implement the simplified surrogate loss in equation (9) of "Proximal
    Policy Optimization Algorithms" https://arxiv.org/abs/1707.06347

    It works with ``ppo_loss.PPOLoss``. It should have same behavior as
    `baselines.ppo2`.
    """

    @property
    def on_policy(self):
        return False

    def train_step(self, inputs: TimeStep, state, rollout_info):
        alg_step = self._rollout_step(inputs, state)
        return alg_step._replace(
            info=rollout_info._replace(
                step_type=alg_step.info.step_type,
                reward=alg_step.info.reward,
                discount=alg_step.info.discount,
                action_distribution=alg_step.info.action_distribution,
                value=alg_step.info.value,
                reward_weights=alg_step.info.reward_weights))

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        """Compute advantages and put it into exp.rollout_info."""

        # The device of rollout_info can be different from the default device
        # when ReplayBuffer.gather_all.convert_to_default_device is configured
        # to False to save gpu memory.
        step_type = convert_device(rollout_info.step_type)
        discount = convert_device(rollout_info.discount)
        reward = convert_device(rollout_info.reward)
        value = convert_device(rollout_info.value)

        if rollout_info.reward.ndim == 3:
            # [B, T, D] or [B, T, 1]
            discounts = discount.unsqueeze(-1) * self._loss.gamma
        else:
            # [B, T]
            discounts = discount * self._loss.gamma

        advantages = value_ops.generalized_advantage_estimation(
            rewards=reward,
            values=value,
            step_types=step_type,
            discounts=discounts,
            td_lambda=self._loss._lambda,
            time_major=False)

        if self._loss.normalizing_scalar_advantages:
            if self.has_multidim_reward():
                scalar_advantages = (advantages * self.reward_weights).sum(-1)
            else:
                scalar_advantages = advantages
            normalized_advantages = normalize(self._loss._adv_norm,
                                              scalar_advantages.reshape(-1, 1))
            normalized_advantages = normalized_advantages.reshape_as(
                scalar_advantages)
            normalized_advantages = tensor_utils.tensor_extend_zero(
                normalized_advantages, dim=1)
        elif self._loss.normalizing_advantages:
            bt = advantages.shape[0] * advantages.shape[1]
            normalized_advantages = normalize(self._loss._adv_norm,
                                              advantages.reshape(bt, -1))
            normalized_advantages = normalized_advantages.reshape_as(
                advantages)
            normalized_advantages = tensor_utils.tensor_extend_zero(
                normalized_advantages, dim=1)
        else:
            normalized_advantages = ()

        advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)
        returns = value + advantages
        return root_inputs, PPOInfo(
            rollout_action_distribution=rollout_info.action_distribution,
            rollout_log_prob=rollout_info.log_prob,
            returns=returns,
            action=rollout_info.action,
            advantages=advantages,
            normalized_advantages=normalized_advantages,
        )
