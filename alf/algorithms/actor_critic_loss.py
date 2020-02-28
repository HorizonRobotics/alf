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

from collections import namedtuple

import gin
import torch
import torch.nn as nn
import alf
from alf.data_structures import TrainingInfo, LossInfo
from alf.utils.losses import element_wise_squared_loss
from alf.utils import tensor_utils, dist_utils, value_ops

ActorCriticLossInfo = namedtuple("ActorCriticLossInfo",
                                 ["pg_loss", "td_loss", "neg_entropy"])


def _normalize_advantages(advantages, variance_epsilon=1e-8):
    # advantages is of shape [rollout_steps, num_envs]
    # this function normalizes over all elements in the input advantages
    adv_mean = advantages.mean()
    adv_var = torch.var(
        advantages.view(-1, 1), dim=0, unbiased=False, keepdim=True)
    normalized_advantages = (
        (advantages - adv_mean) / (torch.sqrt(adv_var) + variance_epsilon))
    return normalized_advantages


@gin.configurable
class ActorCriticLoss(nn.Module):
    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 use_gae=False,
                 td_lambda=0.95,
                 use_td_lambda_return=True,
                 normalize_advantages=False,
                 advantage_clip=None,
                 entropy_regularization=None,
                 td_loss_weight=1.0,
                 debug_summaries=False,
                 name="ActorCriticLoss"):
        """Create a ActorCriticLoss object

        The total loss equals to
        (policy_gradient_loss
         + td_loss_weight * td_loss
         - entropy_regularization * entropy)

        Args:
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            use_gae (bool): If True, uses generalized advantage estimation for
                computing per-timestep advantage. Else, just subtracts value
                predictions from empirical return.
            use_td_lambda_return (bool): Only effective if use_gae is True.
                If True, uses td_lambda_return for training value function.
                (td_lambda_return = gae_advantage + value_predictions)
            td_lambda (float): Lambda parameter for TD-lambda computation.
            normalize_advantages (bool): If True, normalize advantage to zero
                mean and unit variance within batch for caculating policy
                gradient. This is commonly used for PPO.
            advantage_clip (float): If set, clip advantages to [-x, x]
            entropy_regularization (float): Coefficient for entropy
                regularization loss term.
            td_loss_weight (float): the weigt for the loss of td error.
        """
        super().__init__()

        self._td_loss_weight = td_loss_weight
        self._name = name
        self._gamma = gamma
        self._td_error_loss_fn = td_error_loss_fn
        self._use_gae = use_gae
        self._lambda = td_lambda
        self._use_td_lambda_return = use_td_lambda_return
        self._normalize_advantages = normalize_advantages
        assert advantage_clip is None or advantage_clip > 0, (
            "Clipping value should be positive!")
        self._advantage_clip = advantage_clip
        self._entropy_regularization = entropy_regularization
        self._debug_summaries = debug_summaries

    def forward(self, training_info: TrainingInfo, value):
        """Cacluate actor critic loss

        The first dimension of all the tensors is time dimension and the second
        dimesion is the batch dimension.

        Args:
            training_info (TrainingInfo): training_info collected by
                OnPolicyDriver/OffPolicyAlgorithm. All tensors in training_info
                are time-major
            value (torch.Tensor): the time-major tensor for the value at each time
                step
        Returns:
            loss_info (LossInfo): with loss_info.extra being ActorCriticLossInfo
        """

        returns, advantages = self._calc_returns_and_advantages(
            training_info, value)

        if self._debug_summaries and alf.summary.should_record_summaries():
            alf.summary.scalar(self._name + '/values', value.mean())
            alf.summary.scalar(self._name + "/returns", returns.mean())
            alf.summary.scalar(self._name + "/advantages/mean",
                               advantages.mean())
            alf.summary.histogram(self._name + "/advantages/value", advantages)
            alf.summary.histogram(
                self._name + "/explained_variance_of_return_by_value",
                tensor_utils.explained_variance(value, returns))

        if self._normalize_advantages:
            advantages = _normalize_advantages(advantages)

        if self._advantage_clip:
            advantages = torch.clamp(advantages, -self._advantage_clip,
                                     self._advantage_clip)

        pg_loss = self._pg_loss(training_info, advantages.detach())

        td_loss = self._td_error_loss_fn(returns.detach(), value)

        loss = pg_loss + self._td_loss_weight * td_loss

        entropy_loss = ()
        if self._entropy_regularization is not None:
            entropy = dist_utils.compute_entropy(
                training_info.info.action_distribution)
            entropy_loss = -entropy
            loss -= self._entropy_regularization * entropy

        return LossInfo(
            loss=loss,
            extra=ActorCriticLossInfo(
                td_loss=td_loss, pg_loss=pg_loss, neg_entropy=entropy_loss))

    def _pg_loss(self, training_info, advantages):
        action_log_prob = dist_utils.compute_log_probability(
            training_info.info.action_distribution, training_info.action)
        return -advantages * action_log_prob

    def _calc_returns_and_advantages(self, training_info, value):
        returns = value_ops.discounted_return(
            rewards=training_info.reward,
            values=value,
            step_types=training_info.step_type,
            discounts=training_info.discount * self._gamma)
        returns = tensor_utils.tensor_extend(returns, value[-1])

        if not self._use_gae:
            advantages = returns - value
        else:
            advantages = value_ops.generalized_advantage_estimation(
                rewards=training_info.reward,
                values=value,
                step_types=training_info.step_type,
                discounts=training_info.discount * self._gamma,
                td_lambda=self._lambda)
            advantages = tensor_utils.tensor_extend_zero(advantages)
            if self._use_td_lambda_return:
                returns = advantages + value

        return returns, advantages
