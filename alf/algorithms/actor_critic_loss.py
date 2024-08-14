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

import torch
import numpy as np

import alf
from alf.data_structures import LossInfo
from alf.utils.losses import element_wise_squared_loss
from alf.utils.summary_utils import safe_mean_hist_summary
from alf.utils import tensor_utils, dist_utils, value_ops
from .algorithm import Loss

ActorCriticLossInfo = namedtuple("ActorCriticLossInfo",
                                 ["pg_loss", "td_loss", "neg_entropy"])


def normalize(batch_norm, x):
    batch_norm.train()
    momentum = batch_norm.momentum
    if batch_norm.num_batches_tracked * momentum < 1.0:
        # For the first few batches, we do cumulative moving average
        batch_norm.momentum = None
    batch_norm(x)
    batch_norm.momentum = momentum
    # We use the running mean and variance of the advantages to normalize
    # since the batch may not be large enough to properly normalize within
    # the batch.
    batch_norm.eval()
    return batch_norm(x)


@alf.configurable
class ActorCriticLoss(Loss):
    def __init__(self,
                 reward_dim=1,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 use_gae=False,
                 td_lambda=0.95,
                 use_td_lambda_return=True,
                 normalize_advantages=False,
                 normalize_scalar_advantages=False,
                 running_stats_for_normalization=True,
                 advantage_norm_momentum=0.9,
                 advantage_clip=None,
                 entropy_regularization=None,
                 td_loss_weight=1.0,
                 debug_summaries=False,
                 name="ActorCriticLoss"):
        """An actor-critic loss equals to

        .. code-block:: python

            (policy_gradient_loss
            + td_loss_weight * td_loss
            - entropy_regularization * entropy)

        Args:
            reward_dim (int): dimension of the reward.
            gamma (float|list[float]): A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            use_gae (bool): If True, uses generalized advantage estimation for
                computing per-timestep advantage. Else, just subtracts value
                predictions from empirical return.
            use_td_lambda_return (bool): Only effective if use_gae is True.
                If True, uses ``td_lambda_return`` for training value function.
                ``(td_lambda_return = gae_advantage + value_predictions)``.
            td_lambda (float): Lambda parameter for TD-lambda computation.
            normalize_advantages (bool): If True, normalize advantage to zero
                mean and unit variance within batch for calculating policy
                gradient. This is commonly used for PPO.
            normalize_scalar_advantages (bool): If False, the normalization is
                performed for each reward dimension. If True, the normalization
                is performed for the weighted sum of advantages using reward_weights.
                Note that this will take precedence over `normalize_advantages`.
            running_stats_for_normalization (bool): If True, use running stats
                to normalize advantages.
            advantage_norm_momentum (float): Momentum for moving average of
                mean and variance of advantages (same as the momentum for nn.BatchNorm1d).
            advantage_clip (float): If set, clip advantages to :math:`[-x, x]`
            entropy_regularization (float): Coefficient for entropy
                regularization loss term.
            td_loss_weight (float): the weigt for the loss of td error.
        """
        super().__init__(name=name)

        self._td_loss_weight = td_loss_weight
        self._name = name
        self._gamma = torch.tensor(gamma)
        self._td_error_loss_fn = td_error_loss_fn
        self._use_gae = use_gae
        self._lambda = td_lambda
        self._use_td_lambda_return = use_td_lambda_return
        if normalize_scalar_advantages:
            self._adv_norm = torch.nn.BatchNorm1d(
                num_features=1,
                eps=1e-8,
                momentum=advantage_norm_momentum,
                affine=False,
                track_running_stats=running_stats_for_normalization)
            normalize_advantages = False
        elif normalize_advantages:
            self._adv_norm = torch.nn.BatchNorm1d(
                num_features=reward_dim,
                eps=1e-8,
                momentum=advantage_norm_momentum,
                affine=False,
                track_running_stats=running_stats_for_normalization)
        self._normalize_advantages = normalize_advantages
        self._normalize_scalar_advantages = normalize_scalar_advantages
        assert advantage_clip is None or advantage_clip > 0, (
            "Clipping value should be positive!")
        self._advantage_clip = advantage_clip
        self._entropy_regularization = entropy_regularization
        self._debug_summaries = debug_summaries

    @property
    def gamma(self):
        return self._gamma.clone()

    @property
    def normalizing_advantages(self):
        return self._normalize_advantages

    @property
    def normalizing_scalar_advantages(self):
        return self._normalize_scalar_advantages

    def forward(self, info):
        """Calculate actor critic loss. The first dimension of all the tensors is
        time dimension and the second dimension is the batch dimension.

        Args:
            info (namedtuple): information for calculating loss. All tensors are
                time-major. It should contain the following fields:
                - reward:
                - step_type:
                - discount:
                - action:
                - action_distribution:
                - value:
        Returns:
            LossInfo: with ``extra`` being ``ActorCriticLossInfo``.
        """

        value = info.value
        returns, advantages = self._calc_returns_and_advantages(info, value)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):

                def _summarize(v, r, adv, suffix):
                    alf.summary.scalar("values" + suffix, v.mean())
                    alf.summary.scalar("returns" + suffix, r.mean())
                    safe_mean_hist_summary('advantages' + suffix, adv)
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r))

                if value.ndim == 2:
                    _summarize(value, returns, advantages, '')
                else:
                    for i in range(value.shape[2]):
                        suffix = '/' + str(i)
                        _summarize(value[..., i], returns[..., i],
                                   advantages[..., i], suffix)
        if self._normalize_advantages:
            if hasattr(info, "normalized_advantages"):
                advantages = info.normalized_advantages
            else:
                bt = advantages.shape[0] * advantages.shape[1]
                adv = normalize(self._adv_norm, advantages.reshape(bt, -1))
                advantages = adv.reshape_as(advantages)
        elif self._normalize_scalar_advantages:
            if hasattr(info, "normalized_advantages"):
                advantages = info.normalized_advantages
            else:
                advantages = (advantages * info.reward_weights).sum(-1)
                adv = normalize(self._adv_norm, advantages.reshape(-1, 1))
                advantages = adv.reshape_as(advantages)

        if self._advantage_clip:
            advantages = torch.clamp(advantages, -self._advantage_clip,
                                     self._advantage_clip)

        if info.reward_weights != () and not self._normalize_scalar_advantages:
            # reward_weights has already been applied for self._normalize_scalar_advantages
            advantages = (advantages * info.reward_weights).sum(-1)
        pg_loss = self._pg_loss(info, advantages.detach())

        td_loss = self._td_error_loss_fn(returns.detach(), value)

        if td_loss.ndim == 3:
            td_loss = td_loss.mean(dim=2)

        loss = pg_loss + self._td_loss_weight * td_loss

        entropy_loss = ()
        if self._entropy_regularization is not None:
            entropy, entropy_for_gradient = dist_utils.entropy_with_fallback(
                info.action_distribution, return_sum=False)
            entropy_loss = alf.nest.map_structure(lambda x: -x, entropy)
            loss -= self._entropy_regularization * sum(
                alf.nest.flatten(entropy_for_gradient))

        return LossInfo(
            loss=loss,
            extra=ActorCriticLossInfo(
                td_loss=td_loss, pg_loss=pg_loss, neg_entropy=entropy_loss))

    def _pg_loss(self, info, advantages):
        action_log_prob = dist_utils.compute_log_probability(
            info.action_distribution, info.action)
        return -advantages * action_log_prob

    def _calc_returns_and_advantages(self, info, value):

        if info.reward.ndim == 3:
            # [T, B, D] or [T, B, 1]
            discounts = info.discount.unsqueeze(-1) * self._gamma
        else:
            # [T, B]
            discounts = info.discount * self._gamma

        returns = value_ops.discounted_return(
            rewards=info.reward,
            values=value,
            step_types=info.step_type,
            discounts=discounts)
        returns = tensor_utils.tensor_extend(returns, value[-1])

        if not self._use_gae:
            advantages = returns - value
        else:
            advantages = value_ops.generalized_advantage_estimation(
                rewards=info.reward,
                values=value,
                step_types=info.step_type,
                discounts=discounts,
                td_lambda=self._lambda)
            advantages = tensor_utils.tensor_extend_zero(advantages)
            if self._use_td_lambda_return:
                returns = advantages + value

        return returns, advantages

    def calc_loss(self, info):
        return self(info)
