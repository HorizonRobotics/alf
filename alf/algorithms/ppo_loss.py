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
"""Loss for PPO algorithm."""

import torch

import alf

from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.utils.losses import element_wise_squared_loss
from alf.utils import value_ops


@alf.configurable
class PPOLoss(ActorCriticLoss):
    """PPO loss."""

    def __init__(self,
                 reward_dim=1,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 td_lambda=0.95,
                 normalize_advantages=True,
                 normalize_scalar_advantages=False,
                 advantage_norm_momentum=0.9,
                 compute_advantages_internally=False,
                 advantage_clip=None,
                 entropy_regularization=None,
                 td_loss_weight=1.0,
                 importance_ratio_clipping=0.2,
                 log_prob_clipping=0.0,
                 check_numerics=False,
                 debug_summaries=False,
                 name='PPOLoss'):
        """Implement the simplified surrogate loss in equation (9) of `Proximal
        Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_.

        The total loss equals to

        .. code-block:: python

            (policy_gradient_loss      # (L^{CLIP} in equation (9))
            + td_loss_weight * td_loss # (L^{VF} in equation (9))
            - entropy_regularization * entropy)

        This loss works with ``PPOAlgorithm``. The advantages and returns are
        pre-computed by ``PPOAlgorithm.preprocess()``. One known difference with
        `baselines.ppo2` is that value estimation is not clipped here, while
        `baselines.ppo2` also clipped value if it deviates from returns too
        much.

        Args:
            reward_dim (int): dimension of the reward.
            gamma (float|list[float]): A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            td_lambda (float): Lambda parameter for TD-lambda computation.
            normalize_advantages (bool): If True, normalize advantage to zero
                mean and unit variance within batch for calculating policy
                gradient.
            normalize_scalar_advantages (bool): If False, the normalization is
                performed for each reward dimension. If True, the normalization
                is performed for the weighted sum of advantages using reward_weights.
                Note that this will take precedence over `normalize_advantages`.
            advantage_norm_momentum (float): Momentum for moving average of
                mean and variance of advantages (same as the momentum for nn.BatchNorm1d).
            compute_advantages_internally (bool): Normally PPOLoss does not
                compute the advantage and it expects the info to carry the
                already-computed advantage. If this flag is set to True, PPOLoss
                will instead compute the advantage internally without depending
                on the input info, because loading very large amount of
                experiences into GPU memory to compute advantages may not always
                be possible.
            advantage_clip (float): If set, clip advantages to :math:`[-x, x]`
            entropy_regularization (float): Coefficient for entropy
                regularization loss term.
            td_loss_weight (float): the weigt for the loss of td error.
            importance_ratio_clipping (float):  Epsilon in clipped, surrogate
                PPO objective. See the cited paper for more detail.
            log_prob_clipping (float): If >0, clipping log probs to the range
                ``(-log_prob_clipping, log_prob_clipping)`` to prevent ``inf/NaN``
                values.
            check_numerics (bool):  If true, checking for ``NaN/Inf`` values. For
                debugging only.
            name (str):

        """

        super().__init__(
            reward_dim=reward_dim,
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            use_gae=True,
            td_lambda=td_lambda,
            use_td_lambda_return=True,
            normalize_advantages=normalize_advantages,
            normalize_scalar_advantages=normalize_scalar_advantages,
            advantage_norm_momentum=advantage_norm_momentum,
            advantage_clip=advantage_clip,
            entropy_regularization=entropy_regularization,
            td_loss_weight=td_loss_weight,
            debug_summaries=debug_summaries,
            name=name)

        self._importance_ratio_clipping = importance_ratio_clipping
        self._log_prob_clipping = log_prob_clipping
        self._check_numerics = check_numerics
        self._compute_advantages_internally = compute_advantages_internally

    def _pg_loss(self, info, advantages):
        scope = alf.summary.scope(self._name)
        importance_ratio, importance_ratio_clipped = value_ops.action_importance_ratio(
            action_distribution=info.action_distribution,
            rollout_action_distribution=info.rollout_action_distribution,
            action=info.action,
            rollout_log_prob=info.rollout_log_prob,
            clipping_mode='double_sided',
            scope=scope,
            importance_ratio_clipping=self._importance_ratio_clipping,
            log_prob_clipping=self._log_prob_clipping,
            check_numerics=self._check_numerics,
            debug_summaries=self._debug_summaries)
        # Pessimistically choose the maximum objective value for clipped and
        # unclipped importance ratios.
        pg_objective = -importance_ratio * advantages
        pg_objective_clipped = -importance_ratio_clipped * advantages
        policy_gradient_loss = torch.max(pg_objective, pg_objective_clipped)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with scope:
                alf.summary.histogram('pg_objective', pg_objective)
                alf.summary.histogram('pg_objective_clipped',
                                      pg_objective_clipped)

        if self._check_numerics:
            assert torch.all(torch.isfinite(policy_gradient_loss))

        return policy_gradient_loss

    def _calc_returns_and_advantages(self, info, value):
        if not self._compute_advantages_internally:
            return info.returns, info.advantages
        # If rollout_value is present in ``info``, we use it to compute the
        # advantage. This is mainly for algorithms like PPG where at this time
        # info.value is the newly computed value which is different from the
        # rollout time value prediction. The rollout time value prediction is
        # preserved in info.rollout_value.
        assert hasattr(info, 'rollout_value'), (
            'Expect info.rollout_value to exist for computing returns '
            'and advatages inside PPOLoss.')
        return super()._calc_returns_and_advantages(info, info.rollout_value)
