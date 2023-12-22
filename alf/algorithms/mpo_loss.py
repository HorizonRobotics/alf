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

from typing import Any, NamedTuple
import torch.nn as nn
import torch
from torch import Tensor
import torch.distributions as td

import alf
from alf.algorithms.mcts_algorithm import calculate_exploration_policy, calculate_kl_exploration_policy
from alf.data_structures import LossInfo, StepType
from alf.utils import dist_utils, losses, tensor_utils, value_ops
from alf.utils.summary_utils import safe_mean_hist_summary, add_mean_hist_summary
from alf.nest import map_structure


class MPOInfo(NamedTuple):
    repr_info: Any = ()  # info from repr_alg

    # The action used  rollout
    action: Tensor = ()  # [B, .. ]
    reward: Tensor = ()  # [B, reward_dim]
    step_type: Tensor = ()  # [B]
    discount: Tensor = ()  # [B]
    action_distribution: td.Distribution = ()  # [B]

    # [B, num_candidate_actions, ...], candidate actions
    # () means all available discrete actions
    candidate_actions: Tensor = ()

    # If action is randomly sampled from action distribution, the weight are all 1.0.
    # If the action is not sampled, the weight is the probability of the
    # action calculated using ``action_distribution``.
    # It should be normalized (i.e. candidate_action_weights.sum(-1) == 1.0)
    candidate_action_weights: Tensor = ()  # [B, num_candidate_actions]

    # [B, replicas, reward_dim] for scalar prediction.
    # [B, replicas, reward_dim, n] for quantile prediction (n quantiles)
    # or categorical prediction (n categories of value)
    critics_dist: Tensor = ()

    target_critics: Tensor = ()  # [B, replicas, reward_dim]

    action_values: Tensor = ()  # [B, num_candidate_actions]


@alf.configurable
class MPOLoss(nn.Module):
    """The loss for MPO algorithm.

    Args:
        action_weight_regulization: the regularization weight for updating
            the action weights.
        value_loss: the loss function for value prediction.
        gamma: A discount factor for future rewards.
        td_lambda: Lambda parameter for TD-lambda computation.
    """

    def __init__(
            self,
            reward_dim: int = 1,
            action_weight_regulization: float = 1.0,
            max_kld: float = 0.1,
            value_loss: losses.ScalarPredictionLoss = losses.SquareLoss(),
            value_loss_weight=1.0,
            policy_loss_weight=1.0,
            gamma: float = 0.99,
            td_lambda: float = 0.95,
            name: str = 'MPOLoss',
    ):
        super().__init__()
        self._name = name
        self._gamma = gamma
        self._lambda = td_lambda
        self._action_weight_regulization = action_weight_regulization
        self._value_loss = value_loss
        self._reward_weights = torch.ones(reward_dim) / reward_dim
        self._value_loss_weight = value_loss_weight
        self._policy_loss_weight = policy_loss_weight
        self._max_kld = max_kld

    @torch.no_grad()
    def set_reward_weights(self, reward_weights: Tensor):
        self._reward_weights.copy_(reward_weights)

    def forward(self, info: MPOInfo):
        assert info.target_critics.ndim == 4, "target_critics must be 4D"
        length, batch_size, num_replicas, reward_dim = info.target_critics.shape
        # [T, B, replicas, reward_dim, ...]
        assert info.critics_dist.ndim >= 4, "critics_dist must be at least 4D"
        assert (length, batch_size, num_replicas,
                reward_dim) == info.critics_dist.shape[:4]
        num_candidate_actions = info.candidate_action_weights.shape[2]
        assert (length, batch_size,
                num_candidate_actions) == info.action_values.shape
        reward = info.reward
        if info.reward.ndim == 2:
            info = info._replace(reward=reward[:, :, None])
        # [T, B, reward_dim]
        assert (length, batch_size, reward_dim) == info.reward.shape
        # [T, B, num_candidate_actions]
        assert (length, batch_size,
                num_candidate_actions) == info.candidate_action_weights.shape

        action_weight = self._update_action_weight(
            info.candidate_action_weights, info.action_values)

        policy_loss = self._calc_policy_loss(info, action_weight)
        value_loss = self._calc_value_loss(info)

        loss = self._policy_loss_weight * policy_loss + self._value_loss_weight * value_loss

        return LossInfo(
            loss=loss, extra=dict(value=value_loss, policy=policy_loss))

    def _calc_value_loss(self, info):
        """
        Args:
            info:
            action_weight: [T, B, num_candidate_actions]
        """
        length, batch_size, num_replicas, reward_dim = info.target_critics.shape
        # [T, B, num_replicas, reward_dim]
        target_critics = info.target_critics
        # [T, B, reward_dim]
        target_critics = target_critics.min(dim=2)[0]
        # [T-1, B, reward_dim]
        returns = self._calc_return(info.reward, target_critics,
                                    info.step_type, info.discount)
        # [T-1, B, num_replicas, reward_dim, ...]
        critics_dist = info.critics_dist[:-1]
        value_loss = self._value_loss(
            critics_dist,
            returns[:, :, None, :].expand(*critics_dist.shape[:4]))
        # The shape of the loss expected by Algorith.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        # Also times (length / (length-1)) to compensate the scaling
        value_loss = tensor_utils.tensor_extend_zero(
            (length / (length - 1)) * value_loss)

        if alf.summary.should_record_summaries():
            # [T-1, B, replicas, reward_dim]
            q_values = self._value_loss.calc_expectation(critics_dist)
            mask = info.step_type[:-1] != StepType.LAST
            with alf.summary.scope(self._name):
                with alf.summary.scope(self._name):

                    def _summarize(v, r, td, suffix):
                        alf.summary.scalar(
                            "explained_variance_of_return_by_value" + suffix,
                            tensor_utils.explained_variance(v, r, mask))
                        safe_mean_hist_summary('values' + suffix, v, mask)
                        safe_mean_hist_summary('returns' + suffix, r, mask)
                        safe_mean_hist_summary("td_error" + suffix, td, mask)

                    td = returns[:, :, None, :] - q_values
                    for i in range(num_replicas):
                        for j in range(reward_dim):
                            suffix = '/' + str(i) + '/' + str(j)
                            _summarize(q_values[..., i, j], returns[..., j],
                                       td[..., i, j], suffix)
        return value_loss.mean(dim=(2, 3))

    def _calc_policy_loss(self, info: MPOInfo, action_weight: Tensor):
        """
        Args:
            info:
            action_weight: [T, B, num_candidate_actions] or ()
        """
        if info.action == ():
            # This condition is only possible for Categorical distribution
            assert isinstance(info.action_distribution, td.Categorical)
            policy_loss = -info.action_distribution.logits @ action_weight
        else:
            # candidate_actions.shape is [T, B, num_candidate_actions, ...]
            # log_prob() needs sample shape in the beginning
            action = info.candidate_actions.permute(
                2, 0, 1, *list(range(3, info.candidate_actions.ndim)))
            # [num_candidate_actions, T, B]
            action_log_probs = dist_utils.compute_log_probability(
                info.action_distribution, action)
            policy_loss = -torch.einsum('atb,tba->tb', action_log_probs,
                                        action_weight)
        return policy_loss

    def _calc_return(self, reward: Tensor, target_critics: Tensor,
                     step_type: Tensor, discount: Tensor):
        """
        Args:
            reward (Tensor): [T, B, reward_dim]
            target_critics (Tensor): [T, B, reward_dim]
            step_type (Tensor): [T, B]
            discount (float): [T, B]]
        Returns:
            returns (Tensor): [T-1, B, reward_dim]
        """
        if self._lambda == 1.0:
            returns = value_ops.discounted_return(
                rewards=reward,
                values=target_critics,
                step_types=step_type,
                discounts=discount * self._gamma)
        elif self._lambda == 0.0:
            returns = value_ops.one_step_discounted_return(
                rewards=reward,
                values=target_critics,
                step_types=step_type,
                discounts=discount * self._gamma)
        else:
            advantages = value_ops.generalized_advantage_estimation(
                rewards=reward,
                values=target_critics,
                step_types=step_type,
                discounts=discount * self._gamma,
                td_lambda=self._lambda)
            returns = advantages + target_critics[:-1]
        return returns

    def _update_action_weight(self, prior: Tensor, q_values: Tensor):
        """
        Args:
            prior: [T, B, num_candidate_actions], the prior action weight
            q_values: [T, B, num_candidate_actions]
        Returns:
            action_weight (Tensor): [T, B, num_candidate_actions], the updated
                action weight.
        """
        q_values = q_values - q_values.mean(dim=-1, keepdim=True)
        std = q_values.std(dim=-1).mean()
        q_values = q_values / (std + 0.01)

        policy, opt_steps = update_prior(
            q_values, prior, self._action_weight_regulization, self._max_kld)

        if alf.summary.should_record_summaries():
            log_ratio = (prior / policy).log()
            rkld = (prior * log_ratio).sum(-1)
            kld = -(policy * log_ratio).sum(-1)
            policy_entropy = -(policy * policy.log()).sum(-1)
            prior_entropy = -(prior * prior.log()).sum(-1)
            with alf.summary.scope(self._name):
                alf.summary.scalar("candidate_action_critics_std", std)
                alf.summary.scalar('actin_weight_optimization_steps',
                                   opt_steps)
                add_mean_hist_summary("rkld", rkld)
                add_mean_hist_summary("kld", kld)
                add_mean_hist_summary("updated_policy_entropy", policy_entropy)
                add_mean_hist_summary("prior_entropy", prior_entropy)

        return policy

    def calc_value_expectation(self, value_dist: Tensor):
        """Calculate the expected value from its distributional prediction
        """
        return self._value_loss.calc_expectation(value_dist)


def update_prior(value, prior, c: float, delta: float, tol: float = 1e-6):
    r"""Calculate exploration policy.

    This is similar to ``calculate_exploration_policy``, but using :math:`KL(p\|q)`
    instead of :math:`KL(q\|p)` for regularization.

    Notation:

        q: prior policy

        p: sampling probability

        v: value

    The exploration policy is found by minimizing the following:

    .. math::

        p = \arg\min_p \left[ -E_p(v) + c KL(p\|q) \right]
        s.t.  KL(p\|q) \le \delta


    which leads to the following solution:

    .. math::

        p_i = \frac{q_i \exp(v_i/(\lambda+c))}{Z}

    where :math:`Z` is the normalization constant and :math:`\lambda>=0` is the Lagrangian multiplier.

    When c is not big enough (i.e. lambda is strictly positive), Newton's method
    is used to find :math:`\lambda` such that :math:`KL(p\|q)=\delta`.

    Let :math:`\alpha = 1/(c+\lambba)`, the dirivative of :math:`KL(p\|q)` w.r.t.
    :math:`\alpha` is :math:`\alpha E_p(v - E_p(v))^2`.

    Args:
        value (Tensor): [..., K] Tensor
        prior (Tensor): [..., K] Tensor
        alpha:
        c:
    Returns:
        tuple:
        - Tensor: [..., K], q
        - float: c + lambda
        - int: the number of iterations
    """
    assert value.shape == prior.shape
    value = value - value.mean(dim=-1, keepdim=True)
    alpha = 1 / c

    p = prior * (alpha * value).exp()
    z = p.sum(dim=-1)
    p = p / z[..., None]
    Ev = (p * value).sum(-1)
    kl = alpha * Ev - z.log()
    c_is_too_small = kl > delta
    Ev2 = (p * (value - Ev[..., None])**2).sum(-1)
    derivative = alpha * Ev2
    new_alpha = torch.where(c_is_too_small, alpha + (delta - kl) / derivative,
                            alpha)
    iterations = 1

    # Largest alpha so far such that KL(p||q) < delta
    low = torch.zeros_like(new_alpha)
    # Smallest alpha so far such that KL(p||q) > delta
    high = torch.full_like(new_alpha, 1 / c)

    while ((new_alpha - alpha).abs() > tol).any() and iterations < 100:
        # If the new alpha is outside of [low, high], then bisect the interval
        # to get the new alpha.
        alpha = torch.where((new_alpha > high) | (new_alpha < low),
                            0.5 * (low + high), new_alpha)

        p = prior * (alpha[..., None] * value).exp()
        z = p.sum(dim=-1)
        p = p / z[..., None]
        Ev = (p * value).sum(-1)
        Ev2 = (p * (value - Ev[..., None])**2).sum(-1)
        kl = alpha * Ev - z.log()

        # Update low and high
        low = torch.where((kl < delta) & (alpha > low), alpha, low)
        high = torch.where((kl > delta) & (alpha < high), alpha, high)

        # Newton's step
        derivative = alpha * Ev2
        new_alpha = torch.where(c_is_too_small,
                                alpha + (delta - kl) / derivative, alpha)

        iterations += 1

    alpha = new_alpha
    p = prior * (alpha[..., None] * value).exp()
    z = p.sum(dim=-1, keepdim=True)
    p = p / z

    return p, iterations
