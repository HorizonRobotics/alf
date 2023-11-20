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
"""Various functions related to calculating values."""
import torch

import alf
from alf.data_structures import StepType
from alf.utils import common, dist_utils


def action_importance_ratio(action_distribution,
                            rollout_action_distribution,
                            action,
                            clipping_mode,
                            scope,
                            importance_ratio_clipping,
                            log_prob_clipping,
                            check_numerics,
                            debug_summaries,
                            rollout_log_prob=None):
    """ ratio for importance sampling, used in PPO loss and vtrace loss.

        Caller has to save alf.summary.scope() and pass scope to this function.

        Args:
            action_distribution (nested td.distribution): Distribution over
                actions under target policy.
            rollout_action_distribution (nested td.distribution): distribution
                over actions from behavior policy, used to sample actions for
                the rollout.
            action (nested tensor): possibly batched action tuple taken during
                rollout.
            clipping_mode (str): mode for clipping the importance ratio:

                * 'double_sided': clips the range of importance ratio into
                  ``[1-importance_ratio_clipping, 1+importance_ratio_clipping]``,
                  which is used by PPOLoss.
                * 'capping': clips the range of importance ratio into
                  ``min(1+importance_ratio_clipping, importance_ratio)``,
                  which is used by VTraceLoss, where c_bar or rho_bar =
                  1+importance_ratio_clipping.

            scope (name scope manager): returned by ``alf.summary.scope()``, set
                outside.
            importance_ratio_clipping (float):  Epsilon in clipped, surrogate
                PPO objective. See the cited paper for more detail.
            log_prob_clipping (float): If >0, clipping log probs to the range
                (-log_prob_clipping, log_prob_clipping) to prevent inf / NaN
                values.
            check_numerics (bool):  If true, adds checks to help find
                ``NaN``/``Inf`` values. For debugging only.
            debug_summaries (bool): If true, output summary metrics to tensorboard.
            rollout_log_prob (nested tensor): the log probability of the action


        Returns:
            importance_ratio (Tensor), importance_ratio_clipped (Tensor).
    """
    current_policy_distribution = action_distribution

    if rollout_log_prob is not None:
        sample_action_log_probs = rollout_log_prob.detach()
    else:
        sample_action_log_probs = dist_utils.compute_log_probability(
            rollout_action_distribution, action).detach()

    action_log_prob = dist_utils.compute_log_probability(
        current_policy_distribution, action)

    if log_prob_clipping > 0.0:
        action_log_prob = action_log_prob.clamp(-log_prob_clipping,
                                                log_prob_clipping)
    if check_numerics:
        assert torch.all(torch.isfinite(action_log_prob))

    # Prepare both clipped and unclipped importance ratios.
    importance_ratio = (action_log_prob - sample_action_log_probs).exp()
    if check_numerics:
        assert torch.all(torch.isfinite(importance_ratio))

    if clipping_mode == 'double_sided':
        importance_ratio_clipped = importance_ratio.clamp(
            1 - importance_ratio_clipping, 1 + importance_ratio_clipping)
    elif clipping_mode == 'capping':
        importance_ratio_clipped = torch.min(
            importance_ratio, torch.tensor(1 + importance_ratio_clipping))
    else:
        raise Exception('Unsupported clipping mode: ' + clipping_mode)

    if debug_summaries and alf.summary.should_record_summaries():
        with scope:
            if importance_ratio_clipping > 0.0:
                clip_fraction = (torch.abs(importance_ratio - 1.0) >
                                 importance_ratio_clipping).to(
                                     torch.float32).mean()
                alf.summary.scalar('clip_fraction', clip_fraction)

            alf.summary.histogram('action_log_prob', action_log_prob)
            alf.summary.histogram('action_log_prob_sample',
                                  sample_action_log_probs)
            alf.summary.histogram('importance_ratio', importance_ratio)
            alf.summary.scalar('importance_ratio_mean',
                               importance_ratio.mean())
            alf.summary.histogram('importance_ratio_clipped',
                                  importance_ratio_clipped)

    return importance_ratio, importance_ratio_clipped


def discounted_return(rewards, values, step_types, discounts, time_major=True):
    r"""Computes discounted return for the first T-1 steps.

    The difference between this function and the one tf_agents.utils.value_ops
    is that the accumulated_discounted_reward is replaced by value for is_last
    steps in this function.

    .. math::

        Q_t = \sum_{t'=t}^T \gamma^{t'-t} * r_{t'} + \gamma^{T-t+1}*final\_value.


    Define abbreviations:

    - B: batch size representing number of trajectories
    - T: number of steps per trajectory

    Args:
        rewards (Tensor): shape is [T, B] (or [T]) representing rewards.
        values (Tensor): shape is [T, B] (or [T]) when representing values,
            [T, B, n_quantiles] or [T, n_quantiles] when representing quantiles
            of value distributions.
        step_types (Tensor): shape is [T, B] (or [T]) representing step types.
        discounts (Tensor): shape is [T, B] (or [T]) representing discounts.
        time_major (bool): Whether input tensors are time major.
            False means input tensors have shape [B, T].

    Returns:
        A tensor with shape [T-1, B] (or [T-1]) representing the discounted
        returns. Shape is [B, T-1] when time_major is false.
    """
    if not time_major:
        discounts = discounts.transpose(0, 1)
        rewards = rewards.transpose(0, 1)
        values = values.transpose(0, 1)
        step_types = step_types.transpose(0, 1)

    assert values.shape[0] >= 2, ("The sequence length needs to be "
                                  "at least 2. Got {s}".format(
                                      s=values.shape[0]))

    is_lasts = (step_types == StepType.LAST).to(dtype=torch.float32)
    is_lasts = common.expand_dims_as(is_lasts, values)
    discounts = common.expand_dims_as(discounts, values)
    rewards = common.expand_dims_as(rewards, values)

    rets = torch.zeros_like(values)
    rets[-1] = values[-1]

    with torch.no_grad():
        for t in reversed(range(rewards.shape[0] - 1)):
            acc_value = rets[t + 1] * discounts[t + 1] + rewards[t + 1]
            rets[t] = is_lasts[t] * values[t] + (1 - is_lasts[t]) * acc_value

    rets = rets[:-1]

    if not time_major:
        rets = rets.transpose(0, 1)

    return rets.detach()


def one_step_discounted_return(rewards, values, step_types, discounts):
    """Calculate the one step discounted return  for the first T-1 steps.

    return = next_reward + next_discount * next_value if is not the last step;
    otherwise will set return = current_discount * current_value.

    Note: Input tensors must be time major
    Args:
        rewards (Tensor): shape is [T, B] (or [T]) representing rewards.
        values (Tensor): shape is [T, B] (or [T]) when representing values,
            [T, B, n_quantiles] or [T, n_quantiles] when representing quantiles
            of value distributions.
        step_types (Tensor): shape is [T, B] (or [T]) representing step types.
        discounts (Tensor): shape is [T, B] (or [T]) representing discounts.
    Returns:
        A tensor with shape [T-1, B] (or [T-1]) representing the discounted
        returns.
    """
    assert values.shape[0] >= 2, ("The sequence length needs to be "
                                  "at least 2. Got {s}".format(
                                      s=values.shape[0]))

    is_lasts = (step_types == StepType.LAST).to(dtype=torch.float32)
    is_lasts = common.expand_dims_as(is_lasts, values)
    discounts = common.expand_dims_as(discounts, values)
    rewards = common.expand_dims_as(rewards, values)

    discounted_values = discounts * values
    rets = (1 - is_lasts[:-1]) * (rewards[1:] + discounted_values[1:]) + \
                 is_lasts[:-1] * discounted_values[:-1]
    return rets.detach()


def generalized_advantage_estimation(rewards,
                                     values,
                                     step_types,
                                     discounts,
                                     td_lambda=1.0,
                                     time_major=True):
    """Computes generalized advantage estimation (GAE) for the first T-1 steps.

    For theory, see
    "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    by John Schulman, Philipp Moritz et al.
    See https://arxiv.org/abs/1506.02438 for full paper.

    The difference between this function and the one tf_agents.utils.value_ops
    is that the accumulated_td is reset to 0 for is_last steps in this function.

    Define abbreviations:

    - B: batch size representing number of trajectories
    - T: number of steps per trajectory

    Args:
        rewards (Tensor): shape is [T, B] (or [T]) representing rewards.
        values (Tensor): shape is [T,B] (or [T]) representing values.
        step_types (Tensor): shape is [T,B] (or [T]) representing step types.
        discounts (Tensor): shape is [T, B] (or [T]) representing discounts.
        td_lambda (float): A scalar between [0, 1]. It's used for variance
            reduction in temporal difference.
        time_major (bool): Whether input tensors are time major.
            False means input tensors have shape [B, T].

    Returns:
        A tensor with shape [T-1, B] representing advantages. Shape is [B, T-1]
        when time_major is false.
    """

    if not time_major:
        discounts = discounts.transpose(0, 1)
        rewards = rewards.transpose(0, 1)
        values = values.transpose(0, 1)
        step_types = step_types.transpose(0, 1)

    assert values.shape[0] >= 2, ("The sequence length needs to be "
                                  "at least 2. Got {s}".format(
                                      s=values.shape[0]))

    is_lasts = (step_types == StepType.LAST).to(dtype=torch.float32)
    is_lasts = common.expand_dims_as(is_lasts, values)
    discounts = common.expand_dims_as(discounts, values)

    weighted_discounts = discounts[1:] * td_lambda

    advs = torch.zeros_like(values)
    delta = rewards[1:] + discounts[1:] * values[1:] - values[:-1]

    with torch.no_grad():
        for t in reversed(range(rewards.shape[0] - 1)):
            advs[t] = (1 - is_lasts[t]) * \
                      (delta[t] + weighted_discounts[t] * advs[t + 1])
        advs = advs[:-1]

    if not time_major:
        advs = advs.transpose(0, 1)

    return advs.detach()
