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

import torch
import torch.nn as nn
from typing import Union, List, Callable

import alf
from alf.data_structures import LossInfo, namedtuple, StepType
from alf.utils.losses import element_wise_squared_loss
from alf.utils import losses, tensor_utils, value_ops
from alf.utils.summary_utils import safe_mean_hist_summary
from alf.utils.normalizers import AdaptiveNormalizer


@alf.configurable
class TDLoss(nn.Module):
    """Temporal difference loss."""

    def __init__(self,
                 gamma: Union[float, List[float]] = 0.99,
                 td_error_loss_fn: Callable = element_wise_squared_loss,
                 clip: float = 0.,
                 td_lambda: float = 0.95,
                 normalize_target: bool = False,
                 debug_summaries: bool = False,
                 lb_target_q: float = 0.,
                 default_return: float = -1000.,
                 improve_w_goal_return: bool = False,
                 improve_w_nstep_bootstrap: bool = False,
                 improve_w_nstep_only: bool = False,
                 lower_bound_constraint: float = 0.,
                 lb_loss_scale: bool = False,
                 reward_multiplier: float = 1.,
                 positive_reward: bool = True,
                 use_retrace: bool = False,
                 name: str = "TDLoss"):
        r"""
        Let :math:`G_{t:T}` be the bootstraped return from t to T:

        .. math::

          G_{t:T} = \sum_{i=t+1}^T \gamma^{t-i-1}R_i + \gamma^{T-t} V(s_T)

        If ``td_lambda`` = 1, the target for step t is :math:`G_{t:T}`.

        If ``td_lambda`` = 0, the target for step t is :math:`G_{t:t+1}`

        If 0 < ``td_lambda`` < 1, the target for step t is the :math:`\lambda`-return:

        .. math::

            G_t^\lambda = (1 - \lambda) \sum_{i=t+1}^{T-1} \lambda^{i-t}G_{t:i} + \lambda^{T-t-1} G_{t:T}

        There is a simple relationship between :math:`\lambda`-return and
        the generalized advantage estimation :math:`\hat{A}^{GAE}_t`:

        .. math::

            G_t^\lambda = \hat{A}^{GAE}_t + V(s_t)

        where the generalized advantage estimation is defined as:

        .. math::

            \hat{A}^{GAE}_t = \sum_{i=t}^{T-1}(\gamma\lambda)^{i-t}(R_{i+1} + \gamma V(s_{i+1}) - V(s_i))

        References:

        Schulman et al. `High-Dimensional Continuous Control Using Generalized Advantage Estimation
        <https://arxiv.org/abs/1506.02438>`_

        Sutton et al. `Reinforcement Learning: An Introduction
        <http://incompleteideas.net/book/the-book.html>`_, Chapter 12, 2018

        Args:
            gamma: A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_error_loss_fn: A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            clip: When positive, loss clipping to the range [-clip, clip].
            td_lambda: Lambda parameter for TD-lambda computation.
            normalize_target (bool): whether to normalize target.
                Note that the effect of this is to change the loss. The critic
                value itself is not normalized.
            use_retrace: turn on retrace loss
                :math:`\mathcal{R} Q(x, a):=Q(x, a)+\mathbb{E}_{\mu}\left[\sum_{t \geq 0} \gamma^{t}\left(\prod_{s=1}^{t} c_{s}\right)\left(r_{t}+\gamma \mathbb{E}_{\pi} Q\left(x_{t+1}, \cdot\right)-Q\left(x_{t}, a_{t}\right)\right)\right]`
                copied from PR #695.
            lb_target_q: between 0 and 1.  When not zero, use this mixing rate for the
                lower bounded value target.  Only supports batch_length == 2, one step td.
            default_return: Keep it the same as replay_buffer.default_return to plot to
                tensorboard episodic_discounted_return only for the timesteps whose
                episode already ended.
            improve_w_goal_return: Use return calculated from the distance to hindsight
                goals.  Only supports batch_length == 2, one step td.
            improve_w_nstep_bootstrap: Look ahead 2 to n steps, and take the largest
                bootstrapped return to lower bound the value target of the 1st step.
            improve_w_nstep_only: Only use the n-th step bootstrapped return as
                value target lower bound.
            lower_bound_constraint: Use n-step bootstrapped return as lower bound
                constraints of the value.  See reference:
                He, F. S., Liu, Y., Schwing, A. G., and Peng, J.
                Learning to play in a day: Faster deep reinforcement learning
                by optimality tightening.  In 5th International Conference
                on Learning Representations, ICLR 2017, Toulon, France,
                April 24-26, 2017.  https://openreview.net/forum?id=rJ8Je4clg
            lb_loss_scale: Parameter for lower_bound_constraint.
            reward_multiplier: Weight on the hindsight goal return.
            positive_reward: If True, assumes 0/1 goal reward, otherwise, -1/0.
            debug_summaries: True if debug summaries should be created.
            name: The name of this loss.
        """
        super().__init__()

        self._name = name
        self._gamma = torch.tensor(gamma)
        self._td_error_loss_fn = td_error_loss_fn
        self._clip = clip
        self._lambda = td_lambda
        self._lb_target_q = lb_target_q
        self._default_return = default_return
        self._improve_w_goal_return = improve_w_goal_return
        self._improve_w_nstep_bootstrap = improve_w_nstep_bootstrap
        self._improve_w_nstep_only = improve_w_nstep_only
        self._lower_bound_constraint = lower_bound_constraint
        self._lb_loss_scale = lb_loss_scale
        self._reward_multiplier = reward_multiplier
        self._positive_reward = positive_reward
        self._use_retrace = use_retrace
        self._debug_summaries = debug_summaries
        self._normalize_target = normalize_target
        self._target_normalizer = None

    @property
    def gamma(self):
        """Return the :math:`\gamma` value for discounting future rewards.

        Returns:
            Tensor: a rank-0 or rank-1 (multi-dim reward) floating tensor.
        """
        return self._gamma.clone()

    def compute_td_target(self,
                          info: namedtuple,
                          value: torch.Tensor,
                          target_value: torch.Tensor,
                          qr: bool = False):
        """Calculate the td target.

        The first dimension of all the tensors is time dimension and the second
        dimesion is the batch dimension.

        Args:
            info (namedtuple): experience collected from ``unroll()`` or
                a replay buffer. All tensors are time-major. ``info`` should
                contain the following fields:
                - reward:
                - step_type:
                - discount:
            value (torch.Tensor): the time-major tensor for the value at
                each time step. Some of its value can be overwritten and passed
                back to the caller.
            target_value (torch.Tensor): the time-major tensor for the value at
                each time step. This is used to calculate return. ``target_value``
                can be same as ``value``, except for Retrace.
        Returns:
            td_target, updated value, optional constraint_loss
        """
        if not qr and info.reward.ndim == 3:
            # Multi-dim reward, not quantile regression.
            # [T, B, D] or [T, B, 1]
            discounts = info.discount.unsqueeze(-1) * self._gamma
        else:
            # [T, B]
            discounts = info.discount * self._gamma

        if self._lambda == 1.0:
            returns = value_ops.discounted_return(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=discounts)
        elif self._lambda == 0.0:
            returns = value_ops.one_step_discounted_return(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=discounts)
        elif not self._use_retrace:
            advantages = value_ops.generalized_advantage_estimation(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=discounts,
                td_lambda=self._lambda)
            returns = advantages + target_value[:-1]
        else:  # Retrace
            scope = alf.summary.scope(self.__class__.__name__)
            assert info.rollout_info.action_distribution != (), \
                "Algorithm does not provide rollout action_distribution"
            importance_ratio, importance_ratio_clipped = value_ops. \
            action_importance_ratio(
                action_distribution=info.action_distribution,
                rollout_action_distribution=info.rollout_info.action_distribution,
                action=info.action,
                clipping_mode='capping',
                importance_ratio_clipping=0.0,
                log_prob_clipping=0.0,
                scope=scope,
                check_numerics=False,
                debug_summaries=self._debug_summaries)
            advantages = value_ops.generalized_advantage_estimation_retrace(
                importance_ratio=importance_ratio_clipped,
                rewards=info.reward,
                values=value,
                target_value=target_value,
                step_types=info.step_type,
                discounts=discounts,
                use_retrace=True,
                time_major=True,
                td_lambda=self._lambda)

            returns = advantages + value[:-1]
            returns = returns.detach()

        constraint_loss = None
        if self._improve_w_nstep_bootstrap:
            assert self._lambda == 1.0, "td lambda does not work with this"
            future_returns = value_ops.first_step_future_discounted_returns(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=discounts)
            returns = value_ops.one_step_discounted_return(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=discounts)
            assert torch.all((returns[0] == future_returns[0]) | (
                info.step_type[0] == alf.data_structures.StepType.LAST)), \
                    str(returns[0]) + " ne\n" + str(future_returns[0]) + \
                    '\nrwd: ' + str(info.reward[0:2]) + \
                    '\nlast: ' + str(info.step_type[0:2]) + \
                    '\ndisct: ' + str(discounts[0:2]) + \
                    '\nv: ' + str(target_value[0:2])
            if self._improve_w_nstep_only:
                future_returns = future_returns[
                    -1]  # last is the n-step return
            else:
                future_returns = torch.max(future_returns, dim=0)[0]

            with alf.summary.scope(self._name):
                alf.summary.scalar(
                    "max_1_to_n_future_return_gt_td",
                    torch.mean((returns[0] < future_returns).float()))
                if self._lower_bound_constraint > 0:
                    alf.summary.scalar(
                        "max_1_to_n_future_return_gt_value",
                        torch.mean((value[0] < future_returns).float()))
                alf.summary.scalar("first_step_discounted_return",
                                   torch.mean(returns[0]))

            if self._lower_bound_constraint > 0:
                constraint_loss = self._lower_bound_constraint * torch.max(
                    torch.zeros_like(future_returns),
                    future_returns.detach() - value[0])**2
            else:
                returns[0] = torch.max(future_returns, returns[0]).detach()
            returns[1:] = 0
            value = value.clone()
            value[1:] = 0

        disc_ret = ()
        if hasattr(info, "discounted_return"):
            disc_ret = info.discounted_return
        if disc_ret != ():
            with alf.summary.scope(self._name):
                episode_ended = disc_ret > self._default_return
                alf.summary.scalar("episodic_discounted_return_all",
                                   torch.mean(disc_ret[episode_ended]))
                alf.summary.scalar(
                    "value_episode_ended_all",
                    torch.mean(value[:-1][:, episode_ended[0, :]]))

        if self._lb_target_q > 0 and disc_ret != ():
            her_cond = info.her
            mask = torch.ones(returns.shape, dtype=torch.bool)
            if her_cond != () and torch.any(~her_cond):
                mask = ~her_cond[:-1]
            disc_ret = disc_ret[
                1:]  # it's expanded in ddpg_algorithm, need to revert back.
            assert returns.shape == disc_ret.shape, "%s %s" % (returns.shape,
                                                               disc_ret.shape)
            with alf.summary.scope(self._name):
                alf.summary.scalar(
                    "episodic_return_gt_td",
                    torch.mean((returns < disc_ret).float()[mask]))
                alf.summary.scalar(
                    "episodic_discounted_return",
                    torch.mean(
                        disc_ret[mask & (disc_ret > self._default_return)]))
            returns[mask] = (1 - self._lb_target_q) * returns[mask] + \
                self._lb_target_q * torch.max(returns, disc_ret)[mask]

        if self._improve_w_goal_return:
            batch_length, batch_size = returns.shape[:2]
            her_cond = info.her
            if her_cond != () and torch.any(her_cond):
                dist = info.future_distance
                if self._positive_reward:
                    goal_return = torch.pow(
                        self._gamma * torch.ones(her_cond.shape), dist)
                else:
                    goal_return = -(1. - torch.pow(self._gamma, dist)) / (
                        1. - self._gamma)
                goal_return *= self._reward_multiplier
                goal_return = goal_return[:-1]
                returns_0 = returns
                # Multi-dim reward:
                if len(returns.shape) > 2:
                    returns_0 = returns[:, :, 0]
                returns_0 = torch.where(her_cond[:-1],
                                        torch.max(returns_0, goal_return),
                                        returns_0)
                with alf.summary.scope(self._name):
                    alf.summary.scalar(
                        "goal_return_gt_td",
                        torch.mean((returns_0 < goal_return).float()))
                    alf.summary.scalar("goal_return", torch.mean(goal_return))
                if len(returns.shape) > 2:
                    returns[:, :, 0] = returns_0
                else:
                    returns = returns_0

        return returns, value, constraint_loss

    def forward(self, info: namedtuple, value: torch.Tensor,
                target_value: torch.Tensor):
        """Calculate the loss.

        The first dimension of all the tensors is time dimension and the second
        dimesion is the batch dimension.

        Args:
            info: experience collected from ``unroll()`` or
                a replay buffer. All tensors are time-major. ``info`` should
                contain the following fields:
                - reward:
                - step_type:
                - discount:
            value: the time-major tensor for the value at each time
                step. The loss is between this and the calculated return.
            target_value: the time-major tensor for the value at
                each time step. This is used to calculate return. ``target_value``
                can be same as ``value``.
        Returns:
            LossInfo: with the ``extra`` field same as ``loss``.
        """
        returns, value, constraint_loss = self.compute_td_target(
            info, value, target_value)
        value = value[:-1]

        if self._normalize_target:
            if self._target_normalizer is None:
                self._target_normalizer = AdaptiveNormalizer(
                    alf.TensorSpec(value.shape[2:]),
                    auto_update=False,
                    debug_summaries=self._debug_summaries,
                    name=self._name + ".target_normalizer")

            self._target_normalizer.update(returns)
            returns = self._target_normalizer.normalize(returns)
            value = self._target_normalizer.normalize(value)

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type[:-1] != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('values' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("td_error" + suffix, td, mask)

                if value.ndim == 2:
                    _summarize(value, returns, returns - value, '')
                else:
                    td = returns - value
                    for i in range(value.shape[2]):
                        suffix = '/' + str(i)
                        _summarize(value[..., i], returns[..., i], td[..., i],
                                   suffix)

        loss = self._td_error_loss_fn(returns.detach(), value)
        if self._clip > 0:
            loss = torch.clamp(loss, min=-self._clip, max=self._clip)

        if loss.ndim == 3:
            # Multidimensional reward. Average over the critic loss for all dimensions
            loss = loss.mean(dim=2)

        if self._improve_w_nstep_bootstrap:
            # Ignore 2nd to n-th step losses.
            loss[1:] = 0
            if self._lower_bound_constraint > 0:
                assert constraint_loss.shape == loss.shape[1:], \
                    f"{constraint_loss.shape} != {loss.shape}[1:]"
                c_loss = constraint_loss.clone().unsqueeze(0).repeat(
                    (loss.shape[0], 1))
                c_loss[1:] = 0
                if self._lb_loss_scale:
                    scale = (
                        torch.sum(loss) / torch.sum(c_loss + loss)).detach()
                else:
                    scale = 1
                loss = (c_loss + loss) * scale

        # The shape of the loss expected by Algorith.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(loss=loss, extra=loss)


@alf.configurable
class TDQRLoss(TDLoss):
    """Temporal difference quantile regression loss. 
    Compared to TDLoss, GAE support has not been implemented. """

    def __init__(self,
                 num_quantiles: int = 50,
                 gamma: Union[float, List[float]] = 0.99,
                 td_error_loss_fn: Callable = losses.huber_function,
                 td_lambda: float = 1.0,
                 sum_over_quantiles: bool = False,
                 debug_summaries: bool = False,
                 name: str = "TDQRLoss"):
        """
        Args:
            num_quantiles: the number of quantiles.
            gamma: A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_error_loss_fn: A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            td_lambda: Lambda parameter for TD-lambda computation. Currently
                only supports 1 and 0.
            sum_over_quantiles: If True, the quantile regression loss will
                be summed along the quantile dimension. Otherwise, it will be
                averaged along the quantile dimension instead. Default is False.
            debug_summaries: True if debug summaries should be created
            name: The name of this loss.
        """
        assert td_lambda in (0, 1), (
            "Currently GAE is not supported, so td_lambda has to be 0 or 1.")
        super().__init__(
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            td_lambda=td_lambda,
            debug_summaries=debug_summaries,
            name=name)

        self._num_quantiles = num_quantiles
        self._cdf_midpoints = (torch.arange(
            num_quantiles, dtype=torch.float32) + 0.5) / num_quantiles
        self._sum_over_quantiles = sum_over_quantiles

    def forward(self, info: namedtuple, value: torch.Tensor,
                target_value: torch.Tensor):
        """Calculate the loss.

        The first dimension of all the tensors is time dimension and the second
        dimesion is the batch dimension.

        Args:
            info: experience collected from ``unroll()`` or
                a replay buffer. All tensors are time-major. ``info`` should
                contain the following fields:
                - reward:
                - step_type:
                - discount:
            value: the time-major tensor for the value at each time
                step. The loss is between this and the calculated return.
            target_value: the time-major tensor for the value at
                each time step. This is used to calculate return. ``target_value``
                can be same as ``value``.
        Returns:
            LossInfo: with the ``extra`` field same as ``loss``.
        """
        assert value.shape[-1] == self._num_quantiles, (
            "The input value should have same num_quantiles as pre-defiend.")
        assert target_value.shape[-1] == self._num_quantiles, (
            "The input target_value should have same num_quantiles as pre-defiend."
        )
        returns, value, constraint_loss = self.compute_td_target(
            info, value, target_value, qr=True)
        value = value[:-1]

        # for quantile regression TD, the value and target both have shape
        # (T-1, B, n_quantiles) for scalar reward and
        # (T-1, B, reward_dim, n_quantiles) for multi-dim reward.
        # The quantile TD has shape
        # (T-1, B, n_quantiles, n_quantiles) for scalar reward and
        # (T-1, B, reward_dim, n_quantiles, n_quantiles) for multi-dim reward
        quantiles = value.unsqueeze(-2)
        quantiles_target = returns.detach().unsqueeze(-1)
        diff = quantiles_target - quantiles

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type[:-1] != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, d, suffix):
                    cdf = (d <= 0).float().mean(-2)
                    mean_cdf = cdf.mean(0).mean(0)
                    alf.summary.histogram(
                        "explained_cdf_of_return_by_value_quantile" + suffix,
                        mean_cdf)

                if value.ndim == 3:
                    _summarize(value, returns, diff, '')
                else:
                    for i in range(value.shape[-2]):
                        suffix = '/' + str(i)
                        _summarize(value[..., i, :], returns[..., i, :],
                                   diff[..., i, :, :], suffix)

        huber_loss = self._td_error_loss_fn(diff)
        loss = torch.abs(
            (self._cdf_midpoints - (diff.detach() < 0).float())) * huber_loss

        if self._sum_over_quantiles:
            loss = loss.mean(-2).sum(-1)
        else:
            loss = loss.mean(dim=(-2, -1))

        if loss.ndim == 3:
            # Multidimensional reward. Average over the critic loss for all dimensions
            loss = loss.mean(dim=2)

        # The shape of the loss expected by Algorith.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(loss=loss, extra=loss)
