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

import math
import torch
import torch.nn as nn
from typing import Union, List, Callable, Optional

import alf
from alf.data_structures import LossInfo, namedtuple, StepType
from alf.utils.losses import element_wise_squared_loss, iqn_huber_loss
from alf.utils import losses, tensor_utils, value_ops
from alf.utils.summary_utils import safe_mean_hist_summary
from alf.utils.normalizers import AdaptiveNormalizer


@alf.configurable
class TDLoss(nn.Module):
    """Temporal difference loss."""

    def __init__(self,
                 gamma: Union[float, List[float]] = 0.99,
                 td_error_loss_fn: Callable = element_wise_squared_loss,
                 td_lambda: float = 0.95,
                 normalize_target: bool = False,
                 debug_summaries: bool = False,
                 name: str = "TDLoss"):
        r"""
        Let :math:`G_{t:T}` be the bootstrapped return from t to T:

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
            td_lambda: Lambda parameter for TD-lambda computation.
            normalize_target (bool): whether to normalize target.
                Note that the effect of this is to change the loss. The critic
                value itself is not normalized.
            debug_summaries: True if debug summaries should be created.
            name: The name of this loss.
        """
        super().__init__()

        self._name = name
        self._gamma = torch.tensor(gamma)
        self._td_error_loss_fn = td_error_loss_fn
        self._lambda = td_lambda
        self._debug_summaries = debug_summaries
        self._normalize_target = normalize_target
        self._target_normalizer = None

    @property
    def gamma(self):
        r"""Return the :math:`\gamma` value for discounting future rewards.

        Returns:
            Tensor: a rank-0 or rank-1 (multi-dim reward) floating tensor.
        """
        return self._gamma.clone()

    def compute_td_target(self, info: namedtuple, target_value: torch.Tensor):
        """Calculate the td target.

        The first dimension of all the tensors is time dimension and the second
        dimension is the batch dimension.

        Args:
            info (namedtuple): experience collected from ``unroll()`` or
                a replay buffer. All tensors are time-major. ``info`` should
                contain the following fields:
                - reward:
                - step_type:
                - discount:
            target_value (torch.Tensor): the time-major tensor for the value at
                each time step. This is used to calculate return. ``target_value``
                can be same as ``value``.
        Returns:
            td_target
        """
        if self._lambda == 1.0:
            returns = value_ops.discounted_return(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=info.discount * self._gamma)
        elif self._lambda == 0.0:
            returns = value_ops.one_step_discounted_return(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=info.discount * self._gamma)
        else:
            advantages = value_ops.generalized_advantage_estimation(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=info.discount * self._gamma,
                td_lambda=self._lambda)
            returns = advantages + target_value[:-1]

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

        return returns

    def forward(self, info: namedtuple, value: torch.Tensor,
                target_value: torch.Tensor):
        """Calculate the loss.

        The first dimension of all the tensors is time dimension and the second
        dimension is the batch dimension.

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
        returns = self.compute_td_target(info, target_value)
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

        td_error = returns - value
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
                    _summarize(value, returns, td_error, '')
                else:
                    td = returns - value
                    for i in range(value.shape[2]):
                        suffix = '/' + str(i)
                        _summarize(value[..., i], returns[..., i],
                                   td_error[..., i], suffix)

        loss = self._td_error_loss_fn(returns.detach(), value)

        if loss.ndim == 3:
            # Multidimensional reward. Average over the critic loss for all dimensions
            loss = loss.mean(dim=2)

        # The shape of the loss expected by Algorithm.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        td_error = tensor_utils.tensor_extend_zero(td_error)
        return LossInfo(loss=loss, extra=td_error)


@alf.configurable
class TDQRLoss(TDLoss):
    """Temporal difference quantile regression loss. 
    Compared to TDLoss, GAE support has not been implemented. """

    def __init__(self,
                 num_quantiles: int = 50,
                 gamma: Union[float, List[float]] = 0.99,
                 td_error_loss_fn: Callable = iqn_huber_loss,
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

    def forward(self,
                info: namedtuple,
                value: torch.Tensor,
                target_value: torch.Tensor,
                tau_hat: Optional[torch.Tensor] = (),
                delta_tau: Optional[torch.Tensor] = (),
                next_delta_tau: Optional[torch.Tensor] = ()):
        """Calculate the loss.

        The first dimension of all the tensors is time dimension and the second
        dimension is the batch dimension.

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
            tau_hat: midpoints of the sampled tau's, used as the inputs to the
                quantile function of the critics.
            delta_tau: the sampled increments of the probability for input of
                the quantile function of the critics.
            next_delta_tau: the sampled increments of the probability for the input 
                of the quantile function of the target critics.

        Returns:
            LossInfo: with the ``extra`` field same as ``loss``.
        """
        assert value.shape[-1] == self._num_quantiles, (
            "The input value should have same num_quantiles as pre-defined.")
        assert target_value.shape[-1] == self._num_quantiles, (
            "The input target_value should have same num_quantiles as pre-defined."
        )
        returns = self.compute_td_target(info, target_value)
        value = value[:-1]

        loss, diff = self._td_error_loss_fn(
            value, returns, tau_hat[:-1], next_delta_tau[1:],
            self._cdf_midpoints, self._sum_over_quantiles)

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type[:-1] != StepType.LAST
            v = (value * delta_tau[:-1]).sum(-1)
            r = (returns * next_delta_tau[1:]).sum(-1)
            td = r - v
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, d, n, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))

                    safe_mean_hist_summary('values' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("td_error" + suffix, td, mask)

                    cdf = (d <= 0).float().mean(-2)
                    mean_cdf = cdf.mean(0).mean(0)
                    alf.summary.histogram(
                        "explained_cdf_of_return_by_value_quantile" + suffix,
                        mean_cdf)
                    alf.summary.scalar(
                        "explained_0.25_fraction_of_value_by_returns" + suffix,
                        mean_cdf[math.ceil(0.25 * n) - 1])
                    alf.summary.scalar(
                        "explained_0.5_fraction_of_value_by_returns" + suffix,
                        mean_cdf[math.ceil(0.5 * n) - 1])
                    alf.summary.scalar(
                        "explained_0.75_fraction_of_value_by_returns" + suffix,
                        mean_cdf[math.ceil(0.75 * n) - 1])

                if value.ndim == 3:
                    _summarize(v, r, td, diff, self._num_quantiles, '')
                else:
                    for i in range(value.shape[-2]):
                        suffix = '/' + str(i)
                        _summarize(v[..., i], r[..., i], td[..., i],
                                   diff[..., i, :, :], self._num_quantiles,
                                   suffix)

        if loss.ndim == 3:
            # Multidimensional reward. Average over the critic loss for all dimensions
            loss = loss.mean(dim=2)

        # The shape of the loss expected by Algorithm.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(
            loss=loss, extra={
                'value': value,
                'returns': returns,
            })
