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

import alf
from alf.algorithms.td_loss import LossInfo, StepType, TDLoss
from alf.utils import losses, tensor_utils, value_ops
from alf.utils.summary_utils import safe_mean_hist_summary


@alf.configurable
class OneStepTDLoss(TDLoss):
    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=losses.element_wise_squared_loss,
                 debug_summaries=False,
                 name="OneStepTDLoss"):
        """
        Args:
            gamma (float|list[float]): A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_error_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            debug_summaries (bool): True if debug summaries should be created
            name (str): The name of this loss.
        """
        super().__init__(
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            debug_summaries=debug_summaries,
            td_lambda=0.0,
            name=name)


@alf.configurable
class OneStepTDQRLoss(OneStepTDLoss):
    """One step temporal difference quantile regression loss. """

    def __init__(self,
                 num_quantiles=50,
                 gamma=0.99,
                 td_error_loss_fn=losses.huber_function,
                 sum_over_quantiles=False,
                 debug_summaries=False,
                 name="OneStepTDQRLoss"):
        """
        Args:
            num_quantiles (int): the number of quantiles.
            gamma (float|list[float]): A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_error_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            sum_over_quantiles (bool): Whether to sum over the quantiles.
            debug_summaries (bool): True if debug summaries should be created
            name (str): The name of this loss.
        """
        super().__init__(
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            debug_summaries=debug_summaries,
            name=name)

        self._cdf_midpoints = (torch.arange(
            num_quantiles, dtype=torch.float32) + 0.5) / num_quantiles
        self._sum_over_quantiles = sum_over_quantiles

    def compute_td_target(self, info, target_value):
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
            target_value (torch.Tensor): the time-major tensor for the value at
                each time step. This is used to calculate return. ``target_value``
                can be same as ``value``.
        Returns:
            td_target
        """
        if info.reward.ndim == 3:
            # [T, B, 1]
            assert info.reward.shape[-1] == 1, (
                "Cannot handle multi-dimensional reward for OneStepTDQRLoss.")

        value_shape = target_value.shape
        rewards = info.reward.unsqueeze(-1).expand(value_shape)
        step_types = info.step_type.unsqueeze(-1).expand(value_shape)
        discounts = info.discount.unsqueeze(-1).expand(
            value_shape) * self._gamma

        returns = value_ops.one_step_discounted_return(
            rewards=rewards,
            values=target_value,
            step_types=step_types,
            discounts=discounts)

        return returns

    def forward(self, info, value, target_value):
        """Calculate the loss.

        The first dimension of all the tensors is time dimension and the second
        dimesion is the batch dimension.

        Args:
            info (namedtuple): experience collected from ``unroll()`` or
                a replay buffer. All tensors are time-major. ``info`` should
                contain the following fields:
                - reward:
                - step_type:
                - discount:
            value (torch.Tensor): the time-major tensor for the value at each time
                step. The loss is between this and the calculated return.
            target_value (torch.Tensor): the time-major tensor for the value at
                each time step. This is used to calculate return. ``target_value``
                can be same as ``value``.
        Returns:
            LossInfo: with the ``extra`` field same as ``loss``.
        """
        assert value.ndim == 3 and target_value.ndim == 3, (
            "input value and target_value should have shape (T, B, N)")
        returns = self.compute_td_target(info, target_value)
        value = value[:-1]

        cdf_midpoints = self._cdf_midpoints.view(1, 1, 1, -1)
        quantiles = value.unsqueeze(-2)
        quantiles_target = returns.detach().unsqueeze(-1)
        diff = quantiles_target - quantiles

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type[:-1] != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, qtd, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('values' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("quantile_td" + suffix, qtd, mask)

                for i in range(value.shape[-1]):
                    for j in range(returns.shape[-1]):
                        suffix = f'/{j}-{i}'
                        _summarize(value[..., i], returns[..., j],
                                   diff[..., j, i], suffix)

        huber_loss = self._td_error_loss_fn(diff)
        loss = torch.abs(
            (cdf_midpoints - (diff.detach() < 0).float())) * huber_loss

        if self._sum_over_quantiles:
            loss = loss.mean(-2).sum(-1)
        else:
            loss = loss.mean(-2).mean(-1)

        # The shape of the loss expected by Algorith.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(loss=loss, extra=loss)
