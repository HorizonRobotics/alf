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

import gin
import torch
import torch.nn as nn

import alf
from alf.data_structures import LossInfo, StepType
from alf.utils.losses import element_wise_squared_loss
from alf.utils import tensor_utils, value_ops
from alf.utils.summary_utils import safe_mean_hist_summary


@gin.configurable
class TDLoss(nn.Module):
    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 td_lambda=0.95,
                 debug_summaries=False,
                 name="MultiStepTDLoss"):
        r"""Create a MultiStepTDLoss object.

        Let :math:`G_{t:T}` be the bootstaped return from t to T:
            :math:`G_{t:T} = \sum_{i=t+1}^T \gamma^{t-i-1}R_i + \gamma^{T-t} V(s_T)`
        If ``td_lambda`` = 1, the target for step t is :math:`G_{t:T}`.
        If ``td_lambda`` = 0, the target for step t is :math:`G_{t:t+1}`
        If 0 < ``td_lambda`` < 1, the target for step t is the :math:`\lambda`-return:
            :math:`G_t^\lambda = (1 - \lambda) \sum_{i=t+1}^{T-1} \lambda^{i-t}G_{t:i} + \lambda^{T-t-1} G_{t:T}`
        There is a simple relationship between :math:`\lambda`-return and
        the generalized advantage estimation :math:`\hat{A}^{GAE}_t`:
            :math:`G_t^\lambda = \hat{A}^{GAE}_t + V(s_t)`
        where the generalized advantage estimation is defined as:
            :math:`\hat{A}^{GAE}_t = \sum_{i=t}^{T-1}(\gamma\lambda)^{i-t}(R_{i+1} + \gamma V(s_{i+1}) - V(s_i))`

        References:

        Schulman et al. `High-Dimensional Continuous Control Using Generalized Advantage Estimation
        <https://arxiv.org/abs/1506.02438>`_

        Sutton et al. `Reinforcement Learning: An Introduction
        <http://incompleteideas.net/book/the-book.html>`_, Chapter 12, 2018

        Args:
            gamma (float): A discount factor for future rewards.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            td_lambda (float): Lambda parameter for TD-lambda computation.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this loss.
        """
        super().__init__()

        self._name = name
        self._gamma = gamma
        self._td_error_loss_fn = td_error_loss_fn
        self._lambda = td_lambda
        self._debug_summaries = debug_summaries

    def forward(self, experience, value, target_value):
        """Cacluate the loss.

        The first dimension of all the tensors is time dimension and the second
        dimesion is the batch dimension.

        Args:
            experience (Experience): experience collected from ``unroll()`` or
                a replay buffer. All tensors are time-major.
            value (torch.Tensor): the time-major tensor for the value at each time
                step. The loss is between this and the calculated return.
            target_value (torch.Tensor): the time-major tensor for the value at
                each time step. This is used to calculate return. ``target_value``
                can be same as ``value``.
        Returns:
            LossInfo: with the ``extra`` field same as ``loss``.
        """
        if self._lambda == 1.0:
            returns = value_ops.discounted_return(
                rewards=experience.reward,
                values=target_value,
                step_types=experience.step_type,
                discounts=experience.discount * self._gamma)
        elif self._lambda == 0.0:
            returns = value_ops.one_step_discounted_return(
                rewards=experience.reward,
                values=target_value,
                step_types=experience.step_type,
                discounts=experience.discount * self._gamma)
        else:
            advantages = value_ops.generalized_advantage_estimation(
                rewards=experience.reward,
                values=target_value,
                step_types=experience.step_type,
                discounts=experience.discount * self._gamma,
                td_lambda=self._lambda)
            returns = advantages + target_value[:-1]

        value = value[:-1]

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = experience.step_type[:-1] != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix, new_mask=None):
                    m = mask
                    if new_mask is not None:
                        m = new_mask
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, m))
                    safe_mean_hist_summary('values' + suffix, v, m)
                    safe_mean_hist_summary('returns' + suffix, r, m)
                    safe_mean_hist_summary("td_error" + suffix, td, m)

                if value.ndim == 2:
                    _summarize(value, returns, returns - value, '')
                else:
                    td = returns - value
                    for i in range(value.shape[2]):
                        suffix = '/' + str(i)
                        _summarize(value[..., i], returns[..., i], td[..., i],
                                   suffix)
                        if experience.batch_info.her != ():
                            her_cond = experience.batch_info.her.unsqueeze(0)
                            alf.summary.scalar("her_rate" + suffix,
                                               torch.mean(her_cond.float()))
                            non_her = ~her_cond
                            _summarize(value[..., i][her_cond],
                                       returns[..., i][her_cond],
                                       td[..., i][her_cond], suffix + "/her",
                                       mask[her_cond])
                            _summarize(value[..., i][non_her],
                                       returns[..., i][non_her],
                                       td[..., i][non_her], suffix + "/nonher",
                                       mask[non_her])
                        if isinstance(
                                experience.observation, dict
                        ) and "final_goal" in experience.observation:
                            _cond = experience.observation["final_goal"][
                                1:].squeeze(2) > 0
                            alf.summary.scalar("final_goal_rate" + suffix,
                                               torch.mean(_cond.float()))
                            _non = ~_cond
                            _summarize(value[..., i][_cond],
                                       returns[..., i][_cond],
                                       td[..., i][_cond],
                                       suffix + "/final_goal", mask[_cond])
                            _summarize(value[..., i][_non],
                                       returns[..., i][_non], td[..., i][_non],
                                       suffix + "/non_final", mask[_non])
                        observation = experience.observation
                        if (isinstance(observation, dict)
                                and "desired_goal" in observation):
                            if "aux_desired" in observation:
                                o = observation["aux_desired"]
                            elif observation["desired_goal"].shape[-1] > 2:
                                o = observation["desired_goal"][..., 2:]
                            else:
                                o = torch.zeros(
                                    (observation["desired_goal"].shape[0],
                                     observation["desired_goal"].shape[1], 10))
                            # take first n - 1 time steps for judging
                            o = torch.abs(o[:-1, ...])
                            aux_realistic = torch.norm(
                                torch.cat((o[:, :, 2:5], o[:, :, 6:9]), dim=2),
                                dim=2) < 3
                            aux_realistic &= torch.abs(o[:, :, 5]) < 7.5
                            aux_realistic &= torch.abs(o[:, :, 9]) < 3.15 + 0.5
                            dist_th = 5.7 + 0.5
                            aux_realistic &= torch.abs(
                                observation["desired_goal"][0, :, 0]) < dist_th
                            aux_realistic &= torch.abs(
                                observation["desired_goal"][0, :, 1]) < dist_th
                            real_rate = torch.mean(aux_realistic.float())
                            alf.summary.scalar("realistic_rate" + suffix,
                                               real_rate)
                            if real_rate > 0:
                                real_nonher = aux_realistic & non_her
                                _summarize(value[..., i][real_nonher],
                                           returns[..., i][real_nonher],
                                           td[..., i][real_nonher],
                                           suffix + "/real_nonher",
                                           mask[real_nonher])
                            if real_rate < 1:
                                non_real = ~aux_realistic
                                _summarize(value[..., i][non_real],
                                           returns[..., i][non_real],
                                           td[..., i][non_real],
                                           suffix + "/unreal", mask[non_real])
                                if experience.batch_info.her != ():
                                    nr_nonher = non_real & non_her
                                    _summarize(value[..., i][nr_nonher],
                                               returns[..., i][nr_nonher],
                                               td[..., i][nr_nonher],
                                               suffix + "/unreal_nonher",
                                               mask[nr_nonher])

        loss = self._td_error_loss_fn(returns.detach(), value)

        if loss.ndim == 3:
            # Multidimensional reward. Average over the critic loss for all dimensions
            loss = loss.mean(dim=2)

        # The shape of the loss expected by Algorith.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(loss=loss, extra=loss)
