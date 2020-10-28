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
from alf.utils.normalizers import AdaptiveNormalizer


@gin.configurable
class TDLoss(nn.Module):
    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 td_lambda=0.95,
                 normalize_target=False,
                 use_retrace=False,
                 debug_summaries=False,
                 name="TDLoss"):
        r"""Create a TDLoss object.

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
        use_retrace = False means one step or multi_step loss, use_retrace = True means retrace loss
            :math:`\mathcal{R} Q(x, a):=Q(x, a)+\mathbb{E}_{\mu}\left[\sum_{t \geq 0} \gamma^{t}\left(\prod_{s=1}^{t} c_{s}\right)\left(r_{t}+\gamma \mathbb{E}_{\pi} Q\left(x_{t+1}, \cdot\right)-Q\left(x_{t}, a_{t}\right)\right)\right]`
        References:

        Schulman et al. `High-Dimensional Continuous Control Using Generalized Advantage Estimation
        <https://arxiv.org/abs/1506.02438>`_

        Sutton et al. `Reinforcement Learning: An Introduction
        <http://incompleteideas.net/book/the-book.html>`_, Chapter 12, 2018

        Remi Munos et al. `Safe and efficient off-policy reinforcement learning
        <https://arxiv.org/pdf/1606.02647.pdf>`_

        Args:
            gamma (float): A discount factor for future rewards.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            td_lambda (float): Lambda parameter for TD-lambda computation.
            normalize_target (bool): whether to normalize target.
                Note that the effect of this is to change the loss. The critic
                value itself is not normalized.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this loss.
        """
        super().__init__()

        self._name = name
        self._gamma = gamma
        self._td_error_loss_fn = td_error_loss_fn
        self._lambda = td_lambda
        self._debug_summaries = debug_summaries
        self._normalize_target = normalize_target
        self._target_normalizer = None
        self._use_retrace = use_retrace

    def forward(self, experience, value, target_value, train_info):
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
            train_info : train_info includes action distrbution, actor, critic and
                other information. Different algorithm may have different info inside.
                For the retrace method, we can use SarsaInfo, SacInfo or DdpgInfo as train_info 
                for Sac, Sarsa or Ddpg algorithm. Adding train_info to calculate importance_ratio
                and importance_ratio_clipped.               
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
        elif self._use_retrace == False:
            advantages = value_ops.generalized_advantage_estimation(
                rewards=experience.reward,
                values=target_value,
                step_types=experience.step_type,
                discounts=experience.discount * self._gamma,
                td_lambda=self._lambda)
            returns = advantages + target_value[:-1]
        else:
            scope = alf.summary.scope(self.__class__.__name__)
            importance_ratio, importance_ratio_clipped = value_ops. \
            action_importance_ratio(
                action_distribution=train_info.action_distribution,
                collect_action_distribution=experience.rollout_info.
                action_distribution,
                action=experience.action,
                clipping_mode='capping',
                importance_ratio_clipping=0.0,
                log_prob_clipping=0.0,
                scope=scope,
                check_numerics=False,
                debug_summaries=self._debug_summaries)
            advantages = value_ops.generalized_advantage_estimation_retrace(
                importance_ratio=importance_ratio_clipped,
                rewards=experience.reward,
                values=value,
                target_value=target_value,
                step_types=experience.step_type,
                discounts=experience.discount * self._gamma,
                time_major=True,
                td_lambda=self._lambda)
            returns = advantages + value[:-1]
            returns = returns.detach()
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
            mask = experience.step_type[:-1] != StepType.LAST
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

        if loss.ndim == 3:
            # Multidimensional reward. Average over the critic loss for all dimensions
            loss = loss.mean(dim=2)

        # The shape of the loss expected by Algorith.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(loss=loss, extra=loss)
