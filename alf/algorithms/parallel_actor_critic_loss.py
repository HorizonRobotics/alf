# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.data_structures import LossInfo
from alf.utils.losses import element_wise_squared_loss
from alf.utils import tensor_utils, dist_utils, value_ops
from alf.algorithms.actor_critic_loss import ActorCriticLoss, ActorCriticLossInfo, _normalize_advantages


@alf.configurable
class ParallelActorCriticLoss(ActorCriticLoss):

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
                 name="ParallelActorCriticLoss"):
        """An parallel actor-critic loss equals to

        .. code-block:: python

            (policy_gradient_loss
            + td_loss_weight * td_loss
            - entropy_regularization * entropy)
        
        The args are the same as the ActorCriticLoss. 
        
        There are only two different things from ActorCriticLoss. Please see the comments in line 111 and 112 for the first difference. 
        For the other one, check the comments in line 128 and 129.
        """
        super().__init__(gamma=gamma,
                        td_error_loss_fn=td_error_loss_fn,
                        use_gae=use_gae,
                        td_lambda=td_lambda,
                        use_td_lambda_return=use_td_lambda_return,
                        normalize_advantages=normalize_advantages,
                        advantage_clip=advantage_clip,
                        entropy_regularization=entropy_regularization,
                        td_loss_weight=td_loss_weight,
                        debug_summaries=debug_summaries,
                        name=name)

    def forward(self, info):
        """Cacluate the parallel actor critic loss. The first dimension of all the tensors is
        time dimension and the second dimesion is the batch dimension.

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
                alf.summary.scalar("values", value.mean())
                alf.summary.scalar("returns", returns.mean())
                alf.summary.scalar("advantages/mean", advantages.mean())
                alf.summary.histogram("advantages/value", advantages)
                alf.summary.scalar(
                    "explained_variance_of_return_by_value",
                    tensor_utils.explained_variance(value, returns))

        if self._normalize_advantages:
            advantages = _normalize_advantages(advantages)

        if self._advantage_clip:
            advantages = torch.clamp(advantages, -self._advantage_clip,
                                     self._advantage_clip)

        pg_loss = self._pg_loss(info, advantages.detach())

        td_loss = self._td_error_loss_fn(returns.detach(), value)

        loss = pg_loss + self._td_loss_weight * td_loss

        entropy_loss = ()
        if self._entropy_regularization is not None:
            entropy, entropy_for_gradient = dist_utils.entropy_with_fallback(
                info.action_distribution, return_sum=False)
            # The first one of the two differences from ActorCriticLoss is the following two additional lines.
            # We need to squeeze one extra dimension in the parallel setting.
            entropy = entropy.squeeze(1)
            entropy_for_gradient = entropy_for_gradient.squeeze(1)
            entropy_loss = alf.nest.map_structure(lambda x: -x, entropy)
            loss -= self._entropy_regularization * sum(
                alf.nest.flatten(entropy_for_gradient))

        return LossInfo(
            loss=loss,
            extra=ActorCriticLossInfo(
                td_loss=td_loss, pg_loss=pg_loss, neg_entropy=entropy_loss))

    def _pg_loss(self, info, advantages):
        action_log_prob = dist_utils.compute_log_probability(
            info.action_distribution, info.action.unsqueeze(1))
        # The second one of the two differences from ActorCriticLoss is the following additional line.
        # We need to squeeze one extra dimension in the parallel setting.
        action_log_prob = action_log_prob.squeeze(1)
        return -advantages * action_log_prob
