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

from typing import Union, List, Callable
import torch.distributions as td

import alf
from alf.data_structures import namedtuple, LossInfo
from alf.algorithms.algorithm import Loss
from alf.utils.losses import element_wise_squared_loss
from .ppg_utils import PPGTrainInfo
from alf.utils import tensor_utils, value_ops

# The auxiliary phase updates the network parameters based a loss that
# has 3 parts. See PPGAuxPhaseLoss below for details.
PPGAuxPhaseLossInfo = namedtuple('PPGAuxPhaseLossInfo', [
    'td_loss_actual',
    'td_loss_aux',
    'policy_kl_loss',
])


@alf.configurable
class PPGAuxPhaseLoss(Loss):
    """Implementation of the PPG Auxiliary Phase Loss Function

    The loss is used in the auxiliary update phase of the Phasic Policy Gradient
    (PPG) algorithm and the total loss is a (weighted) sum of 3 components

    1. td_loss_actual: the MSE-like loss between the value head's value
       estimation and the TD-based value target.

    2. td_loss_aux: the MSE-like loss between the auxiliary value head's value
       estimation and the TD-based value target.

    3. policy_kl_loss: this is the behavior cloning loss that measures the KL
       divergence between the old policy and the target policy

    Since the first 2 components are comparable, there is one weight defined for
    the last one (behavior cloning KLD) to tune the relative significance of the
    components.

    For detailed illustration of the PPG auxiliary phase loss, see
    https://github.com/HorizonRobotics/alf/issues/965#issuecomment-897949432

    """

    def __init__(self,
                 td_error_loss_fn: Callable = element_wise_squared_loss,
                 policy_kl_loss_weight: float = 1.0,
                 gamma: Union[float, List[float]] = 0.999,
                 td_lambda: float = 0.95,
                 name: str = 'PPGAuxPhaseLoss'):
        """Construct a PPGAuxPhaseLoss instance with parameters

        Args:

            td_error_loss_fn: a binary tensor operator that computes the
                MSE-like loss between the two inputs, and it defines how we
                aggregate the error between the value estimations and the
                TD-based value targets
            policy_kl_loss_weight: this parameter is used to tune up and down
                the relative significance of the behavior cloning KLD in the
                total loss
            gamma: A discount factor for future rewards. For multi-dim reward,
                this can also be a list of discounts, each discount applies to a
                reward dim.
            td_lambda: Lambda parameter for TD-lambda computation.
            name: the name of the loss

        """
        super().__init__(name=name)
        self._td_error_loss_fn = td_error_loss_fn
        self._policy_kl_loss_weight = policy_kl_loss_weight
        self._gamma = gamma
        self._td_lambda = td_lambda

    def forward(self, info: PPGTrainInfo) -> LossInfo:
        """Computes loss based on the input PPGTrainInfo

        Args:

            info (PPGTrainInfo): provide the inputs for computing the loss,
                which includes the value targets, the value estimations, and the
                action distributions from both the old policy and the target
                policy

        """
        returns = self._calc_returns(info, info.rollout_value).detach()
        td_loss_actual = self._td_error_loss_fn(returns, info.value)
        td_loss_aux = self._td_error_loss_fn(returns, info.aux)
        policy_kl_loss = td.kl_divergence(info.rollout_action_distribution,
                                          info.action_distribution)

        # Compute the total loss by combing the above 3 components
        loss = td_loss_actual + td_loss_aux + self._policy_kl_loss_weight * policy_kl_loss

        return LossInfo(
            loss=loss,
            extra=PPGAuxPhaseLossInfo(
                td_loss_actual=td_loss_actual,
                td_loss_aux=td_loss_aux,
                policy_kl_loss=policy_kl_loss))

    def _calc_returns(self, info, value):

        if info.reward.ndim == 3:
            # [T, B, D] or [T, B, 1]
            discounts = info.discount.unsqueeze(-1) * self._gamma
        else:
            # [T, B]
            discounts = info.discount * self._gamma

        advantages = value_ops.generalized_advantage_estimation(
            rewards=info.reward,
            values=value,
            step_types=info.step_type,
            discounts=discounts,
            td_lambda=self._td_lambda)
        advantages = tensor_utils.tensor_extend_zero(advantages)
        returns = advantages + value

        return returns
