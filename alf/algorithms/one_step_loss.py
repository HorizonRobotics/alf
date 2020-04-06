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
from alf.data_structures import TrainingInfo, LossInfo, StepType
from alf.utils import common, losses, value_ops
from alf.utils import tensor_utils
from alf.utils.summary_utils import safe_mean_hist_summary


@gin.configurable
class OneStepTDLoss(nn.Module):
    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=losses.element_wise_squared_loss,
                 debug_summaries=False,
                 name="OneStepLoss"):
        """
        Args:
            gamma (float): A discount factor for future rewards.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            debug_summaries (bool): True if debug summaries should be created
            name (str): The name of this loss.
        """
        super().__init__()
        self._gamma = gamma
        self._td_error_loss_fn = td_error_loss_fn
        self._debug_summaries = debug_summaries
        self._name = name

    def forward(self, training_info: TrainingInfo, value, target_value):
        returns = value_ops.one_step_discounted_return(
            rewards=training_info.reward,
            values=target_value,
            step_types=training_info.step_type,
            discounts=training_info.discount * self._gamma)
        value = value[:-1]
        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = training_info.step_type[:-1] != StepType.LAST
            with alf.summary.scope(self._name):
                alf.summary.scalar(
                    "explained_variance_of_return_by_value",
                    tensor_utils.explained_variance(value, returns, mask))
                safe_mean_hist_summary('values', value, mask)
                safe_mean_hist_summary('returns', returns, mask)
                safe_mean_hist_summary("td_error", returns - value, mask)
        loss = self._td_error_loss_fn(returns.detach(), value)

        # The shape of the loss expected by Algorith.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(loss=loss, extra=loss)

    @property
    def discount(self):
        return self._gamma
