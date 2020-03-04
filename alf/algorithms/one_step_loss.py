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
from alf.data_structures import TrainingInfo, LossInfo
from alf.utils import common, losses, value_ops
from alf.utils import tensor_utils


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
        returns = tensor_utils.tensor_extend(returns, value[-1])
        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar('values', value.mean())
                alf.summary.scalar('returns', returns.mean())
        loss = self._td_error_loss_fn(returns.detach(), value)
        return LossInfo(loss=loss, extra=loss)
