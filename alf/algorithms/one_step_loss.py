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
from alf.utils import common, losses, value_ops
from alf.utils import tensor_utils
from alf.utils.summary_utils import safe_mean_hist_summary
from alf.algorithms.td_loss import TDLoss


@alf.configurable
class OneStepTDLoss(TDLoss):
    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=losses.element_wise_squared_loss,
                 debug_summaries=False,
                 name="TDLoss"):
        """
        Args:
            gamma (float|list[float]): A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_errors_loss_fn (Callable): A function for computing the TD errors
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
