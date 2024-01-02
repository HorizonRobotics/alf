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
from typing import Union, List, Callable

import alf
from alf.algorithms.td_loss import TDLoss, TDQRLoss
from alf.utils import losses


@alf.configurable
class OneStepTDLoss(TDLoss):
    def __init__(self,
                 gamma: Union[float, List[float]] = 0.99,
                 td_error_loss_fn: Callable = losses.element_wise_squared_loss,
                 debug_summaries: bool = False,
                 name: str = "OneStepTDLoss"):
        """
        Args:
            gamma: A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_error_loss_fn: A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            debug_summaries: True if debug summaries should be created
            name: The name of this loss.
        """
        super().__init__(
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            debug_summaries=debug_summaries,
            td_lambda=0.0,
            name=name)


@alf.configurable
class OneStepTDQRLoss(TDQRLoss):
    """One step temporal difference quantile regression loss. """

    def __init__(self,
                 num_quantiles: int = 50,
                 gamma: Union[float, List[float]] = 0.99,
                 td_error_loss_fn: Callable = losses.iqn_huber_loss,
                 sum_over_quantiles: bool = False,
                 debug_summaries: bool = False,
                 name: str = "OneStepTDQRLoss"):
        """
        Args:
            num_quantiles: the number of quantiles.
            gamma: A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_error_loss_fn: A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            sum_over_quantiles: If True, the quantile regression loss will
                be summed along the quantile dimension. Otherwise, it will be
                averaged along the quantile dimension instead. Default is False.
            debug_summaries: True if debug summaries should be created
            name: The name of this loss.
        """
        super().__init__(
            num_quantiles=num_quantiles,
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            td_lambda=0.0,
            sum_over_quantiles=sum_over_quantiles,
            debug_summaries=debug_summaries,
            name=name)
