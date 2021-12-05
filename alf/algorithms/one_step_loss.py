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
from alf.algorithms.td_loss import TDLoss, TDQRLoss
from alf.utils import losses


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
class OneStepTDQRLoss(TDQRLoss):
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
            num_quantiles=num_quantiles,
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            td_lambda=0.0,
            sum_over_quantiles=sum_over_quantiles,
            debug_summaries=debug_summaries,
            name=name)
