# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from typing import SupportsFloat
from torch.optim.swa_utils import AveragedModel as _AveragedModel

from alf.utils.schedulers import Scheduler


def ema_avg_fn(averaged_model_parameter: torch.Tensor,
               model_parameter: torch.Tensor,
               num_averaged: int,
               ema_rate: SupportsFloat | Scheduler,
               starting_average_after=0,
               begin_with_simple_average=True):
    """Exponential moving average of model parameters.

    Args:
        averaged_model_parameter (torch.Tensor): the current value of the
            averaged model parameter
        model_parameter (torch.Tensor): the current value of the model
            parameter
        num_averaged (int): the number of models already averaged
        ema_rate: the exponential moving average rate. Smaller value means
            more smoothing. If ema_rate is 0 and begin_with_simple_average is True,
            it is always simple average.
        starting_average_after (int): start applying ema after this many
            models have been averaged. Before that, no average is applied.
        begin_with_simple_average (bool): if True, before applying ema,
            simple average is applied until (num_averaged - starting_average_after)
            is at least 1/ema_rate.
    """
    if num_averaged <= starting_average_after:
        return model_parameter
    if not isinstance(ema_rate, SupportsFloat):
        assert isinstance(
            ema_rate, Scheduler), ("ema_rate must be a number or a Scheduler")
        ema_rate = ema_rate()
    if begin_with_simple_average:
        ema_rate = max(ema_rate,
                       1 / (num_averaged + 1 - starting_average_after))
    return torch.lerp(averaged_model_parameter, model_parameter, ema_rate)


class AveragedModel(_AveragedModel):
    """torch.optim.swa_utils.AveragedModel with additional call() method."""

    def call(self, name, *args, **kwargs):
        """ Calls a method of the underlying model.

        Args:
            name (str): the name of the method to be called
            *args: positional arguments to be passed to the method
            **kwargs: keyword arguments to be passed to the method
        Returns:
            the result of the method call
        """
        return getattr(self.module, name)(*args, **kwargs)
