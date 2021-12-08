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
"""Various function/classes related to loss computation."""

import torch
import torch.nn.functional as F

import alf


@alf.configurable
def element_wise_huber_loss(x, y):
    """Elementwise Huber loss.

    Args:
        x (Tensor): label
        y (Tensor): prediction
    Returns:
        loss (Tensor)
    """
    return F.smooth_l1_loss(y, x, reduction="none")


@alf.configurable
def element_wise_squared_loss(x, y):
    """Elementwise squared loss.

    Args:
        x (Tensor): label
        y (Tensor): prediction
    Returns:
        loss (Tensor)
    """
    return F.mse_loss(y, x, reduction="none")


@alf.configurable
def huber_function(x: torch.Tensor, delta: float = 1.0):
    """Huber function.

    Args:
        x: difference between the observed and predicted values
        delta: the threshold at which to change between delta-scaled 
            L1 and L2 loss, must be positive. Default value is 1.0
    Returns:
        Huber function (Tensor)
    """
    return torch.where(x.abs() <= delta, 0.5 * x**2,
                       delta * (x.abs() - 0.5 * delta))
