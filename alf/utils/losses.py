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
        x (Tensor): prediction
        y (Tensor): target
    Returns:
        loss (Tensor)
    """
    return F.smooth_l1_loss(x, y, reduction="none")


@alf.configurable
def element_wise_squared_loss(x, y):
    """Elementwise squared loss.

    Args:
        x (Tensor): prediction
        y (Tensor): target
    Returns:
        loss (Tensor)
    """
    return F.mse_loss(x, y, reduction="none")


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


@alf.configurable
def multi_quantile_huber_loss(quantiles: torch.Tensor,
                              target: torch.Tensor,
                              delta: float = 0.1) -> torch.Tensor:
    """Multi-quantile Huber loss

    The loss for simultaneous multiple quantile regression. The number of quantiles
    n is ``quantiles.shape[-1]``. ``quantiles[..., k]`` is the quantile value
    estimation for quantile :math:`(k + 0.5) / n`. For each prediction, there
    can be one or multiple target values.

    This loss is described in the following paper:

    `Dabney et. al. Distributional Reinforcement Learning with Quantile Regression
    <https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17184/16590>`_

    Args:
        quantiles: batch_shape + [num_quantiles,]
        target: batch_shape or batch_shape + [num_targets, ]
        delta: the smoothness parameter for huber loss (larger means smoother).
    Returns:
        loss of batch_shape
    """
    num_quantiles = quantiles.shape[-1]
    t = torch.arange(0.5 / num_quantiles, 1., 1. / num_quantiles)
    if target.ndim == quantiles.ndim - 1:
        target = target.unsqueeze(-1)
    assert quantiles.shape[:-1] == target.shape[:-1]
    # [B, num_quantiles, num_samples]
    d = target[..., :, None] - quantiles[..., None, :]
    if delta == 0.0:
        loss = (t - (d < 0).float()) * d
    else:
        c = (t - (d < 0).float()).abs()
        d_abs = d.abs()
        loss = c * torch.where(d_abs < delta,
                               (0.5 / delta) * d**2, d_abs - 0.5 * delta)
    return loss.mean(dim=(-2, -1))
