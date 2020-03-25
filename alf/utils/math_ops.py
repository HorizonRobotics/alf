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
"""Various math ops."""

import gin
import torch

import alf

nest_map = alf.nest.map_structure


def identity(x):
    """PyTorch doesn't have an identity activation. This can be used as a
    placeholder.
    """
    return x


@gin.configurable
def clipped_exp(value, clip_value_min=-20, clip_value_max=2):
    """ Clip value to the range [`clip_value_min`, `clip_value_max`]
    then compute exponential

    Args:
         value (Tensor): input tensor.
         clip_value_min (float): The minimum value to clip by.
         clip_value_max (float): The maximum value to clip by.
    """
    value = torch.clamp(value, clip_value_min, clip_value_max)
    return torch.exp(value)


def add_ignore_empty(x, y):
    """Add two Tensors which may be None or ().

     If x or y is None, they are assumed to be zero and the other tensor is
     returned.

     Args:
          x (Tensor|None|()):
          y (Tensor(|None|())):
     Returns:
          x + y
     """

    def _ignore(t):
        return t is None or (isinstance(t, tuple) and len(t) == 0)

    if _ignore(y):
        return x
    elif _ignore(x):
        return y
    else:
        return x + y


@gin.configurable
def swish(x):
    """Swish activation.

    This is suggested in arXiv:1710.05941

    Args:
        x (Tensor): input
    Returns:
        Tensor
    """
    return x * torch.sigmoid(x)


def max_n(inputs):
    """Calculate the maximum of n Tensors

    Args:
        inputs (list[Tensor]): list of Tensors, should have the same shape
    Returns:
        the elementwise maximum of all the tensors in `inputs`
    """
    ret = inputs[0]
    inputs = inputs[1:]
    for x in inputs:
        ret = torch.max(ret, x)
    return ret


def min_n(inputs):
    """Calculate the minimum of n Tensors

    Args:
        inputs (list[Tensor]): list of Tensors, should have the same shape
    Returns:
        the elementwise minimum of all the tensors in `inputs`
    """
    ret = inputs[0]
    inputs = inputs[1:]
    for x in inputs:
        ret = torch.min(ret, x)
    return ret


def add_n(inputs):
    """Calculate the sum of n Tensors

    Args:
        inputs (list[Tensor]): list of Tensors, should have the same shape
    Returns:
        the elementwise sum of all the tensors in `inputs`
    """
    ret = inputs[0]
    inputs = inputs[1:]
    for x in inputs:
        ret = torch.add(ret, x)
    return ret


def square(x):
    """torch doesn't have square."""
    return torch.pow(x, 2)


def weighted_reduce_mean(x, weight, dim=()):
    """Weighted mean.

    Args:
        x (Tensor): values for calculating the mean
        weight (Tensor): weight for x. should have same shape as `x`
        dim (int | tuple[int]): The dimensions to reduce. If None (the
            default), reduces all dimensions. Must be in the range
            [-rank(x), rank(x)). Empty tuple means to sum all elements.
    Returns:
        the weighted mean across `axis`
    """
    weight = weight.to(torch.float32)
    sum_weight = weight.sum(dim=dim)
    sum_weight = torch.max(sum_weight, torch.tensor(1e-10))
    return nest_map(lambda y: (y * weight).sum(dim=dim) / sum_weight, x)


def sum_to_leftmost(value, dim):
    """Sum out `value.ndim-dim` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of `.ndim` at least `dim`.
        dim (int): The number of leftmost dims to remain.
    Returns:
        The result tensor whose ndim is `min(dim, value.dim)`.
    """
    if value.ndim <= dim:
        return value
    return value.sum(list(range(dim, value.ndim)))


def argmin(x):
    """Deterministic argmin.

    Different from torch.argmin, which may have undetermined result if the are
    multiple elements equal to the min, this argmin is guaranteed to return the
    index of the first element equal to the min in each row.

    Args:
        x (Tensor): only support rank-2 tensor
    Returns:
        rank-1 int64 Tensor represeting the column of the first element in each
        row equal to the minimum of the row.
    """
    assert x.ndim == 2
    m, _ = x.min(dim=1, keepdims=True)
    r, c = torch.nonzero(x == m, as_tuple=True)
    r, num_mins = torch.unique(r, return_counts=True)
    i = torch.cumsum(num_mins, 0)
    i = torch.cat([torch.tensor([0]), i[:-1]])
    return c[i]
