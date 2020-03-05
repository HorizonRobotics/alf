# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
"""Collection of tensor utility functions."""

import numpy as np
import torch

import alf


def tensor_extend(x, y):
    """Extending tensor with new_slice.

    new_slice.shape should be same as tensor.shape[1:]
    Args:
        x (Tensor): tensor to be extended
        y (Tensor): the tensor which will be appended to `x`
    Returns:
        the extended tensor. Its shape is (x.shape[0]+1, x.shape[1:])
    """
    return torch.cat((x, y.unsqueeze(0)))


def tensor_extend_zero(x):
    """Extending tensor with zeros.

    new_slice.shape should be same as tensor.shape[1:]
    Args:
        x (Tensor): tensor to be extended
    Returns:
        the extended tensor. Its shape is (x.shape[0]+1, x.shape[1:])
    """
    return torch.cat((x, torch.zeros(1, *x.shape[1:], dtype=x.dtype)))


def tensor_prepend(x, y):
    """Prepending tensor with y.

    y.shape should be same as tensor.shape[1:]
    Args:
        x (Tensor): tensor to be prepended
        y (Tensor): the tensor which will be appended to `x`
    Returns:
        the prepended tensor. Its shape is (x.shape[0]+1, x.shape[1:])
    """
    return torch.cat([y.unsqueeze(0), x])


def tensor_prepend_zero(x):
    """Prepending tensor with zeros.

    Args:
        x (Tensor): tensor to be extended
    Returns:
        the prepended tensor. Its shape is (x.shape[0]+1, x.shape[1:])
    """
    return torch.cat((torch.zeros(1, *x.shape[1:], dtype=x.dtype), x))


def explained_variance(ypred, y, valid_mask):
    """Computes fraction of variance that ypred explains about y.

    Adapted from baselines.ppo2 explained_variance()

    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    Args:
        ypred (Tensor): prediction for y
        y (Tensor): target
        valid_mask (Tensor): an optional
    Returns:
        1 - Var[y-ypred] / Var[y]
    """
    if valid_mask is not None:
        n = torch.max(valid_mask.sum().to(y.dtype),
                      torch.tensor(1, dtype=y.dtype))
    else:
        n = np.prod(y.shape)

    def _var(x):
        if valid_mask is not None:
            x = x * valid_mask
        mean = x.sum() / n
        x2 = (x - mean)**2
        if valid_mask is not None:
            x2 = x2 * valid_mask
        var = x2.sum() / n
        return var

    vary = _var(y)
    r = 1 - _var(y - ypred) / (vary + 1e-30)
    return r


def to_tensor(data, dtype=None):
    """Convert the data to a torch tensor.

    Args:
        data (array like): data for the tensor. Can be a list, tuple,
            numpy ndarray, scalar, and other types.
        dtype (torch.dtype): dtype of the converted tensors.

    Returns:
        A tensor of dtype
    """
    if not torch.is_tensor(data):
        # as_tensor reuses the underlying data store of numpy array if possible.
        data = torch.as_tensor(data, dtype=dtype).detach()
    return data
