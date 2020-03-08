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
    return torch.cat((x, torch.zeros(x.shape[1:], dtype=x.dtype).unsqueeze(0)))


def explained_variance(ypred, y):
    """Computes fraction of variance that ypred explains about y.

    Adapted from baselines.ppo2 explained_variance()

    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    Args:
        ypred (Tensor): prediction for y
        y (Tensor): target
    Returns:
        1 - Var[y-ypred] / Var[y]
    """
    ypred = ypred.view(-1)
    y = y.view(-1)
    vary = torch.var(y, dim=0, unbiased=False)
    return 1 - torch.var(y - ypred, dim=0, unbiased=False) / (vary + 1e-30)


def _to_tensor(data, dtype=None):
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
        if alf.get_default_device() == "cuda":
            data = data.cuda()
    return data
