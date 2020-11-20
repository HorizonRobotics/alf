# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.utils import math_ops


def tensor_extend_new_dim(x, dim, n):
    """Extending the tensor along a new dimension with a replica of n.
    Args:
        x (Tensor): tensor to be extended
        dim (int): the value indicating the position of the newly
            inserted dimension
        n (int): the number of replica along dim
    Returns:
        the extended tensor. Its shape is (*x.shape[0:dim], n, *x.shape[dim:])
    """
    return x.unsqueeze(dim).expand(*x.shape[0:dim], n, *x.shape[dim:])


def reverse_cumsum(x, dim):
    """Perform cumsum in a reverse order along the dimension specified by dim.
    Args:
        x (Tensor): the tensor to compute the reverse cumsum on
        dim (int): the value indicating the dimension along which to calculate
            the reverse cumsum
    Returns:
        the reverse cumsumed tensor. It has the same shape as x.
    """
    return torch.flip(torch.cumsum(torch.flip(x, [dim]), dim), [dim])


def reverse_cumprod(x, dim):
    """Perform cumprod in a reverse order along the dimension specified by dim.
    Args:
        x (Tensor): the tensor to compute the reverse cumprod on
        dim (int): the value indicating the dimension along which to calculate
            the reverse cumprod
    Returns:
        the reverse cumprod tensor. It has the same shape as x.
    """
    return torch.flip(torch.cumprod(torch.flip(x, [dim]), dim), [dim])


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


def explained_variance(ypred, y, valid_mask=None):
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
    r = torch.where(vary == 0, vary, 1 - _var(y - ypred) / (vary + 1e-30))
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


def global_norm(tensors):
    """Adapted from TF's version.
    Computes the global norm of a nest of tensors. Given a nest of tensors
    `tensors`, this function returns the global norm of all tensors in `tensors`.
    The global norm is computed as:

        `global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`

    Any entries in `tensors` that are of type None are ignored.

    Args:
        tensors (nested Tensor): a nest of tensors

    Returns:
        norm (Tensor): a scalar tensor
    """
    assert alf.nest.is_nested(tensors), "tensors must be a nest!"
    tensors = alf.nest.flatten(tensors)
    if not tensors:
        return torch.zeros((), dtype=torch.float32)
    return torch.sqrt(
        sum([
            math_ops.square(torch.norm(torch.reshape(t, [-1])))
            for t in tensors if t is not None
        ]))


def clip_by_global_norm(tensors, clip_norm, use_norm=None, in_place=False):
    """Adapted from TF's version.
    Clips values of multiple tensors by the ratio of `clip_norm` to the global
    norm.

    Given a nest of tensors `tensors`, and a clipping norm threshold `clip_norm`,
    this function clips the tensors *in place* and returns the global norm
    (`global_norm`) of all tensors in `tensors`. Optionally, if you've already
    computed the global norm for `tensors`, you can specify the global norm with
    `use_norm`.

    To perform the clipping, each `tensor` are set to:
        tensor * clip_norm / max(global_norm, clip_norm)
    where:
        global_norm = sqrt(sum([l2norm(t)**2 for t in tensors]))

    If `clip_norm > global_norm` then the entries in `tensors` remain as they are,
    otherwise they're all shrunk by the global ratio.

    Any of the entries of `tensors` that are of type `None` are ignored.

    Args:
        tensors (nested Tensor): a nest of tensors to be clipped
        clip_norm (float or Tensor): a positive floating scalar
        use_norm (float or Tensor): the global norm to use. If None,
            `global_norm()` will be used to compute the norm.
        in_place (bool): If True, then the input `tensors` will be changed. For
            tensors that require grads, we cannot modify them in place; on the
            other hand, if you are clipping the gradients hold by an optimizer,
            then probably doing this in place will be easier.
    Returns:
        tensors (nested Tensor): the clipped tensors
        global_norm (Tensor): a scalar tensor representing the global norm. If
            `use_norm` is provided, it will be returned instead.
    """
    assert alf.nest.is_nested(tensors), "tensors must be a nest!"
    if use_norm is None:
        use_norm = global_norm(tensors)

    clip_norm = torch.as_tensor(clip_norm, dtype=torch.float32)
    assert clip_norm.ndim == 0, "clip_norm must be a scalar!"
    assert clip_norm > 0, "clip_norm must be positive!"

    scale = clip_norm / torch.max(clip_norm, use_norm)

    def _clip(tensor):
        if tensor is not None:
            if in_place:
                tensor.mul_(scale)
                return tensor
            else:
                return tensor * scale

    if scale < 1:
        tensors = alf.nest.map_structure(_clip, tensors)
    return tensors, use_norm


def clip_by_norms(tensors, clip_norm, in_place=False):
    """Clipping a nest of tensors *in place* to a maximum L2-norm.

    Given a tensor, and a maximum clip value `clip_norm`, this function
    normalizes the tensor so that its L2-norm is less than or equal to
    `clip_norm`.

    To perform the clipping:
        tensor * clip_norm / max(l2norm(tensor), clip_norm)

    Args:
        tensors (nested Tensor): a nest of tensors
        clip_norm (float or Tensor): a positive scalar
        in_place (bool): If True, then the input `tensors` will be changed. For
            tensors that require grads, we cannot modify them in place; on the
            other hand, if you are clipping the gradients hold by an optimizer,
            then probably doing this in place will be easier.

    Returns:
        the clipped tensors
    """
    return alf.nest.map_structure(
        lambda t: clip_by_global_norm([t], clip_norm, in_place=in_place)[0]
        if t is not None else t, tensors)


def cov(data, rowvar=False):
    """Estimate a covariance matrix given data.

    Args:
        data (tensor): A 1-D or 2-D tensor containing multiple observations 
            of multiple dimentions. Each row of ``mat`` represents a
            dimension of the observation, and each column a single
            observation.
        rowvar (bool): If True, then each row represents a dimension, with
            observations in the columns. Othewise, each column represents
            a dimension while the rows contains observations.

    Returns:
        The covariance matrix
    """
    x = data.detach().clone()

    if x.ndim > 3:
        raise ValueError('data has more than 3 dimensions')
    if x.ndim == 3:
        fact = 1.0 / (x.shape[1] - 1)
        x -= torch.mean(x, dim=1, keepdim=True)
        x_t = x.permute(0, 2, 1)
        out = fact * torch.bmm(x_t, x)
    else:
        if x.dim() < 2:
            x = x.view(1, -1)
        if not rowvar and x.size(0) != 1:
            x = x.t()
        fact = 1.0 / (x.shape[1] - 1)
        x -= torch.mean(x, dim=1, keepdim=True)
        out = fact * x.matmul(x.t()).squeeze()

    return out
