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
from alf.utils import math_ops


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

    clip_norm = torch.as_tensor(clip_norm)
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
        lambda t: clip_by_global_norm([t], clip_norm, in_place=in_place)[0],
        tensors)
