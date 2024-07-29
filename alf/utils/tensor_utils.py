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

from typing import Tuple

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
        Tensor: the extended tensor. Its shape is ``(*x.shape[0:dim], n, *x.shape[dim:])``
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
    """Extending tensor ``x`` with new_slice ``y``.

    ``y.shape`` should be same as ``x.shape[1:]``

    Args:
        x (Tensor): tensor to be extended
        y (Tensor): the tensor which will be appended to `x`
    Returns:
        Tensor: the extended tensor. Its shape is ``(x.shape[0]+1, x.shape[1:])``
    """
    return torch.cat((x, y.unsqueeze(0)))


def tensor_extend_zero(x, dim=0):
    """Extending tensor with zeros along an axis.

    Args:
        x (Tensor): tensor to be extended
        dim (int): the axis to extend zeros
    Returns:
        Tensor: the extended tensor. Its shape is
            ``(*x.shape[:dim], x.shape[dim]+1, *x.shape[dim+1:])``
    """
    zero_shape = list(x.shape)
    zero_shape[dim] = 1
    zeros = torch.zeros(zero_shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, zeros), dim=dim)


def tensor_prepend(x, y):
    """Prepending tensor with y.

    y.shape should be same as tensor.shape[1:]
    Args:
        x (Tensor): tensor to be prepended
        y (Tensor): the tensor which will be appended to `x`
    Returns:
        Tensor: the prepended tensor. Its shape is ``(x.shape[0]+1, x.shape[1:])``
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


def explained_variance(ypred, y, valid_mask=None, dim=None):
    """Computes fraction of variance that ypred explains about y.

    Adapted from baselines.ppo2 explained_variance()

    Interpretation:

        * ev=0:  might as well have predicted zero
        * ev=1:  perfect prediction
        * ev<0:  worse than just predicting zero

    Args:
        ypred (Tensor): prediction for y
        y (Tensor): target
        valid_mask (Tensor): an optional
        dim (None|int): the dimension to reduce. If not provided, the explained
            variance is calculated for all dimensions.
    Returns:
        1 - Var[y-ypred] / Var[y]
    """
    if dim is None:
        if valid_mask is not None:
            valid_mask = valid_mask.reshape(-1)
        return explained_variance(
            ypred.reshape(-1), y.reshape(-1), valid_mask, dim=0)

    if valid_mask is not None:
        n = torch.max(
            valid_mask.sum(dim=dim).to(y.dtype), torch.tensor(
                1, dtype=y.dtype))
    else:
        n = y.shape[dim]

    def _var(x):
        if valid_mask is not None:
            x = x * valid_mask
        mean = x.sum(dim=dim, keepdims=True) / n
        x2 = (x - mean)**2
        if valid_mask is not None:
            x2 = x2 * valid_mask
        var = x2.sum(dim=dim) / n
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
    """Computes the global norm of a nest of tensors.

    Adapted from TF's version.

    Given a nest of tensors ``tensors``, this function returns the global norm
    of all tensors in ``tensors``. The global norm is computed as:

    .. code-block:: python

        global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

    Any entries in ``tensors`` that are of type ``None`` are ignored.

    Args:
        tensors (nested Tensor): a nest of tensors

    Returns:
        norm (Tensor): a scalar tensor
    """
    assert alf.nest.is_nested(tensors), "tensors must be a nest! %s" % tensors
    tensors = alf.nest.flatten(tensors)
    if not tensors:
        return torch.zeros((), dtype=torch.float32)
    return torch.sqrt(
        sum([
            math_ops.square(torch.norm(torch.reshape(t, [-1])))
            for t in tensors if t is not None
        ]))


def clip_by_global_norm(tensors, clip_norm, use_norm=None, in_place=False):
    """Clips values of multiple tensors by the ratio of ``clip_norm`` to the global
    norm.

    Adapted from TF's version.

    Given a nest of tensors ``tensors``, and a clipping norm threshold ``clip_norm``,
    this function clips the tensors *in place* and returns the global norm
    (``global_norm``) of all tensors in ``tensors``. Optionally, if you've already
    computed the global norm for `tensors`, you can specify the global norm with
    ``use_norm``.

    To perform the clipping, each `tensor` are set to:

    .. code-block:: python

        tensor * clip_norm / max(global_norm, clip_norm)

    where:

    .. code-block:: python

        global_norm = sqrt(sum([l2norm(t)**2 for t in tensors]))

    If ``clip_norm > global_norm`` then the entries in ``tensors`` remain as they are,
    otherwise they're all shrunk by the global ratio.

    Any of the entries of ``tensors`` that are of type `None` are ignored.

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

    scale = clip_norm / use_norm

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
            of multiple dimensions. Each row of ``mat`` represents a
            dimension of the observation, and each column a single
            observation.
        rowvar (bool): If True, then each row represents a dimension, with
            observations in the columns. Otherwise, each column represents
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


def scale_gradient(tensor, scale, clone_input=True):
    """Scales the gradient of `tensor` for the backward pass.
    Args:
        tensor (Tensor): a tensor which requires gradient.
        scale (float): a scalar factor to be multiplied to the gradient
            of `tensor`.
        clone_input (bool): If True, clone the input tensor before applying
            gradient scaling. This option is useful when there are multiple
            computational branches originated from `tensor` and we want to
            apply gradient scaling to part of them without impacting the rest.
            If False, apply gradient scaling to the input tensor directly.
    Returns:
        The (cloned) tensor with gradient scaling hook registered.
    """
    if not tensor.requires_grad:
        return tensor
    if clone_input:
        output = tensor.clone()
    else:
        output = tensor
    output.register_hook(lambda grad: grad * scale)
    return output


class BatchSquash(object):
    """Facilitates flattening and unflattening batch dims of a tensor. Copied
    from `tf_agents`.

    Exposes a pair of matched flatten and unflatten methods. After flattening
    only 1 batch dimension will be left. This facilitates evaluating networks
    that expect inputs to have only 1 batch dimension.
    """

    def __init__(self, batch_dims):
        """Create two tied ops to flatten and unflatten the front dimensions.

        Args:
            batch_dims (int): Number of batch dimensions the flatten/unflatten
                ops should handle.

        Raises:
            ValueError: if batch dims is negative.
        """
        if batch_dims < 0:
            raise ValueError('Batch dims must be non-negative.')
        self._batch_dims = batch_dims
        self._original_tensor_shape = None

    def flatten(self, tensor):
        """Flattens and caches the tensor's batch_dims."""
        if self._batch_dims == 1:
            return tensor
        self._original_tensor_shape = tensor.shape
        return torch.reshape(tensor,
                             (-1, ) + tuple(tensor.shape[self._batch_dims:]))

    def unflatten(self, tensor):
        """Unflattens the tensor's batch_dims using the cached shape."""
        if self._batch_dims == 1:
            return tensor

        if self._original_tensor_shape is None:
            raise ValueError('Please call flatten before unflatten.')

        return torch.reshape(
            tensor, (tuple(self._original_tensor_shape[:self._batch_dims]) +
                     tuple(tensor.shape[1:])))


def append_coordinate(im: torch.Tensor):
    """For the image, we append coordinates as two channels. The image is assumed
    to be channel-first. The coordinates will range from -1 to 1 evenly.

    Args:
        im: an image of shape ``[B,C,H,W]``.
    Returns:
        torch.Tensor: an output image of shape ``[B,C+2,H,W]`` where the extra 2
            dimensions are xy meshgrid from -1 to 1.
    """
    assert len(im.shape) == 4, "Image must have a shape of [B,C,H,W]!"
    y = torch.arange(-1., 1., step=2. / im.shape[-2])
    x = torch.arange(-1., 1., step=2. / im.shape[-1])
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    # [H,W] -> [B,H,W]
    yy = alf.utils.tensor_utils.tensor_extend_new_dim(yy, dim=0, n=im.shape[0])
    xx = alf.utils.tensor_utils.tensor_extend_new_dim(xx, dim=0, n=im.shape[0])
    # [B,C+2,H,W]
    return torch.cat([im, yy.unsqueeze(1), xx.unsqueeze(1)], dim=1)


def spatial_broadcast(z: torch.Tensor, im_shape: Tuple[int]):
    """Broadcasting an embedding across the image spatial domain. The image shape
    is assumed to be channel-first.

    Args:
        z: embedding of shape ``[...,D]`` to be broadcast spatially
        im_shape: a tuple of ints where the last two are height and width.
    Returns:
        torch.Tensor: a broadcast image of spec ``[...,D,H,W]`` where ``D`` is the
            input embedding size and ``[H,W]`` are input height and width.
    """
    return z.reshape(z.shape + (1, 1)).expand(*(z.shape + im_shape[-2:]))
