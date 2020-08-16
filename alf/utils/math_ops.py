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

import functools
import gin
import torch

import alf

nest_map = alf.nest.map_structure


@gin.configurable
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
    """Calculate the maximum of n tensors.

    Args:
        inputs (iterable[Tensor]): an iterable of tensors. It requires that
            all tensor shapes can be broadcast to the same shape.
    Returns:
        Tensor: the element-wise maximum of all the tensors in ``inputs``.
    """
    return functools.reduce(torch.max, inputs)


def min_n(inputs):
    """Calculate the minimum of n tensors.

    Args:
        inputs (iterable[Tensor]): an iterable of tensors. It requires that
            all tensor shapes can be broadcast to the same shape.
    Returns:
        Tensor: the element-wise minimum of all the tensors in ``inputs``.
    """
    return functools.reduce(torch.min, inputs)


def add_n(inputs):
    """Calculate the sum of n tensors.

    Args:
        inputs (iterable[Tensor]): an iterable of tensors. It requires that
            all tensor shapes can be broadcast to the same shape.
    Returns:
        Tensor: the element-wise sum of all the tensors in ``inputs``.
    """
    return sum(inputs)


def mul_n(inputs):
    """Calculate the product of n tensors.

    Args:
        inputs (iterable[Tensor]): an iterable of tensors. It requires that
            all tensor shapes can be broadcast to the same shape.
    Returns:
        Tensor: the element-wise multiplication of all the tensors in ``inputs``.
    """
    return functools.reduce(torch.mul, inputs)


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


def shuffle(values):
    """Shuffle a nest.

    Shuffle all the tensors in ``values`` by a same random order.

    Args:
        values (nested Tensor): nested Tensor to be shuffled. All the tensor
            need to have the same batch size (i.e. shape[0]).
    Returns:
        shuffled value along dimension 0.
    """
    batch_size = alf.nest.get_nest_batch_size(values)
    indices = torch.randperm(batch_size)
    return nest_map(lambda value: value[indices], values)


class Softsign_(torch.autograd.Function):
    r"""Inplace version of softsign function.

    Applies element-wise inplace, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

    The `current pytorch implementation of softsign
    <https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#softsign>`_
    is inefficient for backward because it relies on automatic differentiation
    and does not have an inplace version. Hence we provide a more efficient
    implementation.

    Reference:
    `PyTorch: Defining New Autograd Functions
    <https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html>`_
    """

    @staticmethod
    def forward(ctx, input):
        output = torch.div(input, input.abs() + 1, out=input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return torch.mul(grad_output, torch.pow(1 - output.abs(), 2))


softsign_ = Softsign_.apply


class Softsign(torch.autograd.Function):
    r"""Softsign function.

    Applies element-wise, the function :math:`\text{SoftSign}(x) = \frac{x}{1 + |x|}`

    Compared to ``Softsign_``, this uses more memory but is faster and has higher precision
    for backward.
    """

    @staticmethod
    def forward(ctx, input):
        x = torch.pow(input.abs() + 1, -1)
        output = torch.mul(input, x)
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return torch.mul(grad_output, torch.pow(x, 2))


softsign = Softsign.apply
