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
import torch
import torch.nn as nn

import alf

nest_map = alf.nest.map_structure


@alf.configurable
def identity(x):
    """PyTorch doesn't have an identity activation. This can be used as a
    placeholder.
    """
    return x


@alf.configurable
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


@alf.configurable
def swish(x):
    """Swish activation.

    This is suggested in arXiv:1710.05941

    Args:
        x (Tensor): input
    Returns:
        Tensor
    """
    return x * torch.sigmoid(x)


@alf.configurable
def softlower(x, low, hinge_softness=1.):
    """Softly lower bound ``x`` by ``low``, namely,
    ``softlower(x, low) = softplus(x - low) + low``

    Args:
        x (Tensor): input
        low (float|Tensor): the lower bound
        hinge_softness (float): this positive parameter changes the transition
            slope. A higher softness results in a smoother transition from
            ``low`` to identity. Default to 1.

    Returns:
        Tensor
    """
    assert hinge_softness > 0
    return nn.functional.softplus(x - low, beta=1. / hinge_softness) + low


@alf.configurable
def softupper(x, high, hinge_softness=1.):
    """Softly upper bound ``x`` by ``high``, namely,
    ``softupper(x, high) = -softplus(high - x) + high``.

    Args:
        x (Tensor): input
        high (float|Tensor): the upper bound
        hinge_softness (float): this positive parameter changes the transition
            slope. A higher softness results in a smoother transition from
            identity to ``high``. Default to 1.

    Returns:
        Tensor
    """
    assert hinge_softness > 0
    return -nn.functional.softplus(high - x, beta=1. / hinge_softness) + high


@alf.configurable
def softclip_tf(x, low, high, hinge_softness=1.):
    """Softly bound ``x`` in between ``[low, high]``, namely,

    .. code-block:: python

        clipped = softupper(softlower(x, low), high)
        softclip(x) = (clipped - high) / (high - softupper(low, high)) * (high - low) + high

    The second scaling step is because we will have
    ``softupper(low, high) < low`` due to distortion of softplus, so we need to
    shrink the interval slightly by ``(high - low) / (high - softupper(low, high))``
    to preserve the lower bound. Due to this rescaling, the bijector can be mildly
    asymmetric.

    Args:
        x (Tensor): input
        low (float|Tensor): the lower bound
        high (float|Tensor): the upper bound
        hinge_softness (float): this positive parameter changes the transition
            slope. A higher softness results in a smoother transition from
            ``low`` to ``high``. Default to 1.
    """
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(low)
    assert torch.all(high > low), "Invalid clipping range"

    softupper_high_at_low = softupper(low, high, hinge_softness)
    clipped = softupper(
        softlower(x, low, hinge_softness), high, hinge_softness)
    return ((clipped - high) / (high - softupper_high_at_low) * (high - low) +
            high)


@alf.configurable
def softclip(x, low, high, hinge_softness=1.):
    r"""Softly bound ``x`` in between ``[low, high]``. Unlike ``softclip_tf``,
    this transform is symmetric regarding the lower and upper bound when
    squashing. The softclip function can be defined in several forms:

    .. math::

        \begin{array}{lll}
            &\ln(\frac{e^{l-x}+1}{e^{x-h}+1}) + x & (1)\\
            =&\ln(\frac{e^{x-l}+1}{e^{x-h}+1}) + l & (2)\\
            =&\ln(\frac{e^{l-x}+1}{e^{h-x}+1}) + h & (3)\\
        \end{array}

    Args:
        x (Tensor): input
        low (float|Tensor): the lower bound
        high (float|Tensor): the upper bound
        hinge_softness (float): this positive parameter changes the transition
            slope. A higher softness results in a smoother transition from
            ``low`` to ``high``. Default to 1.
    """
    l, h, s = low, high, hinge_softness
    u = ((l - x) / s).exp()
    v = ((x - h) / s).exp()
    u1 = u.log1p()
    v1 = v.log1p()
    return torch.where(
        x < l, l + s * ((1 / u).log1p() - v1),
        torch.where(x > h, h + s * (u1 - (1 / v).log1p()), x + s * (u1 - v1)))


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
        rank-1 int64 Tensor representing the column of the first element in each
        row equal to the minimum of the row.
    """
    assert x.ndim == 2
    m, _ = x.min(dim=1, keepdim=True)
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


def normalize_min_max(x: torch.Tensor):
    """Normalize the min and max of each sample x[i] to 0 and 1.

    normalize x to [0, 1] as suggested in Appendix G. of MuZero paper.

    Args:
        x: a batch of samples
    Returns:
        Tensor: same shape as x
    """
    batch_size = x.shape[0]
    shape = [1] * x.ndim
    shape[0] = batch_size
    min = x.reshape(batch_size, -1).min(dim=1)[0].reshape(shape)
    max = x.reshape(batch_size, -1).max(dim=1)[0].reshape(shape)
    return (x - min) / (max - min + 1e-10)


class InvertibleTransform(object):
    """Base class for InvertibleTransform."""

    def transform(self, x):
        raise NotImplementedError

    def inverse_transform(self, y):
        raise NotImplementedError


@alf.repr_wrapper
class SqrtLinearTransform(InvertibleTransform):
    r"""The transformation used by MuZero.

    .. math::

        y=sign(x) (\sqrt{|x| +1} - 1) + \epsilon x

    Args:
        eps: :math:`\epsilon` in the above formula
    """

    def __init__(self, eps: float = 1e-3):
        self._eps = eps

    def transform(self, x):
        return x.sign() * ((x.abs() + 1).sqrt() - 1) + self._eps * x

    def inverse_transform(self, y):
        a = (1 + 4 * self._eps * (y.abs() + (1 + self._eps))).sqrt() - 1
        return y.sign() * ((a / (2 * self._eps))**2 - 1)


@alf.repr_wrapper
class Sqrt1pTransform(InvertibleTransform):
    r"""The transformation used by MuZero with epsilon = 0.

    .. math::

        y=sign(x) (\sqrt{|x| +1} - 1) = x / (\sqrt{|x|+1} + 1)

    The second form has better numerical precision for small x.
    """

    def transform(self, x):
        return x / ((x.abs() + 1).sqrt() + 1)

    def inverse_transform(self, y):
        # y.sign() * ((y.abs() + 1) ** 2 - 1)
        # = y.sign() * (y^2 + 2|y| + 1 - 1)
        # = y.sign() * (y^2 + 2|y|)
        # = |y|y + 2y = (|y| + 2) y
        return (y.abs() + 2) * y


@alf.repr_wrapper
class Log1pTransform(InvertibleTransform):
    r"""Implementing the following transformation:

    .. math::

        y=\alpha sign(x)\log(1+|x|)

    Args:
        alpha: :math:`\alpha` in the above formula
    """

    def __init__(self, alpha: float = 20):
        self._alpha = alpha

    def transform(self, x):
        return self._alpha * x.sign() * torch.log1p(x.abs())

    def inverse_transform(self, y):
        return y.sign() * ((y / self._alpha).abs().exp() - 1)


def binary_neg_entropy(p: torch.Tensor):
    """Negative entropy for binary outcome.

    Args:
        p: the probability of one outcome and hence 1-p are the probabilities for
            the other outcome
    Returns:
        Tensor with the same shape as p
    """
    q = 1 - p
    return p.xlogy(p) + q.xlogy(q)
