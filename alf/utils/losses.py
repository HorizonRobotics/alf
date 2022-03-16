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
from typing import Optional, Tuple

import alf
from alf.utils.math_ops import InvertibleTransform


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
            Note that the quantile estimation with delta > 0 is biased. You should
            use a small value for ``delta`` if you want the quantile estimation
            to be less biased (so that the mean of the quantile will be close
            to mean of the samples).
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


class ScalarPredictionLoss(object):
    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        """Calculate the loss given ``pred`` and ``target``.

        Args:
            pred: raw prediction
            target: target value
        Returns:
            loss with the same shape as target
        """
        raise NotImplementedError()

    def calc_expectation(self, pred: torch.Tensor):
        """Calculate the expected predition in the untransfomred domain from ``pred``.
        """
        raise NotImplementedError()

    def initialize_bias(self, bias: torch.Tensor):
        """Initialize the bias of the last FC layer for the prediction properly.

        This function can be passed to FC as bias_initializer.

        For some losses (e.g. OrderedDiscreteRegresion), initializing bias to
        zero can have very bad initial predictions. So we provide an interface
        for doing loss specific intializations. Note that the weight of the last
        FC should be initialized to zero in general.

        Args:
            bias: the bias parameter to be initialized.
        """
        with torch.no_grad():
            bias.zero_()


@alf.repr_wrapper
class SquareLoss(ScalarPredictionLoss):
    """Square loss for predicting scalar target.

    Args:
        transform: the transformation applied to target. If it is provided, the
            the regression target will be transformed.
    """

    def __init__(self, transform: Optional[InvertibleTransform] = None):
        super().__init__()
        self._transform = transform

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        """Calculate the loss.

        Args:
            pred: shape is [B]
            target: the shape is [B]
        Returns:
            loss with the same shape as target
        """
        assert pred.shape == target.shape

        if self._transform is not None:
            target = self._transform.transform(target)
        return (pred - target)**2

    def calc_expectation(self, pred: torch.Tensor):
        """Calculate the expected predition in the untransfomred domain from ``pred``.

        Args:
            pred: raw model prediction
        """
        if self._transform is not None:
            pred = self._transform.inverse_transform(pred)
        return pred


def _get_indexer(shape: Tuple[int]):
    """Return a tuple of Tensors which can be used to index a Tensor.

    The purpose of this function can be better illustrated by an example.
    Suppose ``shape`` is ``[n0, n1, n2]``. Then shape of the three returned
    Tensors will be: [n0, 1, 1], [n1, 1], [n2]. And each of them has elements
    ranging from 0 to n0-1, n1-1 and n2-1 respectively. The returned tuple ``B``
    can be combined with another int64 Tensor ``I`` of shape ``[n0, n1, n2]``
    to access the element of a Tensor ``X`` with shape``[n0, n1, n2, n]`` as
    ``Y=X[B + (I,)]`` so that ``Y[i,j,k] = X[i, j, k, I[i,j,k]]``

    Args:
        shape: The shape of the tensor to be accessed exclusing the last dimension.
    Returns:
        the tuple of index for accessing the tensor.
    """
    ndim = len(shape)
    ones = [1] * ndim
    B = tuple(
        torch.arange(d).reshape(d, *ones[i + 1:]) for i, d in enumerate(shape))
    return B


class _DiscreteRegressionLossBase(ScalarPredictionLoss):
    """The base class for DiscreteRegressionLoss and OrderedDiscreteRegresionLoss."""

    def __init__(self,
                 transform: Optional[InvertibleTransform] = None,
                 inverse_after_mean=False):
        super().__init__()
        self._transform = transform
        if self._transform is not None:
            self._inverse_after_mean = inverse_after_mean
        else:
            self._inverse_after_mean = True
        self._support = None

    def _calc_support(self, n: int):
        if self._support is not None and self._support.shape[0] == n:
            return self._support
        upper_bound = n // 2
        lower_bound = -((n - 1) // 2)
        x = torch.arange(lower_bound, upper_bound + 1, dtype=torch.float32)
        if self._transform is not None and not self._inverse_after_mean:
            x = self._transform.inverse_transform(x)
        self._support = x
        return x

    def _calc_bin(self, logits, target):
        """Discretize ``target`` such that:

        bin1 <= transform(target) - lower_bound < bin2
        and w2 is the weight assign to bin2. Hence 1 - w1 is the weight assigned
        to bin1.
        If inverse_after_mean is False,  w2 is chosen so that the expectation
        will be equal to target.
        If inverse_after_mean is True, w2 is simply ``transform(target) - lower_bound - bin1``
        """
        assert logits.shape[:-1] == target.shape
        n = logits.shape[-1]
        lower_bound = -((n - 1) // 2)
        upper_bound = n // 2
        original_target = target
        if self._transform is not None:
            target = self._transform.transform(target)
        target = target.clamp(min=lower_bound, max=upper_bound)
        low = target.floor()
        high = low + 1
        bin1 = low.to(torch.int64) - lower_bound
        bin2 = (bin1 + 1).clamp(max=n - 1)
        if self._inverse_after_mean:
            w2 = target - low
        else:
            low = self._transform.inverse_transform(low)
            high = self._transform.inverse_transform(high)
            w2 = (original_target - low) / (high - low)
        return bin1, bin2, w2


@alf.repr_wrapper
class DiscreteRegressionLoss(_DiscreteRegressionLossBase):
    r"""A loss for predicting the distribution of a scalar.

    The target is assumed to be in the range ``[-(n-1)//2, n//2]``, where ``n=logits.shape[-1]``.
    The logits are used to calculate the probabilities of being one of the ``n``
    values. If a target value y is not an integer, it is treated as having
    prabability mass of :math:`y- \lfloor y \rfloor` at :math:`\lfloor y \rfloor + 1`
    and probability mass of :math:`1 + \lfloor y \rfloor - y` at :math:`\lfloor y \rfloor`.
    Then cross entropy loss is applied.

    More specifically, the ``logits`` passed to ``calc_loss`` represents the following:
    P = softmax(logits) and P[i] means the probability that the (transformed)
    ``target`` is equal to ``i - (n-1)//2``

    Note: ``DescreteRegressionLoss(SqrtLinearTransform(0.001), inverse_after_mean=True)``
        is the loss used by MuZero paper.

    Args:
        transform: the transformation applied to target. If it is provided, the
            the regression target will be transformed.
        inverse_after_mean: when calculating the expected prediction, whether to
            do the inverse transformation after calculating the the expectation
            in the transformed space. Note that using ``inverse_after_mean=True``
            will make the expectation biased in general. This is because
            :math:`f^{-1}(E(x)) \le E(f^{-1}(x))` (Jensen inequality) if
            :math:`f^{-1}` is convex. In our case, :math:`f^{-1}` is convex for
            :math:`x \ge 0`.
    """

    def __call__(self, logits: torch.Tensor, target: torch.Tensor):
        """Caculate the loss.

        Args:
            logits: shape is [B, n]
            target: the shape is [B]
        Returns:
            loss with the same shape as target
        """
        bin1, bin2, w2 = self._calc_bin(logits, target)
        w1 = 1 - w2
        nlp = -F.log_softmax(logits, dim=-1)
        B = _get_indexer(logits.shape[:-1])
        loss = w1 * nlp[B + (bin1, )] + w2 * nlp[B + (bin2, )]
        return loss

    def calc_expectation(self, logits):
        """Calculate the expected predition in the untransfomred domain from ``pred``.

        Args:
            pred: raw model prediction
        """
        support = self._calc_support(logits.shape[-1])
        ret = torch.mv(logits.softmax(dim=-1), support)
        if self._inverse_after_mean and self._transform is not None:
            ret = self._transform.inverse_transform(ret)
        return ret

    def initialize_bias(self, bias: torch.Tensor):
        """Initialize the bias of the last FC layer for the prediction properly.

        This function set the bias so that the initial distribution of the prediction
        in the original domain of target is approximatedly Cauchy: :math:`p(x) \propto \frac{1}{1+x^2}`

        Args:
            bias: the bias parameter to be initialized.
        """
        assert bias.ndim == 1
        n = bias.shape[0]
        upper_bound = n // 2
        lower_bound = -((n - 1) // 2)
        x = torch.arange(lower_bound, upper_bound + 1, dtype=torch.float32)
        x1 = x - 0.5
        x2 = x + 0.5
        if self._transform is not None:
            x = self._transform.inverse_transform(x)
            x1 = self._transform.inverse_transform(x1)
            x2 = self._transform.inverse_transform(x2)
        probs = (x2 - x1) / (x**2 + 1)
        probs = probs / probs.sum()
        with torch.no_grad():
            bias.copy_(probs.log())


@alf.repr_wrapper
class OrderedDiscreteRegressionLoss(_DiscreteRegressionLossBase):
    r"""A loss for predicting the distribution of a scalar.

    The target is assumed to be in the range ``[-(n-1)//2, n//2]``, where ``n=logits.shape[-1]``.
    The logits are used to calculate the probabilities of being greater than or
    equal to each of these ``n`` values. If a target value y is not an integer,
    it is treated as having prabability mass of :math:`y- \lfloor y \rfloor` at
    :math:`\lfloor y \rfloor + 1`  and probability mass of :math:`1 + \lfloor y \rfloor - y`
    at :math:`\lfloor y \rfloor`. Then binary cross entropy loss is applied.

    More specifically, the ``logits`` passed to ``calc_loss`` represents the following:
    P = sigmoid(logits) and  P[i] means the probability that the (transformed)
    ``target`` is greater than or equal to  ``i - (n-1)//2``

    Args:
        transform: the transformation applied to target. If it is provided, the
            the regression target will be transformed.
        inverse_after_mean: when calculating the expected prediction, whether to
            do the inverse transformation after calculating the the expectation
            in the transformed space. Note that using ``inverse_after_mean=True``
            will make the expectation biased in general. This is because
            :math:`f^{-1}(E(x)) \le E(f^{-1}(x))` (Jensen inequality) if
            :math:`f^{-1}` is convex. In our case, :math:`f^{-1}` is convex for
            :math:`x \ge 0`.
    """

    def __call__(self, logits: torch.Tensor, target: torch.Tensor):
        """Caculate the loss.

        Args:
            logits: shape is [B, n]
            target: the shape is [B]
        Returns:
            loss with the same shape as target
        """
        n = logits.shape[-1]
        bin1, bin2, w2 = self._calc_bin(logits, target)
        w = F.one_hot(bin1, num_classes=n).to(logits.dtype)
        w = 1 - w.cumsum(dim=-1)
        B = _get_indexer(target.shape)
        w[B + (bin2, )] = w2
        w[B + (bin1, )] = 1
        loss = F.binary_cross_entropy_with_logits(
            logits, w, reduction='none').sum(dim=-1)
        return loss

    def calc_expectation(self, logits: torch.Tensor):
        """Calculate the expected predition in the untransfomred domain from ``pred``.

        Args:
            pred: raw model prediction
        """
        n = logits.shape[-1]
        lower_bound = -((n - 1) // 2)
        logits = logits.cummin(dim=-1).values
        probs = logits.sigmoid()
        if self._inverse_after_mean:
            pred = probs.sum(dim=-1) + (lower_bound - 1)
            if self._transform is not None:
                pred = self._transform.inverse_transform(pred)
        else:
            probs = torch.cat(
                [probs[..., :-1] - probs[..., 1:], probs[..., -1:]], dim=-1)
            support = self._calc_support(logits.shape[-1])
            pred = torch.mv(probs, support)
        return pred

    def initialize_bias(self, bias: torch.Tensor):
        """Initialize the bias of the last FC layer for the prediction properly.

        This function set the bias so that the initial distribution of the prediction
        in the original domain of target is approximatedly Cauchy: :math:`p(x) \propto \frac{1}{1+x^2}`

        Args:
            bias: the bias parameter to be initialized.
        """
        assert bias.ndim == 1
        n = bias.shape[0]
        upper_bound = n // 2
        lower_bound = -((n - 1) // 2)
        x = torch.arange(lower_bound, upper_bound + 1, dtype=torch.float32)
        x1 = x - 0.5
        x2 = x + 0.5
        if self._transform is not None:
            x = self._transform.inverse_transform(x)
            x1 = self._transform.inverse_transform(x1)
            x2 = self._transform.inverse_transform(x2)
        probs = (x2 - x1) / (x**2 + 1)
        probs = probs / probs.sum()
        probs = probs.cumsum(dim=0)
        probs = torch.cat([torch.tensor([1e-6]), probs[:-1]], dim=0)
        with torch.no_grad():
            bias.copy_(((1 - probs) / probs).log())


@alf.repr_wrapper
class QuantileRegressionLoss(ScalarPredictionLoss):
    """Multi-quantile Huber loss

    The loss for simultaneous multiple quantile regression. The number of quantiles
    n is ``quantiles.shape[-1]``. ``quantiles[..., k]`` is the quantile value
    estimation for quantile :math:`(k + 0.5) / n`. For each prediction, there
    can be one or multiple target values.

    This loss is described in the following paper:

    `Dabney et. al. Distributional Reinforcement Learning with Quantile Regression
    <https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17184/16590>`_

    Args:
        transform: the transformation applied to target. If it is provided, the
            the regression target will be transformed.
        inverse_after_mean: when calculating the expected prediction, whether to
            do the inverse transformation after calculating the the expectation
            in the transformed space. Note that using ``inverse_after_mean=True``
            will make the expectation biased in general. This is because
            :math:`f^{-1}(E(x)) \le E(f^{-1}(x))` (Jensen inequality) if
            :math:`f^{-1}` is convex. In our case, :math:`f^{-1}` is convex for
            :math:`x \ge 0`.
        delta: the smoothness parameter for huber loss (larger means smoother).
            Note that the quantile estimation with delta > 0 is biased. You should
            use a small value for ``delta`` if you want the quantile estimation
            to be less biased (so that the mean of the quantile will be close
            to mean of the samples).
    """

    def __init__(self,
                 transform: Optional[InvertibleTransform] = None,
                 inverse_after_mean: bool = False,
                 delta: float = 0.0):
        super().__init__()
        self._transform = transform
        self._delta = delta
        self._inverse_after_mean = inverse_after_mean

    def __call__(self, quantiles: torch.Tensor, target: torch.Tensor):
        """Calculate the loss.

        Args:
            quantiles: batch_shape + [num_quantiles,]
            target: batch_shape or batch_shape + [num_targets, ]
        Returns:
            loss whose shape is batch_shape
        """
        assert quantiles.shape[:-1] == target.shape
        if self._transform is not None:
            target = self._transform.transform(target)
        return multi_quantile_huber_loss(quantiles, target, delta=self._delta)

    def calc_expectation(self, quantiles: torch.Tensor):
        """Calculate the expected predition in the untransfomred domain from ``pred``.

        Args:
            quantiles: predicted quantile values in the transformed space.
        """
        if self._transform is not None:
            if self._inverse_after_mean:
                return self._transform.inverse_transform(
                    quantiles.mean(dim=-1))
            else:
                return self._transform.inverse_transform(quantiles).mean(
                    dim=-1)
        else:
            return quantiles.mean(dim=-1)
