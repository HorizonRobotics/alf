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
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from scipy.optimize import linear_sum_assignment

import alf
from alf.utils.math_ops import InvertibleTransform, binary_neg_entropy
from alf.utils import summary_utils


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


@alf.configurable
def iqn_huber_loss(value: torch.Tensor,
                   target: torch.Tensor,
                   tau_hat: Optional[torch.Tensor] = (),
                   next_delta_tau: Optional[torch.Tensor] = (),
                   fixed_tau: Optional[torch.Tensor] = (),
                   sum_over_quantiles: bool = True,
                   loss_fn: Callable = huber_function):
    """Huber loss used by the following IQN paper:

    ::
        Dabney et al "Implicit Quantile Networks for Distributional Reinforcement Learning",
        arXiv:1806.06923

    Args:
        value: the time-major tensor for the value at each time step. The loss
            is between this and the target.
        target: the time-major tensor for return, this is used as the target
            for computing the loss.
        next_delta_tau: the sampled increments of the probability for the input 
            of the quantile function of the target critics.
        fixed_tau: the fixed increments of probability, for non iqn style
            quantile regression.
        sum_over_quantiles: If True, the quantile regression loss will
            be summed along the quantile dimension. Otherwise, it will be
            averaged along the quantile dimension instead. Default is False.
        loss_fn: the loss function for the difference between value and target.
            Default is huber_function.

    Returns:
        loss: the computed loss.
        diff: the difference between target and value, for summary purpose.

    """

    # for quantile regression TD, the value and target both have shape
    # (T-1 or T, B, n_quantiles) for scalar reward and
    # (T-1 or T, B, reward_dim, n_quantiles) for multi-dim reward.
    # The quantile TD has shape
    # (T-1 or T, B, n_quantiles, n_quantiles) for scalar reward and
    # (T-1 or T, B, reward_dim, n_quantiles, n_quantiles) for multi-dim reward
    assert value.shape[0] == target.shape[0]
    if isinstance(tau_hat, torch.Tensor) and isinstance(
            next_delta_tau, torch.Tensor):
        assert tau_hat.shape[0] == next_delta_tau.shape[0] == target.shape[0]
        iqn_tau = True
    else:
        assert isinstance(fixed_tau, torch.Tensor)
        iqn_tau = False
    quantiles = value.unsqueeze(-2)
    quantiles_target = target.detach().unsqueeze(-1)
    diff = quantiles_target - quantiles

    error = loss_fn(diff)
    if iqn_tau:
        if diff.ndim - tau_hat.ndim > 1:
            # For multidimentional reward:
            # diff is of shape [T or T-1, B, reward_dim, n_quantiles, n_quantiles]
            # while tau_hat and next_delta_tau have shape [T or T-1, B, n_quantiles]
            tau_hat = tau_hat.unsqueeze(-2)
            next_delta_tau = next_delta_tau.unsqueeze(-2)
        loss = torch.abs((tau_hat.unsqueeze(-2) - (diff.detach() < 0).float()
                          )) * error * next_delta_tau.unsqueeze(-1)
    else:
        loss = torch.abs((fixed_tau - (diff.detach() < 0).float())) * error
    if sum_over_quantiles:
        loss = loss.mean(-2).sum(-1)
    else:
        loss = loss.mean(dim=(-2, -1))

    return loss, diff


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
        for doing loss specific initializations. Note that the weight of the last
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
        shape: The shape of the tensor to be accessed excluding the last dimension.
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
        # Due to limited numerical precision, w2 may be slightly out of the range
        # of [0, 1]. So we clamp it to the right range.
        w2 = w2.clamp(0, 1)
        return bin1, bin2, w2


@alf.repr_wrapper
class DiscreteRegressionLoss(_DiscreteRegressionLossBase):
    r"""A loss for predicting the distribution of a scalar.

    The target is assumed to be in the range ``[-(n-1)//2, n//2]``, where ``n=logits.shape[-1]``.
    The logits are used to calculate the probabilities of being one of the ``n``
    values. If a target value y is not an integer, it is treated as having
    probability mass of :math:`y- \lfloor y \rfloor` at :math:`\lfloor y \rfloor + 1`
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
        """Calculate the loss.

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
        neg_entropy = w1.xlogy(w1) + w2.xlogy(w2)
        return (loss + neg_entropy).relu()

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
        r"""Initialize the bias of the last FC layer for the prediction properly.

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
    it is treated as having probability mass of :math:`y- \lfloor y \rfloor` at
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
        """Calculate the loss.

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
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits, w, reduction='none')
        kld = cross_entropy + binary_neg_entropy(w)
        return kld.relu().sum(dim=-1)

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
        r"""Initialize the bias of the last FC layer for the prediction properly.

        This function set the bias so that the initial distribution of the prediction
        in the original domain of target is approximatedly Cauchy: :math:`p(x) \propto \frac{1}{1+x^2}`

        Args:
            bias: the bias parameter to be initialized.
        """
        assert bias.ndim == 1
        n = bias.shape[0]
        upper_bound = n // 2
        lower_bound = -((n - 1) // 2)
        # Use float64 to prevent precision loss due to cumsum
        x = torch.arange(lower_bound, upper_bound + 1, dtype=torch.float64)
        x1 = x - 0.5
        x2 = x + 0.5
        if self._transform is not None:
            x = self._transform.inverse_transform(x)
            x1 = self._transform.inverse_transform(x1)
            x2 = self._transform.inverse_transform(x2)
        probs = (x2 - x1) / (x**2 + 1)
        probs = probs / probs.sum()
        probs = probs.cumsum(dim=0)
        probs = torch.cat(
            [torch.tensor([1e-20], dtype=torch.float64), probs[:-1]], dim=0)
        with torch.no_grad():
            bias.copy_(((1 - probs) / probs).log().to(torch.float32))


@alf.repr_wrapper
class QuantileRegressionLoss(ScalarPredictionLoss):
    r"""Multi-quantile Huber loss

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


@alf.repr_wrapper
class AsymmetricSimSiamLoss(nn.Module):
    """The siamese loss proposed in:

    Chen Xinlei et. al. "Exploring Simple Siamese Representation Learning" CVPR 2021

    The loss is ``1-cosine(pred(proj(x), detach(proj(y)))``, where x is the predicted
    representation, y is the target representation, and pred and proj are computed
    using ``proj_net`` and ``pred_net``.

    Args:
        proj_net: if not provided, a default MLP with two hidden layers and
            output size as ``output_size`` will be created.
        pred_net: if not provided, a default MLP with one hidden layer will
            be created.
        input_size: input size of ``proj_net``
        proj_hidden_size: the size of the hidden layers of proj_net. Only useful
            if ``proj_net`` is not provided.
        pred_hidden_size: the size of the hidden layer of pred_net. Only useful
            if ``pred_net`` is not provided.
        proj_last_use_bn: whether to use batch norm for the output layer of
            proj_net. Only useful if ``proj_net`` is not provided
        eps: the ``eps`` for calling ``F.normalize()`` when calculating the
            normalized vector in order to calculate cosine.
        fixed_weight_norm: whether to fix the norm of the weight parameter of
            the FC layers.
        lr: learning rate. If None, the default learning rate will be used.
        debug_summaries: whether to write debug summaries
        name: name of this loss
    """

    def __init__(self,
                 proj_net: Optional[alf.nn.Network] = None,
                 pred_net: Optional[alf.nn.Network] = None,
                 input_size: Optional[int] = None,
                 proj_hidden_size: int = 256,
                 pred_hidden_size: int = 128,
                 output_size: int = 256,
                 proj_last_use_bn: bool = False,
                 eps: float = 1e-5,
                 fixed_weight_norm: bool = False,
                 lr: Optional[float] = None,
                 debug_summaries: bool = True,
                 name: str = "SimSiamLoss"):
        super().__init__()
        if proj_net is None:
            assert input_size is not None, "input_size must be provided if proj_net is not given"
            proj_net = alf.nn.Sequential(
                alf.layers.Reshape(-1),
                alf.layers.FC(
                    input_size,
                    proj_hidden_size,
                    activation=torch.relu_,
                    use_bn=True,
                    weight_opt_args=dict(fixed_norm=fixed_weight_norm, lr=lr)),
                alf.layers.FC(
                    proj_hidden_size,
                    proj_hidden_size,
                    activation=torch.relu_,
                    use_bn=True,
                    weight_opt_args=dict(fixed_norm=fixed_weight_norm, lr=lr)),
                alf.layers.FC(
                    proj_hidden_size,
                    output_size,
                    use_bn=proj_last_use_bn,
                    weight_opt_args=dict(
                        lr=lr,
                        fixed_norm=fixed_weight_norm and proj_last_use_bn)),
                input_tensor_spec=alf.TensorSpec((input_size, )))
        output_size = proj_net.output_spec.numel
        if pred_net is None:
            pred_net = alf.nn.Sequential(
                alf.layers.FC(
                    output_size,
                    pred_hidden_size,
                    activation=torch.relu_,
                    use_bn=True,
                    weight_opt_args=dict(lr=lr, fixed_norm=fixed_weight_norm)),
                alf.layers.FC(
                    pred_hidden_size,
                    output_size,
                    weight_opt_args=dict(lr=lr, fixed_norm=False)))
        self._proj_net = proj_net
        self._pred_net = pred_net
        self._eps = eps
        self._debug_summaries = debug_summaries
        self._name = name

    @alf.summary.enter_summary_scope
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Calculate the loss.

        Args:
            pred: predicted representation of shape [B, T, ...]
            target: target representation of shape [B, T, ...]
        Returns:
            loss of shape [B, T]
        """
        assert pred.shape == target.shape
        B, T = pred.shape[:2]
        target = target.reshape(B * T, *target.shape[2:])
        pred = pred.reshape(B * T, *pred.shape[2:])
        if self._debug_summaries and alf.summary.should_record_summaries():
            pred = summary_utils.summarize_tensor_gradients(
                "pred_grad", pred, clone=True)
        with torch.no_grad():
            projected_target = self._proj_net(target.to(pred.dtype))[0]
            norm_projected_target = F.normalize(
                projected_target.detach(), dim=1, eps=self._eps)
        projected_pred = self._proj_net(pred)[0]
        predicted_projected_pred = self._pred_net(projected_pred)[0]
        norm_predicted_projected_pred = F.normalize(
            predicted_projected_pred, dim=1, eps=self._eps)
        cos = (norm_projected_target * norm_predicted_projected_pred).sum(
            dim=1)
        if self._debug_summaries and alf.summary.should_record_summaries():
            summary_utils.add_mean_hist_summary("cos", cos)
            summary_utils.add_mean_hist_summary(
                "predicted_projected_pred_norm",
                predicted_projected_pred.norm(dim=1))
        return (1 - cos).reshape(B, T)


@alf.repr_wrapper
class MeanSquaredLoss(object):
    """Mean squared loss.

    For a prediction and target pair (x,y), the loss is ``((x - y) ** 2).mean()``.

    Args:
        batch_dims: the first so many dims of prediction and target are treated
            as batch dimension. The mean is performed on the rest of the dimensions.
    """

    def __init__(self,
                 batch_dims: int = 1,
                 debug_summaries: bool = True,
                 name: str = "MSELoss"):
        super().__init__()
        self._debug_summaries = debug_summaries
        self._name = name
        self._batch_dims = batch_dims

    @alf.summary.enter_summary_scope
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Calculate the loss.

        Args:
            pred: prediction of shape [B, ...]
            target: target of shape [B, ...]
        Returns:
            loss of shape [B]
        """
        assert pred.shape == target.shape
        if self._debug_summaries and alf.summary.should_record_summaries():
            pred = summary_utils.summarize_tensor_gradients(
                "pred_grad", pred, clone=True)
        ndim = pred.ndim
        assert ndim >= self._batch_dims
        loss = (pred - target)**2
        if ndim > self._batch_dims:
            loss = loss.mean(dim=list(range(self._batch_dims, ndim)))
        return loss


class BipartiteMatchingLoss(object):
    r"""Bipartite matching loss.

    This order-invariant loss can be used to evaluate the matching between a predicted
    set and a target set. The idea is that for every forward, an optimal one-to-one
    mapping assignment from the predicted set to the target set is first found using
    some efficient bipartite graph matching algorithm, and the optimal loss is
    minimized.

    Mathematically, suppose there are :math:`N` objects in either set,
    :math:`L(x,y)` is the matching loss between any :math:`(x,y)` object pair,
    and :math:`\mathcal{G}_N` is the permutation space. The forward loss to be
    minimized is:

    .. math::

        \min_{g\in\mathcal{G}_N}\sum_n^N L(x_n(\theta),y_{g(n)})

    where :math:`\theta` is the model parameters.

    In practice, to find the optimal assignment, we simply use ``scipy.optimize.linear_sum_assignment``.

    References::
        `End-to-End Object Detection with Transformers <https://arxiv.org/pdf/2005.12872.pdf>`_, Carion et al.

        `<https://github.com/facebookresearch/detr/blob/main/models/matcher.py>`_
    """

    def __init__(self,
                 reduction: str = 'mean',
                 name: str = "BipartiteMatchingLoss"):
        """
        Args:
            reduction: 'sum', 'mean' or 'none'. This is how to reduce the matching
                loss. For the former two, the loss shape is ``[B]``, while for
                the 'none', the loss shape is ``[B,N]``.
        """
        super().__init__()
        self._reduction = reduction
        assert reduction in ['mean', 'sum', 'none']
        self._name = name

    def forward(self,
                matching_cost_mat: torch.Tensor,
                cost_mat: torch.Tensor = None):
        """Compute the optimal matching loss.

        Args:
            matching_cost_mat: the cost matrix used to determine the optimal
                matching. It shape should be ``[B,N,N]``.
            cost_mat: the cost matrix used to compute the optimal loss once the
                optimal matching is found. According to the DETR paper, this
                cost matrix might be different from the one used for matching.
                If None, then it will be the same matrix for matching.

        Returns:
            tuple:
            - the optimal loss. If reduction is 'mean' or 'sum', its shape is
              ``[B]``, otherwise its shape is ``[B,N]``.
            - the optimal matching given the cost matrix. Its shape is ``[B,N]``,
              where the value of n-th entry is its mapped index in the target set.
        """
        if cost_mat is None:
            cost_mat = matching_cost_mat

        with torch.no_grad():
            B, N = matching_cost_mat.shape[:2]
            max_cost = matching_cost_mat.max() + 1.
            # [B*N, B*N]
            # Subtract all diag entries by a max cost so that no off-diag matchings
            # will be optimal.
            big_cost_mat = torch.block_diag(
                *list(matching_cost_mat - max_cost))
            np_big_cost_mat = big_cost_mat.cpu().numpy()
            # col_ind: [B*N]
            row_ind, col_ind = linear_sum_assignment(np_big_cost_mat)
            col_ind = col_ind % N
            col_ind = col_ind.reshape(B, N, 1)
            col_ind = torch.tensor(col_ind).to(cost_mat.device)

        # [B,N]
        optimal_loss = cost_mat.gather(dim=-1, index=col_ind).squeeze(-1)
        if self._reduction == 'mean':
            optimal_loss = optimal_loss.mean(-1)
        elif self._reduction == 'sum':
            optimal_loss = optimal_loss.sum(-1)
        return optimal_loss, col_ind.squeeze(-1)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
