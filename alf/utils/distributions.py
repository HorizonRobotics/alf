# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import math
import torch
from torch import Tensor

import torch.distributions as td


class InverseTransformSampling(object):
    """Interface for defining inverse transform sampling."""

    @staticmethod
    def cdf(x):
        """Cumulative distribution function of this distribution."""
        raise NotImplementedError

    @staticmethod
    def icdf(x):
        """Inverse of the CDF"""
        raise NotImplementedError

    @staticmethod
    def log_prob(x):
        """Log probability density."""
        raise NotImplementedError


class NormalITS(InverseTransformSampling):
    """Normal distribution.

    .. math::

        p(x) = 1/sqrt(2*pi) * exp(-x^2/2)

    """

    @staticmethod
    def cdf(x):
        # sqrt(0.5) = 0.7071067811865476
        return 0.5 * (1 + torch.erf(x * 0.7071067811865476))

    @staticmethod
    def icdf(x):
        # sqrt(2) = 1.4142135623730951
        return torch.erfinv(2 * x - 1) * 1.4142135623730951

    @staticmethod
    def log_prob(x):
        # log(sqrt(2 * pi)) = 0.9189385332046727
        return -0.5 * (x**2) - 0.9189385332046727


class CauchyITS(InverseTransformSampling):
    """Cauchy distribution.

    .. math::

        p(x) = 1 / (pi * (1 + x*x))

    """

    @staticmethod
    def cdf(x):
        return torch.atan(x) / math.pi + 0.5

    @staticmethod
    def icdf(x):
        return torch.tan(math.pi * (x - 0.5))

    @staticmethod
    def log_prob(x):
        return -(math.pi * (1 + x**2)).log()


class T2Cdf_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # return 0.5 + 0.5 * x / (1 + x**2).sqrt()
        # The original form is not numerically increasing (i.e. for some large
        # value of x2>x1, f(x2) < f(x1) because of insufficient numerical
        # precision), so we use the following form. And we need to implement our
        # own backward because autograd cannot calculate the gradient of the
        # following form when x is 0.
        y = 0.5 + 0.5 * x.sign() * (1 + x**-2).rsqrt()
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return 0.5 * (1 + x**2)**(-1.5) * grad_output


t2cdf = T2Cdf_.apply


class T2ITS(InverseTransformSampling):
    """Student's t-distribution with DOF 2.

    .. math::

        p(x) = 1 / (2 * (1 + x*x) ** 1.5)

    """

    @staticmethod
    def cdf(x):
        return t2cdf(x)

    @staticmethod
    def icdf(x):
        return (x - 0.5) * (x * (1 - x)).rsqrt()

    @staticmethod
    def log_prob(x):
        # log(2) = 0.6931471805599453
        return -1.5 * (1 + x**2).log() - 0.6931471805599453


class TruncatedDistribution(td.Distribution):
    r"""The base class of truncated distributions.

    A truncated distribution :math:`q(x)` is defined as a standard base distribution :math:`p(x)` and
    location :math:`\mu`, scale parameters :math:`s`, lower bound :math:`l` and
    upper bound :math:`u`

    .. math::

        q(x) = \frac{1}{s (P(u) - P(l))}p(\frac{x-\mu}{s}) if l \le x le u
        q(x) = 0 otherwise

    where :math:`P` is the cdf of :math:`p`.

    Args:
        loc: the location parameter. Its shape is batch_shape + event_shape.
        scale: the scale parameter. Its shape is batch_shape + event_shape.
        lower_bound: the lower bound. Its shape is event_shape.
        upper_bound: the upper bound. Its shape is event_shape.
        its: the standard distribution to be used.
    """

    arg_constraints = {
        'loc': td.constraints.real,
        'scale': td.constraints.positive
    }
    has_rsample = True

    def __init__(self, loc: Tensor, scale: Tensor, lower_bound: Tensor,
                 upper_bound: Tensor, its: InverseTransformSampling):
        event_shape = torch.broadcast_shapes(lower_bound.shape,
                                             upper_bound.shape)
        batch_shape = torch.broadcast_shapes(scale.shape, loc.shape,
                                             event_shape)
        if len(event_shape) > 0:
            batch_shape = batch_shape[:-len(event_shape)]

        self._scale = scale
        self._loc = loc

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)
        self._its = its
        self._lower_bound = lower_bound.to(loc.device)
        self._upper_bound = upper_bound.to(loc.device)
        self._cdf_lb = self._its.cdf((self._lower_bound - loc) / scale)
        self._cdf_ub = self._its.cdf((self._upper_bound - loc) / scale)
        self._logz = (scale * (self._cdf_ub - self._cdf_lb + 1e-30)).log()

    @property
    def scale(self):
        """Scale parameter of this distribution."""
        return self._scale

    @property
    def loc(self):
        """Location parameter of this distribution."""
        return self._loc

    @property
    def lower_bound(self):
        """Lower bound of this distribution."""
        return self._lower_bound

    @property
    def upper_bound(self):
        """Upper bound of this distribution."""
        return self._upper_bound

    @property
    def mode(self):
        """Mode of this distribution."""
        result = torch.maximum(self._lower_bound, self._loc)
        result = torch.minimum(self._upper_bound, result)

        return result

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        Args:
            sample_shape: sample shape
        Returns:
            Tensor of shape ``sample_shape + batch_shape + event_shape``
        """
        r = torch.rand(sample_shape + self._batch_shape + self._event_shape)
        r = (1 - r) * self._cdf_lb + r * self._cdf_ub
        r = r.clamp(0.001, 0.999)
        x = self._its.icdf(r) * self._scale + self._loc
        assert torch.isfinite(x).all()
        # because of the r.clamp() above, x may be out of bound
        x = torch.maximum(x, self._lower_bound)
        x = torch.minimum(x, self._upper_bound)
        return x

    def log_prob(self, value: Tensor):
        """The log of the probability density evaluated at ``value``.

        Args:
            value: its shape should be ``sample_shape + batch_shape + event_shape``
        Returns:
            Tensor of shape ``sample_shape + batch_shape``
        """
        y = self._its.log_prob((value - self._loc) / self._scale) - self._logz
        assert torch.isfinite(y).all()
        n = len(self._event_shape)
        if n > 0:
            return y.sum(dim=list(range(-n, 0)))
        else:
            return y


class TruncatedNormal(TruncatedDistribution):
    r"""Truncated normal distribution.

    The truncated normal distribution :math:`q(x)` is defined by 4 parameters:
    location :math:`\mu`, scale parameters :math:`s`, lower bound :math:`l` and
    upper bound :math:`u`.

    .. math::

        q(x) = \frac{1}{s (P(u) - P(l))}p(\frac{x-\mu}{s})

    where :math:`p` and :math:`P` are the pdf and cdf of the standard normal
    distribution respectively.

    Args:
        loc: the location parameter
        scale: the scale parameter
        lower_bound: the lower bound
        upper_bound: the upper bound
        its: the standard distribution to be used.
    """

    def __init__(self, loc, scale, lower_bound, upper_bound):
        super().__init__(loc, scale, lower_bound, upper_bound, NormalITS())


@td.kl.register_kl(TruncatedNormal, TruncatedNormal)
def _kl_truncated_normal_trucated_normal(p, q):
    """Registered KL Divergence computation specialized for TruncatedNormal

    It is closed form w.r.t. torch.erf.

    """
    assert torch.all(
        torch.logical_and(
            torch.isclose(p.lower_bound, q.lower_bound),
            torch.isclose(p.upper_bound, q.upper_bound)))

    delta = p.loc - q.loc
    delta2 = delta**2

    sigma_p2 = p.scale**2
    # Pad sigma_q2 as it is positive will only be served as denominator
    sigma_q2 = q.scale**2 + 1e-30

    c1 = 0.5 * (torch.log(q.scale) - torch.log(p.scale)) + 0.25 * (
        delta2 + sigma_p2) / sigma_q2 - 0.25

    # 1 / sqrt(2 pi) = 0.3989422804014327
    c2 = -0.3989422804014327 * p.scale * delta / sigma_q2

    # 0.5 / sqrt(pi) = 0.28209479177387814
    c3 = (1.0 - sigma_p2 / sigma_q2) * 0.28209479177387814

    # sqrt(0.5) = 0.7071067811865475
    t_u = (p.upper_bound - p.loc) * 0.7071067811865475 / (p.scale + 1e-30)
    t_l = (p.lower_bound - p.loc) * 0.7071067811865475 / (p.scale + 1e-30)

    upper = c1 * torch.erf(t_u) + (c3 * t_u + c2) * torch.exp(-t_u * t_u)
    lower = c1 * torch.erf(t_l) + (c3 * t_l + c2) * torch.exp(-t_l * t_l)

    # At this moment, before_normalization holds the integral part of
    # original gaussian (but both p and q are not normalized by area_p
    # and area_q respectively). This will be handled below.
    before_normalization = upper - lower

    # Pad area_p as it is positive will only be served as denominator
    area_p = p._cdf_ub - p._cdf_lb + 1e-30
    area_q = q._cdf_ub - q._cdf_lb

    return (torch.log(area_q / area_p) + before_normalization / area_p).sum(
        dim=list(range(-len(p._event_shape), 0)))


class TruncatedCauchy(TruncatedDistribution):
    r"""Truncated Cauchy distribution.

    The truncated normal distribution :math:`q(x)` is defined by 4 parameters:
    location :math:`\mu`, scale parameters :math:`s`, lower bound :math:`l` and
    upper bound :math:`u`.

    .. math::

        q(x) = \frac{1}{s (P(u) - P(l))}p(\frac{x-\mu}{s})

    where :math:`p` and :math:`P` are the pdf and cdf of the standard Cauchy
    distribution respectively.

    Args:
        loc: the location parameter
        scale: the scale parameter
        lower_bound: the lower bound
        upper_bound: the upper bound
        its: the standard distribution to be used.
    """

    def __init__(self, loc, scale, lower_bound, upper_bound):
        super().__init__(loc, scale, lower_bound, upper_bound, CauchyITS())


class TruncatedT2(TruncatedDistribution):
    r"""Truncated Student's t distribution with degree of freedom 2.

    The truncated normal distribution :math:`q(x)` is defined by 4 parameters:
    location :math:`\mu`, scale parameters :math:`s`, lower bound :math:`l` and
    upper bound :math:`u`.

    .. math::

        q(x) = \frac{1}{s (P(u) - P(l))}p(\frac{x-\mu}{s})

    where :math:`p(x)=1 / (2 * (1 + x^2)^1.5)` and :math:`P` is the cdf of
    :math:`p(x)`.

    Args:
        loc: the location parameter
        scale: the scale parameter
        lower_bound: the lower bound
        upper_bound: the upper bound
        its: the standard distribution to be used.
    """

    def __init__(self, loc, scale, lower_bound, upper_bound):
        super().__init__(loc, scale, lower_bound, upper_bound, T2ITS())
