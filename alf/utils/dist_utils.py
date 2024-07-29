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

import functools
import numbers
import numpy as np
import math
import torch
import torch.distributions as td
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
import torch.nn as nn
from typing import Union

import alf
import alf.nest as nest
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from .distributions import TruncatedDistribution, TruncatedNormal, TruncatedCauchy, TruncatedT2


def get_invertible(cls):
    """A helper function to turn on the cache mechanism for transformation.
    This is useful as some transformations (say :math:`g`) may not be able to
    provide an accurate inversion therefore the difference between :math:`x` and
    :math:`g^{-1}(g(x))` is large. This could lead to unstable training in
    practice. For a torch transformation :math:`y=g(x)`, when ``cache_size`` is
    set to one, the latest value for :math:`(x, y)` is cached and will be used
    later for future computations. E.g. for inversion, a call to
    :math:`g^{-1}(y)` will return :math:`x`, solving the inversion error issue
    mentioned above. Note that in the case of having a chain of transformations
    (:math:`G`), all the element transformations need to turn on the cache to
    ensure the composite transformation :math:`G` satisfy:
    :math:`x=G^{-1}(G(x))`.
    """

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, cache_size=1)

    return NewCls


"""
WARNING: If you need to train policy gradient with a ``TransformedDistribution``,
then make sure to detach the sampled action when the transforms have trainable
parameters.

For detailed reasons, please refer to ``alf/docs/notes/compute_probs_of_transformed_dist.rst``.
"""

AbsTransform = get_invertible(td.AbsTransform)
ExpTransform = get_invertible(td.ExpTransform)
PowerTransform = get_invertible(td.PowerTransform)
SigmoidTransform = get_invertible(td.SigmoidTransform)
SoftmaxTransform = get_invertible(td.SoftmaxTransform)


class AffineTransform(get_invertible(td.AffineTransform)):
    """Overwrite PyTorch's ``AffineTransform`` to provide a builder to be
    compatible with ``DistributionSpec.build_distribution()``.
    """

    def get_builder(self):
        return functools.partial(
            AffineTransform, loc=self.loc, scale=self.scale)


@alf.configurable
class Softplus(td.Transform):
    r"""Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.

    Code adapted from `pyro <https://docs.pyro.ai/en/latest/_modules/pyro/distributions/transforms/softplus.html>`_
    and `tensorflow <https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/bijectors/softplus.py#L61-L189>`_.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __init__(self, hinge_softness=1., cache_size=1):
        """
        Args:
            hinge_softness (float): this positive parameter changes the transition
                slope. A higher softness results in a smoother transition from
                0 to identity.
        """
        super().__init__(cache_size=cache_size)
        self._hinge_softness = float(hinge_softness)
        assert self._hinge_softness > 0, "Must be a positive softness number!"

    def __eq__(self, other):
        return (isinstance(other, Softplus)
                and self._hinge_softness == other._hinge_softness)

    def _call(self, x):
        return nn.functional.softplus(x, beta=1. / self._hinge_softness)

    def _inverse(self, y):
        return (y / self._hinge_softness).expm1().log() * self._hinge_softness

    def log_abs_det_jacobian(self, x, y):
        return -nn.functional.softplus(-x / self._hinge_softness)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return Softplus(self._hinge_softness, cache_size)


@alf.configurable
def Softlower(low, hinge_softness=1.):
    """Create a Softlower transform by composing the Softplus and Affine
    transforms. Mathematically, ``softlower(x, low) = softplus(x - low) + low``.

    Args:
        low (float|Tensor): the lower bound
        hinge_softness (float): this positive parameter changes the transition
                slope. A higher softness results in a smoother transition from
                ``low`` to identity.
    """
    return td.transforms.ComposeTransform([
        AffineTransform(loc=-low, scale=1.),
        Softplus(hinge_softness=hinge_softness),
        AffineTransform(loc=low, scale=1.)
    ])


@alf.configurable
def Softupper(high, hinge_softness=1.):
    """Create a Softupper transform by composing the Softplus and Affine
    transforms. Mathematically, ``softupper(x, high) = -softplus(high - x) + high``.

    Args:
        high (float|Tensor): the upper bound
        hinge_softness (float): this positive parameter changes the transition
                slope. A higher softness results in a smoother transition from
                identity to ``high``.
    """
    return td.transforms.ComposeTransform([
        AffineTransform(loc=high, scale=-1.),
        Softplus(hinge_softness=hinge_softness),
        AffineTransform(loc=high, scale=-1.)
    ])


@alf.configurable
def SoftclipTF(low, high, hinge_softness=1.):
    """Create a Softclip transform by composing Softlower, Softupper, and Affine
    transforms, adapted from `tensorflow <https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/SoftClip>`_.
    Mathematically,

    .. code-block:: python

        clipped = softupper(softlower(x, low), high)
        softclip(x) = (clipped - high) / (high - softupper(low, high)) * (high - low) + high

    The second scaling step is because we will have
    ``softupper(low, high) < low`` due to distortion of softplus, so we need to
    shrink the interval slightly by ``(high - low) / (high - softupper(low, high))``
    to preserve the lower bound. Due to this rescaling, the bijector can be mildly
    asymmetric.

    Args:
        low (float|Tensor): the lower bound
        high (float|Tensor): the upper bound
        hinge_softness (float): this positive parameter changes the transition
                slope. A higher softness results in a smoother transition from
                ``low`` to ``high``.
    """
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(low)
    assert torch.all(high > low), "Invalid clipping range"

    # Compute the clipped value of ``low`` upper bounded by ``high``
    softupper_high_at_low = Softupper(high, hinge_softness=hinge_softness)(low)
    return td.transforms.ComposeTransform([
        Softlower(low=low, hinge_softness=hinge_softness),
        Softupper(high=high, hinge_softness=hinge_softness),  # clipped
        AffineTransform(loc=-high, scale=1.),
        AffineTransform(
            loc=high, scale=(high - low) / (high - softupper_high_at_low))
    ])


@alf.configurable
class Softclip(td.Transform):
    r"""Transform via the mapping defined in ``alf.math_ops.softclip()``.
    Unlike ``SoftclipTF``, this transform is symmetric regarding the lower and
    upper bound when squashing.
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, low, high, hinge_softness=1., cache_size=1):
        """
        Args:
            low (float): the lower bound
            high (float): the upper bound
            hinge_softness (float): this positive parameter changes the transition
                slope. A higher softness results in a smoother transition from
                ``low`` to ``high``.
        """
        super().__init__(cache_size=cache_size)
        self._hinge_softness = float(hinge_softness)
        assert self._hinge_softness > 0, "Must be a positive softness number!"
        self._l = float(low)
        self._h = float(high)
        self.codomain = constraints.interval(self._l, self._h)

    def __eq__(self, other):
        return (isinstance(other, Softclip)
                and self._hinge_softness == other._hinge_softness
                and self._l == other._l and self._h == other._h)

    def get_builder(self):
        return functools.partial(Softclip, low=self._l, high=self._h)

    def _call(self, x):
        return alf.math.softclip(x, self._l, self._h, self._hinge_softness)

    def _inverse(self, y):
        """``y`` should be in ``[self._l, self._h]``. Note that when ``y`` is
        close to boundaries, this inverse function might have numerical issues.
        Since we use ``cache_size=1`` in the init function, here we don't clip
        ``y``.
        """
        s = self._hinge_softness
        return (y + s * (((self._l - y) / s).expm1() / (
            (y - self._h) / s).expm1()).log())

    def log_abs_det_jacobian(self, x, y):
        r"""Compute ``log|dy/dx|``.
        """
        s = self._hinge_softness
        return (1 - 1 / (1 + ((x - self._l) / s).exp()) - 1 / (1 + (
            (self._h - x) / s).exp())).log()

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return Softclip(self._l, self._h, self._hinge_softness, cache_size)


@alf.configurable
class Softsign(td.Transform):
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return isinstance(other, Softsign)

    def _call(self, x):
        return alf.math.softsign(x)

    def _inverse(self, y):
        r"""
        .. math::

            \begin{array}{lll}
                y = \frac{x}{1+x} \rightarrow x = \frac{y}{1 - y}, &\text{if}  &y > 0\\
                y = \frac{x}{1-x} \rightarrow x = \frac{y}{1 + y}, &\text{else}&\\
            \end{array}
        """
        return torch.where(y > 0, y / (1 - y), y / (1 + y))

    def log_abs_det_jacobian(self, x, y):
        r"""
        .. math::

            \begin{array}{lll}
                y = \frac{x}{1+x} \rightarrow \frac{dy}{dx} = \frac{1}{(1+x)^2}, &\text{if}  &x > 0\\
                y = \frac{x}{1-x} \rightarrow \frac{dy}{dx} = \frac{1}{(1-x)^2}, &\text{else}&\\
            \end{array}
        """
        return -2. * torch.log(1 + x.abs())

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return Softsign(cache_size)


@alf.configurable
class StableTanh(td.Transform):
    r"""Invertible transformation (bijector) that computes :math:`Y = tanh(X)`,
    therefore :math:`Y \in (-1, 1)`.

    This can be achieved by an affine transform of the Sigmoid transformation,
    i.e., it is equivalent to applying a list of transformations sequentially:

    .. code-block:: python

        transforms = [AffineTransform(loc=0, scale=2)
                      SigmoidTransform(),
                      AffineTransform(
                            loc=-1,
                            scale=2]

    However, using the ``StableTanh`` transformation directly is more numerically
    stable.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        # We use cache by default as it is numerically unstable for inversion
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return isinstance(other, StableTanh)

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        # Based on https://github.com/tensorflow/agents/commit/dfb8c85a01d65832b05315928c010336df13f7b9#diff-a572e559b953f965c5c2cd1b9ded2c7b

        # 0.99999997 is the maximum value such that atanh(x) is valid for both
        # float32 and float64
        def _atanh(x):
            return 0.5 * torch.log((1 + x) / (1 - x))

        y = torch.where(
            torch.abs(y) <= 1.0, torch.clamp(y, -0.99999997, 0.99999997), y)
        return _atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (
            torch.log(torch.tensor(2.0, dtype=x.dtype, requires_grad=False)) -
            x - nn.functional.softplus(-2.0 * x))

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return StableTanh(cache_size)


class DiagMultivariateNormal(td.Independent):
    def __init__(self, loc, scale):
        """Create multivariate normal distribution with diagonal variance.

        Args:
            loc (Tensor): mean of the distribution
            scale (Tensor): standard deviation. Should have same shape as ``loc``.
        """
        # set validate_args to False here to enable the construction of Normal
        # distribution with zero scale.
        super().__init__(
            td.Normal(loc, scale, validate_args=False),
            reinterpreted_batch_ndims=1)

    @property
    def stddev(self):
        return self.base_dist.stddev


@alf.configurable(whitelist=['eps'])
class Beta(td.Beta):
    r"""Beta distribution parameterized by ``concentration1`` and ``concentration0``.

    Note: we need to wrap ``td.Beta`` so that ``self.concentration1`` and
    ``self.concentration0`` are the actual tensors passed in to construct the
    distribution. This is important in certain situation. For example, if you want
    to register a hook to process the gradient to ``concentration1`` and ``concentration0``,
    ``td.Beta.concentration0.register_hook()`` will not work because gradient will
    not be backpropped to ``td.Beta.concentration0`` since it is sliced from
    ``td.Dirichlet.concentration`` and gradient will only be backpropped to
    ``td.Dirichlet.concentration`` instead of ``td.Beta.concentration0`` or
    ``td.Beta.concentration1``.

    """

    def __init__(self,
                 concentration1,
                 concentration0,
                 eps=None,
                 validate_args=None):
        """
        Args:
            concentration1 (float or Tensor): 1st concentration parameter of the distribution
                (often referred to as alpha)
            concentration0 (float or Tensor): 2nd concentration parameter of the distribution
                (often referred to as beta)
            eps (float): a very small value indicating the interval ``[eps, 1-eps]``
                into which the sampled values will be clipped. This clipping can
                prevent ``NaN`` and ``Inf`` values in the gradients. If None,
                a small value defined by PyTorch will be used.
        """
        self._concentration1 = concentration1
        self._concentration0 = concentration0
        super().__init__(concentration1, concentration0, validate_args)
        if eps is None:
            self._eps = torch.finfo(self._dirichlet.concentration.dtype).eps
        else:
            self._eps = float(eps)

    @property
    def concentration0(self):
        return self._concentration0

    @property
    def concentration1(self):
        return self._concentration1

    @property
    def mode(self):
        alpha = self.concentration1
        beta = self.concentration0
        mode = torch.where((alpha > 1) & (beta > 1),
                           (alpha - 1) / (alpha + beta - 2),
                           torch.where(alpha < beta, torch.zeros(()),
                                       torch.ones(())))
        return mode

    def rsample(self, sample_shape=()):
        """We override the original ``rsample()`` in order to clamp the output
        to avoid `NaN` and `Inf` values in the gradients. See Pyro's
        ``rsample()`` implementation in
        `<https://docs.pyro.ai/en/dev/_modules/pyro/distributions/affine_beta.html#AffineBeta>`_.
        """
        x = super(Beta, self).rsample(sample_shape)
        return torch.clamp(x, min=self._eps, max=1 - self._eps)


class DiagMultivariateBeta(td.Independent):
    def __init__(self, concentration1, concentration0):
        """Create multivariate independent beta distribution.

        Args:
            concentration1 (float or Tensor): 1st concentration parameter of the
                distribution (often referred to as alpha)
            concentration0 (float or Tensor): 2nd concentration parameter of the
                distribution (often referred to as beta)
        """
        super().__init__(
            Beta(concentration1, concentration0), reinterpreted_batch_ndims=1)


class AffineTransformedDistribution(td.TransformedDistribution):
    r"""Transform via the pointwise affine mapping :math:`y = \text{loc} + \text{scale} \times x`.

    The reason of not using ``td.TransformedDistribution`` is that we can implement
    ``entropy``, ``mean``, ``variance`` and ``stddev`` for ``AffineTransforma``.
    """

    def __init__(self, base_dist: td.Distribution, loc, scale):
        """
        Args:
            loc (Tensor or float): Location parameter.
            scale (Tensor or float): Scale parameter.
        """
        super().__init__(
            base_distribution=base_dist,
            transforms=AffineTransform(loc, scale))
        self.loc = loc
        self.scale = scale

        # broadcast scale to event_shape if necessary
        s = torch.ones(base_dist.event_shape) * scale
        self._log_abs_scale = s.abs().log().sum()

    def entropy(self):
        """Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        return self._log_abs_scale + self.base_dist.entropy()

    @property
    def mean(self):
        """Returns the mean of the distribution."""
        return self.scale * self.base_dist.mean + self.loc

    @property
    def variance(self):
        """Returns the variance of the distribution."""
        return self.scale * self.scale * self.base_dist.variance

    @property
    def stddev(self):
        """Returns the variance of the distribution."""
        return self.scale * self.base_dist.stddev


class StableCauchy(td.Cauchy):
    def rsample(self, sample_shape=torch.Size(), clipping_value=0.49):
        r"""Overwrite Pytorch's Cauchy rsample for a more stable result. Basically
        the sampled number is clipped to fall within a reasonable range.


        For reference::

            > np.tan(math.pi * -0.499)
            -318.30883898554157
            > np.tan(math.pi * -0.49)
            -31.820515953773853

        Args:
            clipping_value (float): suppose eps is sampled from ``(-0.5,0.5)``.
                It will be clipped to ``[-clipping_value, clipping_value]`` to
                avoid values with huge magnitudes.
        """
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).uniform_()
        eps = torch.clamp(eps - 0.5, min=-clipping_value, max=clipping_value)
        return torch.tan(eps * math.pi) * self.scale + self.loc


class DiagMultivariateCauchy(td.Independent):
    def __init__(self, loc, scale):
        """Create multivariate cauchy distribution with diagonal scale matrix.

        Args:
            loc (Tensor): median of the distribution. Note that Cauchy doesn't
                have a mean (divergent).
            scale (Tensor): also known as "half width". Should have the same
                shape as ``loc``.
        """
        super().__init__(StableCauchy(loc, scale), reinterpreted_batch_ndims=1)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale


class OneHotCategoricalStraightThrough(td.OneHotCategoricalStraightThrough):
    """Provide an additional property ``mode`` with gradient enabled.
    """

    @property
    def mode(self):
        mode = torch.nn.functional.one_hot(
            torch.argmax(self.logits, -1), num_classes=self.logits.shape[-1])
        return mode.to(self.logits) + self.probs - self.probs.detach()


@alf.configurable
class OneHotCategoricalGumbelSoftmax(td.OneHotCategorical):
    r"""Create a reparameterizable ``td.OneHotCategorical`` distribution based on
    the Gumbel-softmax gradient estimator from

    ::

        Jang et al., "CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX", 2017.
    """
    has_rsample = True

    def __init__(self, hard_sample: bool = True, tau: float = 1., **kwargs):
        """
        Args:
            hard_sample: If False, the rsampled result will be a "soft" vector
                of Gumbel softmax distribution, which naturally supports gradient
                backprop. If True, ``argmax`` will be applied on top of it and then
                a straight-through gradient estimator is used.
            tau: the Gumbel-softmax temperature for ``rsample``. A higher
                temperature leads to a more uniform sample.
        """
        super(OneHotCategoricalGumbelSoftmax, self).__init__(**kwargs)
        self._hard_sample = hard_sample
        self._tau = tau

    def rsample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        # expand additional first dims according to ``sample_shape``
        shape = sample_shape + (1, ) * len(self.param_shape)
        logits = self.logits.repeat(*shape)
        return torch.nn.functional.gumbel_softmax(
            logits=logits, tau=self._tau, hard=self._hard_sample, dim=-1)

    @property
    def mode(self):
        mode = torch.nn.functional.one_hot(
            torch.argmax(self.logits, -1), num_classes=self.logits.shape[-1])
        return mode.to(self.logits) + self.probs - self.probs.detach()


def _builder_independent(base_builder, reinterpreted_batch_ndims_, **kwargs):
    return td.Independent(base_builder(**kwargs), reinterpreted_batch_ndims_)


def _builder_transformed(base_builder, transform_builders, params_,
                         transforms_params_):
    transforms = [
        b(**p) for b, p in zip(transform_builders, transforms_params_)
    ]
    return td.TransformedDistribution(base_builder(**params_), transforms)


def _get_categorical_builder(obj: Union[
        td.Categorical, td.OneHotCategorical, td.
        OneHotCategoricalStraightThrough, OneHotCategoricalStraightThrough]):

    dist_cls = type(obj)

    if 'probs' in obj.__dict__ and id(obj.probs) == id(obj._param):
        # This means that obj is constructed using probs
        return dist_cls, {'probs': obj.probs}
    else:
        return dist_cls, {'logits': obj.logits}


def _get_gumbelsoftmax_categorical_builder(
        obj: OneHotCategoricalGumbelSoftmax):
    builder = functools.partial(
        OneHotCategoricalGumbelSoftmax,
        hard_sample=obj._hard_sample,
        tau=obj._tau)
    if 'probs' in obj.__dict__ and id(obj.probs) == id(obj._param):
        # This means that obj is constructed using probs
        return builder, {'probs': obj.probs}
    else:
        return builder, {'logits': obj.logits}


def _get_independent_builder(obj: td.Independent):
    builder, params = _get_builder(obj.base_dist)
    new_builder = functools.partial(_builder_independent, builder,
                                    obj.reinterpreted_batch_ndims)
    return new_builder, params


def _get_transform_builders_params(transforms):
    """Return a nested structure where each node is a non-composed transform,
    after expanding any composed transform in ``transforms``.
    """

    def _get_transform_builder(transform):
        if hasattr(transform, "get_builder"):
            return transform.get_builder()
        return transform.__class__

    def _get_transform_params(transform):
        if hasattr(transform, 'params') and transform.params is not None:
            # We assume that if a td.Transform has attribute 'params', then they are the
            # parameters we'll extract and store.
            assert isinstance(
                transform.params,
                dict), ("Transform params must be provided as a dict! "
                        f"Got {transform.params}")
            return transform.params
        return {}  # the transform doesn't have any parameter

    if isinstance(transforms, td.Transform):
        if isinstance(transforms, td.ComposeTransform):
            builders, params = _get_transform_builders_params(transforms.parts)
            compose_transform_builder = lambda parts_params: td.ComposeTransform(
                [b(**p) for b, p in zip(builders, parts_params)])
            return compose_transform_builder, {'parts_params': params}
        else:
            builder = _get_transform_builder(transforms)
            params = _get_transform_params(transforms)
            return builder, params

    assert isinstance(transforms, list), f"Incorrect transforms {transforms}!"
    builders_and_params = [
        _get_transform_builders_params(t) for t in transforms
    ]
    builders, params = zip(*builders_and_params)
    return list(builders), list(params)


def _get_transformed_builder(obj: td.TransformedDistribution):
    # 'params' contains the dist params and all wrapped transform params starting
    # 'obj.base_dist' downwards
    builder, params = _get_builder(obj.base_dist)
    transform_builders, transform_params = _get_transform_builders_params(
        obj.transforms)
    new_builder = functools.partial(_builder_transformed, builder,
                                    transform_builders)
    new_params = {"params_": params, 'transforms_params_': transform_params}
    return new_builder, new_params


def _builder_affine_transformed(base_builder, loc_, scale_, **kwargs):
    # 'loc' and 'scale' may conflict with the names in kwargs. So we add suffix '_'.
    return AffineTransformedDistribution(base_builder(**kwargs), loc_, scale_)


def _get_affine_transformed_builder(obj: AffineTransformedDistribution):
    builder, params = _get_builder(obj.base_dist)
    new_builder = functools.partial(_builder_affine_transformed, builder,
                                    obj.loc, obj.scale)
    return new_builder, params


def _get_mixture_same_family_builder(obj: td.MixtureSameFamily):
    mixture_builder, mixture_params = _get_builder(obj.mixture_distribution)
    components_builder, components_params = _get_builder(
        obj.component_distribution)

    def _mixture_builder(mixture, components):
        return td.MixtureSameFamily(
            mixture_builder(**mixture), components_builder(**components))

    return _mixture_builder, {
        "mixture": mixture_params,
        "components": components_params
    }


_get_builder_map = {
    td.Categorical:
        _get_categorical_builder,
    td.OneHotCategorical:
        _get_categorical_builder,
    td.OneHotCategoricalStraightThrough:
        _get_categorical_builder,
    OneHotCategoricalStraightThrough:
        _get_categorical_builder,
    OneHotCategoricalGumbelSoftmax:
        _get_gumbelsoftmax_categorical_builder,
    td.Normal:
        lambda obj: (td.Normal, {
            'loc': obj.mean,
            'scale': obj.stddev
        }),
    StableCauchy:
        lambda obj: (StableCauchy, {
            'loc': obj.loc,
            'scale': obj.scale
        }),
    td.Independent:
        _get_independent_builder,
    DiagMultivariateNormal:
        lambda obj: (DiagMultivariateNormal, {
            'loc': obj.mean,
            'scale': obj.stddev
        }),
    DiagMultivariateCauchy:
        lambda obj: (DiagMultivariateCauchy, {
            'loc': obj.loc,
            'scale': obj.scale
        }),
    td.TransformedDistribution:
        _get_transformed_builder,
    AffineTransformedDistribution:
        _get_affine_transformed_builder,
    Beta:
        lambda obj: (Beta, {
            'concentration1': obj.concentration1,
            'concentration0': obj.concentration0
        }),
    DiagMultivariateBeta:
        lambda obj: (DiagMultivariateBeta, {
            'concentration1': obj.base_dist.concentration1,
            'concentration0': obj.base_dist.concentration0
        }),
    TruncatedNormal:
        lambda obj: (functools.partial(
            TruncatedNormal,
            lower_bound=obj.lower_bound,
            upper_bound=obj.upper_bound), {
                'loc': obj.loc,
                'scale': obj.scale
            }),
    TruncatedCauchy:
        lambda obj: (functools.partial(
            TruncatedCauchy,
            lower_bound=obj.lower_bound,
            upper_bound=obj.upper_bound), {
                'loc': obj.loc,
                'scale': obj.scale
            }),
    TruncatedT2:
        lambda obj: (functools.partial(
            TruncatedT2,
            lower_bound=obj.lower_bound,
            upper_bound=obj.upper_bound), {
                'loc': obj.loc,
                'scale': obj.scale
            }),
    td.MixtureSameFamily:
        _get_mixture_same_family_builder,
}


def _get_builder(obj):
    return _get_builder_map[type(obj)](obj)


def extract_distribution_parameters(dist: td.Distribution):
    """Extract the input parameters of a distribution.

    Args:
        dist (Distribution): distribution from which to extract parameters
    Returns:
        the nest of the input parameter of the distribution
    """
    return _get_builder(dist)[1]


class DistributionSpec(object):
    def __init__(self, builder, input_params_spec):
        """

        Args:
            builder (Callable): the function which is used to build the
                distribution. The returned value of ``builder(input_params)``
                is a ``Distribution`` with input parameter as ``input_params``.
            input_params_spec (nested TensorSpec): the spec for the argument of
                ``builder``.
        """
        self.builder = builder
        self.input_params_spec = input_params_spec

    def build_distribution(self, input_params):
        """Build a Distribution using ``input_params``.

        Args:
            input_params (nested Tensor): the parameters for build the
                distribution. It should match ``input_params_spec`` provided as
                ``__init__``.
        Returns:
            Distribution:
        """
        nest.assert_same_structure(input_params, self.input_params_spec)
        return self.builder(**input_params)

    @classmethod
    def from_distribution(cls, dist, from_dim=0):
        """Create a ``DistributionSpec`` from a ``Distribution``.
        Args:
            dist (Distribution): the ``Distribution`` from which the spec is
                extracted.
            from_dim (int): only use the dimensions from this. The reason of
                using ``from_dim>0`` is that ``[0, from_dim)`` might be batch
                dimension in some scenario.
        Returns:
            DistributionSpec:
        """
        builder, input_params = _get_builder(dist)
        input_param_spec = extract_spec(input_params, from_dim)
        return cls(builder, input_param_spec)


def extract_spec(nests, from_dim=1):
    """
    Extract ``TensorSpec`` or ``DistributionSpec`` for each element of a nested
    structure. It assumes that the first dimension of each element is the batch
    size.

    Args:
        nests (nested structure): each leaf node of the nested structure is a
            Tensor or Distribution of the same batch size.
        from_dim (int): ignore dimension before this when constructing the spec.
    Returns:
        nest: each leaf node of the returned nested spec is the corresponding
        spec (excluding batch size) of the element of ``nest``.
    """

    def _extract_spec(obj):
        if isinstance(obj, torch.Tensor):
            return TensorSpec.from_tensor(obj, from_dim)
        elif isinstance(obj, td.Distribution):
            return DistributionSpec.from_distribution(obj, from_dim)
        else:
            raise ValueError("Unsupported value type: %s" % type(obj))

    return nest.map_structure(_extract_spec, nests)


def to_distribution_param_spec(nests):
    """Convert the ``DistributionSpecs`` in nests to their parameter specs.

    Args:
        nests (nested DistributionSpec of TensorSpec):  Each ``DistributionSpec``
            will be converted to a dictionary of the spec of its input ``Tensor``
            parameters.
    Returns:
        nested TensorSpec: Each leaf is a ``TensorSpec`` or a ``dict``
        corresponding to one distribution, with keys as parameter name and
        values as ``TensorSpecs`` for the parameters.
    """

    def _to_param_spec(spec):
        if isinstance(spec, DistributionSpec):
            return spec.input_params_spec
        elif isinstance(spec, TensorSpec):
            return spec
        else:
            raise ValueError("Only TensorSpec or DistributionSpec is allowed "
                             "in nest, got %s. nest is %s" % (spec, nests))

    return nest.map_structure(_to_param_spec, nests)


def params_to_distributions(nests, nest_spec):
    """Convert distribution parameters to ``Distribution``, keep tensors unchanged.
    Args:
        nests (nested Tensor): a nested ``Tensor`` and dictionary of tensor
            parameters of ``Distribution``. Typically, ``nest`` is obtained using
            ``distributions_to_params()``.
        nest_spec (nested DistributionSpec and TensorSpec): The distribution
            params will be converted to ``Distribution`` according to the
            corresponding ``DistributionSpec`` in ``nest_spec``.
    Returns:
        nested Distribution or Tensor:
    """

    def _to_dist(spec, params):
        if isinstance(spec, DistributionSpec):
            return spec.build_distribution(params)
        elif isinstance(spec, TensorSpec):
            return params
        else:
            raise ValueError(
                "Only DistributionSpec or TensorSpec is allowed "
                "in nest_spec, got %s. nest_spec is %s" % (spec, nest_spec))

    return nest.map_structure_up_to(nest_spec, _to_dist, nest_spec, nests)


def distributions_to_params(nests):
    """Convert distributions to its parameters, and keep tensors unchanged.
    Only returns parameters that have ``Tensor`` values.

    Args:
        nests (nested Distribution and Tensor): Each ``Distribution`` will be
            converted to dictionary of its ``Tensor`` parameters.
    Returns:
        nested Tensor/Distribution: Each leaf is a ``Tensor`` or a ``dict``
        corresponding to one distribution, with keys as parameter name and
        values as tensors containing parameter values.
    """

    def _to_params(dist_or_tensor):
        if isinstance(dist_or_tensor, td.Distribution):
            return extract_distribution_parameters(dist_or_tensor)
        elif isinstance(dist_or_tensor, torch.Tensor):
            return dist_or_tensor
        else:
            raise ValueError(
                "Only Tensor or Distribution is allowed in nest, ",
                "got %s. nest is %s" % (dist_or_tensor, nests))

    return nest.map_structure(_to_params, nests)


def compute_entropy(distributions):
    """Computes total entropy of nested distribution.
    Args:
        distributions (nested Distribution): A possibly batched tuple of
            distributions.
    Returns:
        entropy
    """

    def _compute_entropy(dist: td.Distribution):
        entropy = dist.entropy()
        return entropy

    entropies = nest.map_structure(_compute_entropy, distributions)
    total_entropies = sum(nest.flatten(entropies))
    return total_entropies


def compute_log_probability(distributions, actions):
    """Computes log probability of actions given distribution.

    Args:
        distributions: A possibly batched tuple of distributions.
        actions: A possibly batched action tuple.

    Returns:
        Tensor: the log probability summed over actions in the batch.
    """

    def _compute_log_prob(single_distribution, single_action):
        single_log_prob = single_distribution.log_prob(single_action)
        return single_log_prob

    nest.assert_same_structure(distributions, actions)
    log_probs = nest.map_structure(_compute_log_prob, distributions, actions)
    total_log_probs = sum(nest.flatten(log_probs))
    return total_log_probs


def rsample_action_distribution(nested_distributions, return_log_prob=False):
    """Sample actions from distributions with reparameterization-based sampling.

    It uses ``Distribution.rsample()`` to do the sampling to enable backpropagation.

    Args:
        nested_distributions (nested Distribution): action distributions.
        return_log_prob (bool): whether to compute and return the log
            probability of the sampled actions, in addition to the sampled
            actions. In some cases, it is useful to compute the log probability
            immediately after the actions are sampled, as some subsequent
            operations might makes the cache mechanism (if turned on) invalid.
            Some example scenarios include 1) additional sampling operation
            applied on ``nested_distributions``, 2) some operations applied to
            the actions sampled from ``nested_distributions`` (e.g., cloning).
            This which could cause numerical issues if we want to compute the
            log probability for actions sampled at an early stage,
            especially for actions that are close to action bounds.
            For more details on PyTorch Transform, its cache mechanism, and its
            impacts on RL algorithms, please check
            `<https://alf.readthedocs.io/en/latest/notes/pytorch_notes.html#transform-bijector>`_.
    Returns:
        - rsampled actions if return_log_prob is False
        - rsampled actions and log_prob if return_log_prob is True
    """
    assert all(nest.flatten(nest.map_structure(lambda d: d.has_rsample,
                nested_distributions))), \
            ("all the distributions need to support rsample in order to enable "
            "backpropagation")
    sample = nest.map_structure(lambda d: d.rsample(), nested_distributions)
    if return_log_prob:
        log_prob = compute_log_probability(nested_distributions, sample)
        return sample, log_prob
    else:
        return sample


def sample_action_distribution(nested_distributions, return_log_prob=False):
    """Sample actions from distributions with conventional sampling without
        enabling backpropagation.
    Args:
        nested_distributions (nested Distribution): action distributions.
        return_log_prob (bool): whether to compute and return the log
            probability of the sampled actions, in addition to the sampled
            actions. In some cases, it is useful to compute the log probability
            immediately after the actions are sampled, as some subsequent
            operations might make the cache mechanism (if turned on) invalid.
            Some example scenarios include 1) additional sampling operation
            applied on ``nested_distributions``, 2) some operations applied to
            the actions sampled from ``nested_distributions`` (e.g., cloning).
            This which could cause numerical issues if we want to compute the
            log probability for actions sampled at an early stage,
            especially for actions that are close to action bounds.
            For more details on PyTorch Transform, its cache mechanism, and its
            impacts on RL algorithms, please check
            `<https://alf.readthedocs.io/en/latest/notes/pytorch_notes.html#transform-bijector>`_.
    Returns:
        - sampled actions if return_log_prob is False
        - sampled actions and log_prob if return_log_prob is True
    """
    sample = nest.map_structure(lambda d: d.sample(), nested_distributions)
    if return_log_prob:
        log_prob = compute_log_probability(nested_distributions, sample)
        return sample, log_prob
    else:
        return sample


def epsilon_greedy_sample(nested_distributions, eps=0.1):
    """Generate greedy sample that maximizes the probability.

    Args:
        nested_distributions (nested Distribution): distribution to sample from
        eps (float): a floating value in :math:`[0,1]`, representing the chance of
            action sampling instead of taking argmax. This can help prevent
            a dead loop in some deterministic environment like `Breakout`.
    Returns:
        (nested) Tensor:
    """

    def greedy_fn(dist):
        # pytorch distribution has no 'mode' operation
        greedy_action = get_mode(dist)
        if eps == 0.0:
            return greedy_action
        sample_action = dist.sample()
        greedy_mask = torch.rand(sample_action.shape[0]) > eps
        sample_action[greedy_mask] = greedy_action[greedy_mask]
        return sample_action

    if eps >= 1.0:
        return sample_action_distribution(nested_distributions)
    else:
        return nest.map_structure(greedy_fn, nested_distributions)


def get_mode(dist):
    """Get the mode of the distribution. Note that if ``dist`` is a transformed
    distribution, the result may not be the actual mode of ``dist``.

    Args:
        dist (td.Distribution):
    Returns:
        The mode of the distribution. If ``dist`` is a transformed distribution,
        the result is calculated by transforming the mode of its base
        distribution and may not be the actual mode for ``dist``.
    Raises:
        NotImplementedError: if dist or its base distribution is not
            ``td.Categorical``, ``td.Normal``, ``td.Independent`` or
            ``td.TransformedDistribution``.
    """
    if isinstance(dist, td.categorical.Categorical):
        mode = torch.argmax(dist.logits, -1)
    elif isinstance(
            dist,
        (OneHotCategoricalStraightThrough, OneHotCategoricalGumbelSoftmax)):
        # Our version of one-hot st supports mode with grad
        mode = dist.mode
    elif isinstance(
            dist, (td.OneHotCategorical, td.OneHotCategoricalStraightThrough)):
        mode = torch.nn.functional.one_hot(
            torch.argmax(dist.logits, -1), num_classes=dist.logits.shape[-1])
    elif isinstance(dist, td.normal.Normal):
        mode = dist.mean
    elif isinstance(dist, td.MixtureSameFamily):
        # Note that this just computes an approximate mode. We use an approximate
        # approach to compute the mode, by using the mode of the component
        # distribution that has the highest component probability.
        # [B]
        batch_shape = dist.batch_shape
        ind = get_mode(dist.mixture_distribution)
        # [B, num_component, d]
        component_mode = get_mode(dist.component_distribution)
        if len(batch_shape) == 1:
            mode = component_mode[torch.arange(batch_shape[0]), ind]
        elif len(batch_shape) == 2:
            d0, d1 = batch_shape
            mode = component_mode[torch.arange(d0).unsqueeze(-1),
                                  torch.arange(d1), ind]
        else:
            raise NotImplementedError(
                "Batch shape %s is not supported" % batch_shape)
    elif isinstance(dist, StableCauchy):
        mode = dist.loc
    elif isinstance(dist, td.Independent):
        mode = get_mode(dist.base_dist)
    elif isinstance(dist, td.TransformedDistribution):
        base_mode = get_mode(dist.base_dist)
        with torch.no_grad():
            mode = base_mode
            for transform in dist.transforms:
                mode = transform(mode)
    elif isinstance(dist, (Beta, TruncatedDistribution)):
        return dist.mode
    else:
        raise NotImplementedError(
            "Distribution type %s is not supported" % type(dist))

    return mode


def get_rmode(dist):
    """Get the mode of the distribution that support backpropogation.
    Note that if ``dist`` is a transformed
    distribution, the result may not be the actual mode of ``dist``.

    Args:
        dist (td.Distribution):
    Returns:
        The mode of the distribution. If ``dist`` is a transformed distribution,
        the result is calculated by transforming the mode of its base
        distribution and may not be the actual mode for ``dist``.
    Raises:
        NotImplementedError: if dist or its base distribution is not
            ``td.Normal``, ``StableCauchy``, ``Beta``, ``TruncatedDistribution``,
            ``td.Independent`` or ``td.TransformedDistribution``.
    """
    if isinstance(dist, td.normal.Normal):
        mode = dist.mean
    elif isinstance(dist, td.MixtureSameFamily):
        # note that for the mixture distribution, there is no gradient back-propagation
        # [B]
        ind = get_mode(dist.mixture_distribution)
        # [B, num_component, d]
        component_mode = get_rmode(dist.component_distribution)
        mode = component_mode[torch.arange(component_mode.shape[0]), ind]
    elif isinstance(dist, StableCauchy):
        mode = dist.loc
    elif isinstance(dist, Beta) or isinstance(dist, TruncatedDistribution):
        return dist.mode
    elif isinstance(dist, td.Independent):
        mode = get_rmode(dist.base_dist)
    elif isinstance(dist, td.TransformedDistribution):
        base_mode = get_rmode(dist.base_dist)
        mode = base_mode
        for transform in dist.transforms:
            mode = transform(mode)
    else:
        raise NotImplementedError(
            "Distribution type %s is not supported" % type(dist))

    return mode


def get_base_dist(dist):
    """Get the base distribution.

    Args:
        dist (td.Distribution):
    Returns:
        The base distribution if dist is ``td.Independent`` or
            ``td.TransformedDistribution``, and ``dist`` if it is ``td.Normal``.
    Raises:
        NotImplementedError: if ``dist`` or its based distribution is not
            ``td.Normal``, ``td.Independent`` or ``td.TransformedDistribution``.
    """
    if isinstance(dist, (td.Normal, td.Categorical, StableCauchy, Beta,
                         TruncatedDistribution)):
        return dist
    elif isinstance(dist, (td.Independent, td.TransformedDistribution)):
        return get_base_dist(dist.base_dist)
    else:
        raise NotImplementedError(
            "Distribution type %s is not supported" % type(dist))


@alf.configurable
def estimated_entropy(dist, num_samples=1, check_numerics=False):
    r"""Estimate entropy by sampling.

    Use sampling to calculate entropy. The unbiased estimator for entropy is
    :math:`-\log(p(x))` where :math:`x` is an unbiased sample of :math:`p`.
    However, the gradient of :math:`-\log(p(x))` is not an unbiased estimator
    of the gradient of entropy. So we also calculate a value whose gradient is
    an unbiased estimator of the gradient of entropy. See ``notes/subtleties_of_estimating_entropy.py``
    for detail.

    Args:
        dist (torch.distributions.Distribution): concerned distribution
        num_samples (int): number of random samples used for estimating entropy.
        check_numerics (bool): If true, find NaN / Inf values. For debugging only.
    Returns:
        tuple:
        - entropy
        - entropy_for_gradient: for calculating gradient.
    """
    sample_shape = (num_samples, )
    if dist.has_rsample:
        single_action = dist.rsample(sample_shape=sample_shape)
    else:
        single_action = dist.sample(sample_shape=sample_shape)
    if single_action.dtype.is_floating_point and dist.has_rsample:
        entropy = -dist.log_prob(single_action)
        if check_numerics:
            assert torch.all(torch.isfinite(entropy))
        entropy = entropy.mean(dim=0)
        entropy_for_gradient = entropy
    else:
        entropy = -dist.log_prob(single_action.detach())
        if check_numerics:
            assert torch.all(torch.isfinite(entropy))
        entropy_for_gradient = -0.5 * entropy**2
        entropy = entropy.mean(dim=0)
        entropy_for_gradient = entropy_for_gradient.mean(dim=0)
    return entropy, entropy_for_gradient


# NOTE(hnyu): It might be possible to get a closed-form of entropy given a
# Normal as the base dist with only affine transformation?
# It's better (lower variance) than this estimated one.
#
# Something like what TFP does:
# https://github.com/tensorflow/probability/blob/356cfddef026b3339b8f2a81e600acd2ff8e22b4/tensorflow_probability/python/distributions/transformed_distribution.py#L636
# (Probably it's complicated, but we need to spend time figuring out if the
# current estimation is the best way to do this).


# Here, we compute entropy of transformed distributions using sampling.
def entropy_with_fallback(distributions, return_sum=True):
    r"""Computes total entropy of nested distribution.
    If ``entropy()`` of a distribution is not implemented, this function will
    fallback to use sampling to calculate the entropy. It returns two values:
    ``(entropy, entropy_for_gradient)``.

    There are two situations:

    - ``entropy()`` is implemented and the same as ``entropy_for_gradient``.
    - ``entropy()`` is not implemented. We use sampling to calculate entropy. The
      unbiased estimator for entropy is :math:`-\log(p(x))`. However, the gradient
      of :math:`-\log(p(x))` is not an unbiased estimator of the gradient of
      entropy. So we also calculate a value whose gradient is an unbiased
      estimator of the gradient of entropy. See ``estimated_entropy()`` for detail.

    Examples:

    .. code-block:: python

        ent, ent_for_grad = entropy_with_fall_back(dist, action_spec)
        alf.summary.scalar("entropy", ent)
        ent_for_grad.backward()

    Args:
        distributions (nested Distribution): A possibly batched tuple of
            distributions.
        return_sum (bool): if True, return the total entropy. If not True,
            return the entropy for each distribution in the nest.

    Returns:
        tuple:
        - entropy
        - entropy_for_gradient: You should use ``entropy`` in situations where its
          value is needed, and ``entropy_for_gradient`` where you need to calculate the
          gradient of entropy.
    """

    def _compute_entropy(dist: td.Distribution):
        if isinstance(dist, AffineTransformedDistribution):
            entropy, entropy_for_gradient = _compute_entropy(dist.base_dist)
            entropy = entropy + dist._log_abs_scale
            entropy_for_gradient = entropy_for_gradient + dist._log_abs_scale
        elif isinstance(dist, (td.TransformedDistribution,
                               TruncatedDistribution, td.MixtureSameFamily)):
            # TransformedDistribution is used by NormalProjectionNetwork with
            # scale_distribution=True, in which case we estimate with sampling.
            entropy, entropy_for_gradient = estimated_entropy(dist)
        else:
            entropy = dist.entropy()
            entropy_for_gradient = entropy
        return entropy, entropy_for_gradient

    entropies = list(map(_compute_entropy, nest.flatten(distributions)))
    entropies, entropies_for_gradient = zip(*entropies)

    if return_sum:
        return sum(entropies), sum(entropies_for_gradient)
    else:
        return (nest.pack_sequence_as(distributions, entropies),
                nest.pack_sequence_as(distributions, entropies_for_gradient))


@alf.configurable
def calc_default_target_entropy(spec, min_prob=0.1):
    """Calculate default target entropy.

    Args:
        spec (BoundedTensorSpec): action spec
        min_prob (float): If continuous spec, we suppose the prob concentrates on
            a delta of ``min_prob * (M-m)``; if discrete spec, we uniformly
            distribute ``min_prob`` on all entries except the peak which has
            a probability of ``1 - min_prob``.
    Returns:
        target entropy
    """

    def _calc_discrete_entropy(m, M, log_mp):
        N = M - m + 1
        if N == 1:
            return 0
        return (min_prob * (np.log(N - 1) - log_mp) -
                (1 - min_prob) * np.log(1 - min_prob))

    zeros = np.zeros(spec.shape)
    min_max = np.broadcast(spec.minimum, spec.maximum, zeros)
    cont = spec.is_continuous
    log_mp = np.log(min_prob + 1e-30)
    e = np.sum([(np.log(M - m) + log_mp if cont else _calc_discrete_entropy(
        m, M, log_mp)) for m, M, _ in min_max])
    return e


@alf.configurable
def calc_default_target_entropy_quantized(spec,
                                          num_bins,
                                          ent_per_action_dim=-1.0):
    """Calc default target entropy for quantized continuous action.
    Args:
        spec (BoundedTensorSpec): action spec
        num_bins (int): number of quantization bins used to represent the
            continuous action
        ent_per_action_dim (int): desired entropy per action dimension
            for the non-quantized continuous action; default value is -1.0
            as suggested by the SAC paper.
    Returns:
        target entropy for quantized representation
    """
    cont = spec.is_continuous
    assert cont, "only support continuous action-based computation"

    log_Mn = np.log(spec.maximum - spec.minimum)
    log_mp = ent_per_action_dim - log_Mn
    log_B = np.log(num_bins)

    ents = [log_mp + log_B for _ in range(spec.shape[0])]
    e = np.sum(ents)

    assert e > 0, "wrong target entropy for discrete distribution {}".format(e)
    return e


def calc_default_max_entropy(spec, fraction=0.8):
    """Calc default max entropy.
    Args:
        spec (BoundedTensorSpec): action spec
        fraction (float): this fraction of the theoretical entropy upper bound
            will be used as the max entropy
    Returns:
        A default max entropy for adjusting the entropy weight
    """
    assert fraction <= 1.0 and fraction > 0
    zeros = np.zeros(spec.shape)
    min_max = np.broadcast(spec.minimum, spec.maximum, zeros)
    cont = spec.is_continuous
    # use uniform distributions to compute upper bounds
    e = np.sum([(np.log(M - m) * (fraction if M - m > 1 else 1.0 / fraction)
                 if cont else np.log(M - m + 1) * fraction)
                for m, M, _ in min_max])
    return e


def calc_uniform_log_prob(spec):
    """Given an action spec, calculate the uniform log prob.

    Args:
        spec (BoundedTensorSpec): action spec must be a bounded spec

    Returns:
        The uniform log probability
    """
    assert isinstance(spec, BoundedTensorSpec)
    zeros = np.zeros(spec.shape)
    min_max = np.broadcast(spec.minimum, spec.maximum, zeros)
    if spec.is_continuous:
        log_prob = np.sum([-np.log(M - m) for m, M, _ in min_max])
    else:
        log_prob = np.sum([-np.log(M - m + 1) for m, M, _ in min_max])
    return log_prob
