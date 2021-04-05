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
import gin
import hashlib
import numpy as np
import math
import torch
import torch.distributions as td
from torch.distributions import constraints
import torch.nn as nn

import alf
import alf.nest as nest
import alf.nest.utils as nest_utils
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, spec_utils


def get_invertable(cls):
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


AbsTransform = get_invertable(td.AbsTransform)
AffineTransform = get_invertable(td.AffineTransform)
ExpTransform = get_invertable(td.ExpTransform)
PowerTransform = get_invertable(td.PowerTransform)
SigmoidTransform = get_invertable(td.SigmoidTransform)
SoftmaxTransform = get_invertable(td.SoftmaxTransform)


@alf.configurable
class Softsign(td.Transform):
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self):
        super().__init__(cache_size=1)

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


@alf.configurable
class StableTanh(td.Transform):
    r"""Invertable transformation (bijector) that computes :math:`Y = tanh(X)`,
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


# The pytorch kl_divergence has a bug
# (https://github.com/pytorch/pytorch/issues/34859)
# So we use our own:
@td.kl.register_kl(td.TransformedDistribution, td.TransformedDistribution)
def _kl_transformed_transformed(p, q):
    if p.transforms != q.transforms:
        raise NotImplementedError
    if p.event_shape != q.event_shape:
        raise NotImplementedError
    return td.kl.kl_divergence(p.base_dist, q.base_dist)


class OUProcess(nn.Module):
    """A zero-mean Ornstein-Uhlenbeck process for generating noises."""

    def __init__(self, initial_value, damping=0.15, stddev=0.2):
        """
        The Ornstein-Uhlenbeck process is a process that generates temporally
        correlated noise via a random walk with damping. This process describes
        the velocity of a particle undergoing brownian motion in the presence of
        friction. This can be useful for exploration in continuous action
        environments with momentum.

        The temporal update equation is:

        .. code-block:: python

            x_next = (1 - damping) * x + N(0, std_dev)

        Args:
            initial_value (Tensor): Initial value of the process.
            damping (float): The rate at which the noise trajectory is damped
                towards the mean. We must have :math:`0 <= damping <= 1`, where
                a value of 0 gives an undamped random walk and a value of 1 gives
                uncorrelated Gaussian noise. Hence in most applications a small
                non-zero value is appropriate.
            stddev (float): Standard deviation of the Gaussian component.
        """
        super(OUProcess, self).__init__()
        self._damping = damping
        self._stddev = stddev
        self._x = initial_value.clone().detach()

    def forward(self):
        noise = torch.randn_like(self._x) * self._stddev
        return self._x.data.copy_((1 - self._damping) * self._x + noise)


class DiagMultivariateNormal(td.Independent):
    def __init__(self, loc, scale):
        """Create multivariate normal distribution with diagonal variance.

        Args:
            loc (Tensor): mean of the distribution
            scale (Tensor): standard deviation. Should have same shape as ``loc``.
        """
        super().__init__(td.Normal(loc, scale), reinterpreted_batch_ndims=1)

    @property
    def stddev(self):
        return self.base_dist.stddev


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


def _builder_independent(base_builder, reinterpreted_batch_ndims, **kwargs):
    return td.Independent(base_builder(**kwargs), reinterpreted_batch_ndims)


def _builder_transformed(base_builder, transforms, **kwargs):
    return td.TransformedDistribution(base_builder(**kwargs), transforms)


def _get_builder(obj):
    if type(obj) == td.Categorical:
        if 'probs' in obj.__dict__ and id(obj.probs) == id(obj._param):
            # This means that obj is constructed using probs
            return td.Categorical, {'probs': obj.probs}
        else:
            return td.Categorical, {'logits': obj.logits}
    elif type(obj) == td.Normal:
        return td.Normal, {'loc': obj.mean, 'scale': obj.stddev}
    elif type(obj) == StableCauchy:
        return StableCauchy, {'loc': obj.loc, 'scale': obj.scale}
    elif type(obj) == td.Independent:
        builder, params = _get_builder(obj.base_dist)
        new_builder = functools.partial(_builder_independent, builder,
                                        obj.reinterpreted_batch_ndims)
        return new_builder, params
    elif type(obj) == DiagMultivariateNormal:
        return DiagMultivariateNormal, {'loc': obj.mean, 'scale': obj.stddev}
    elif type(obj) == DiagMultivariateCauchy:
        return DiagMultivariateCauchy, {'loc': obj.loc, 'scale': obj.scale}
    elif isinstance(obj, td.TransformedDistribution):
        builder, params = _get_builder(obj.base_dist)
        new_builder = functools.partial(_builder_transformed, builder,
                                        obj.transforms)
        return new_builder, params
    else:
        raise ValueError("Unsupported value type: %s" % type(obj))


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
            from_dim (int): only use the dimenions from this. The reason of
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


def rsample_action_distribution(nested_distributions):
    """Sample actions from distributions with reparameterization-based sampling.

    It uses ``Distribution.rsample()`` to do the sampling to enable backpropagation.

    Args:
        nested_distributions (nested Distribution): action distributions.
    Returns:
        rsampled actions
    """
    assert all(nest.flatten(nest.map_structure(lambda d: d.has_rsample,
                nested_distributions))), \
            ("all the distributions need to support rsample in order to enable "
            "backpropagation")
    return nest.map_structure(lambda d: d.rsample(), nested_distributions)


def sample_action_distribution(nested_distributions):
    """Sample actions from distributions with conventional sampling without
        enabling backpropagation.
    Args:
        nested_distributions (nested Distribution): action distributions.
    Returns:
        sampled actions
    """
    return nest.map_structure(lambda d: d.sample(), nested_distributions)


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
    elif isinstance(dist, td.normal.Normal):
        mode = dist.mean
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
    if (isinstance(dist, td.Normal) or isinstance(dist, td.Categorical)
            or isinstance(dist, StableCauchy)):
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
    an unbiased estimator of the gradient of entropy. See :doc:`notes/subtleties_of_estimating_entropy`
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

    - ``entropy()`` is implemented  and it's same as ``entropy_for_gradient``.
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
        if isinstance(dist, td.TransformedDistribution):
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
        spec (TensorSpec): action spec
        min_prob (float): If continuous spec, we suppose the prob concentrates on
            a delta of ``min_prob * (M-m)``; if discrete spec, we ignore the entry
            of ``1 - min_prob`` and uniformly distribute probs on rest.
    Returns:
        target entropy
    """

    def _calc_discrete_entropy(m, M, log_mp):
        N = M - m + 1
        if N == 1:
            return 0
        return min_prob * (np.log(N - 1) - log_mp)

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
        spec (TensorSpec): action spec
        num_bins (int): number of quantization bins used to represent the
            continuous action
        ent_per_action_dim (int): desired entropy per action dimension
            for the non-quantized continuous action; default value is -1.0
            as suggested by the SAC paper.
    Returns:
        target entropy for quantized representation
    """

    zeros = np.zeros(spec.shape)
    min_max = np.broadcast(spec.minimum, spec.maximum, zeros)

    cont = spec.is_continuous
    assert cont, "only support continuous action-based computation"

    log_Mn = np.log(spec.maximum - spec.minimum)
    log_mp = ent_per_action_dim - log_Mn
    log_B = np.log(num_bins)

    ents = [log_mp + log_B for i in range(spec.shape[0])]
    e = np.sum(ents)

    assert e > 0, "wrong target entropy for discrete distribution {}".format(e)
    return e


def calc_default_max_entropy(spec, fraction=0.8):
    """Calc default max entropy.
    Args:
        spec (TensorSpec): action spec
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
