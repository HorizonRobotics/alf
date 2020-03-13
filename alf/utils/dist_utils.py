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

import functools
import gin
import hashlib
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

import alf.nest as nest
from alf.tensor_specs import TensorSpec


class OUProcess(nn.Module):
    """A zero-mean Ornstein-Uhlenbeck process."""

    def __init__(self, initial_value, damping=0.15, stddev=0.2):
        """A Class for generating noise from a zero-mean Ornstein-Uhlenbeck process.
        The Ornstein-Uhlenbeck process is a process that generates temporally
        correlated noise via a random walk with damping. This process describes
        the velocity of a particle undergoing brownian motion in the presence of
        friction. This can be useful for exploration in continuous action
        environments with momentum.
        The temporal update equation is:
        `x_next = (1 - damping) * x + N(0, std_dev)`
        Args:
        initial_value: Initial value of the process.
        damping: The rate at which the noise trajectory is damped towards the
            mean. We must have 0 <= damping <= 1, where a value of 0 gives an
            undamped random walk and a value of 1 gives uncorrelated Gaussian noise.
            Hence in most applications a small non-zero value is appropriate.
        stddev: Standard deviation of the Gaussian component.
        """
        super(OUProcess, self).__init__()
        self._damping = damping
        self._stddev = stddev
        self._x = initial_value
        self._x.requires_grad = False

    def forward(self):
        noise = torch.randn_like(self._x) * self._stddev
        self._x.data.copy_((1 - self._damping) * self._x + noise)
        return self._x


def DiagMultivariateNormal(loc, scale_diag):
    """Create a Normal distribution with diagonal variance."""
    return td.Independent(td.Normal(loc, scale_diag), 1)


def _builder_independent(base_builder, reinterpreted_batch_ndims, **kwargs):
    return td.Independent(base_builder(**kwargs), reinterpreted_batch_ndims)


def _builder_transformed(base_builder, transforms, **kwargs):
    return td.TransformedDistribution(base_builder(**kwargs), transforms)


def _get_builder(obj):
    if type(obj) == td.Categorical:
        return td.Categorical, {'logits': obj.logits}
    elif type(obj) == td.Normal:
        return td.Normal, {'loc': obj.mean, 'scale': obj.stddev}
    elif type(obj) == td.Independent:
        builder, params = _get_builder(obj.base_dist)
        new_builder = functools.partial(_builder_independent, builder,
                                        obj.reinterpreted_batch_ndims)
        return new_builder, params
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
        """Create a DistributionSpec instance.

        Args:
            builder (Callable): the function which is used to build the
                distribution. The returned value of `builder(input_params)`
                is a Distribution with input parameter as `input_params`
            input_params_spec (nested TensorSpec): the spec for the argument of
                `builder`
        """
        self.builder = builder
        self.input_params_spec = input_params_spec

    def build_distribution(self, input_params):
        """Build a Distribution using `input_params`

        Args:
            input_params (nested Tensor): the parameters for build the
                distribution. It should match `input_params_spec` provided as
                `__init__`
        Returns:
            A Distribution
        """
        nest.assert_same_structure(input_params, self.input_params_spec)
        return self.builder(**input_params)

    @classmethod
    def from_distribution(cls, dist, from_dim=0):
        """Create a DistributionSpec from a Distribution.
        Args:
            dist (Distribution): the Distribution from which the spec is
                extracted.
            from_dim (int): only use the dimenions from this. The reason of
                using `from_dim`>0 is that [0, from_dim) might be batch
                dimension in some scenario.
        Returns:
            DistributionSpec
        """
        builder, input_params = _get_builder(dist)
        input_param_spec = nest.map_structure(
            lambda tensor: TensorSpec.from_tensor(tensor, from_dim),
            input_params)
        return cls(builder, input_param_spec)


def extract_spec(nests, from_dim=1):
    """
    Extract TensorSpec or DistributionSpec for each element of a nested structure.
    It assumes that the first dimension of each element is the batch size.

    Args:
        nests (nested structure): each leaf node of the nested structure is a
            Tensor or Distribution of the same batch size
        from_dim (int): ignore dimension before this when constructing the spec.
    Returns:
        spec (nested structure): each leaf node of the returned nested spec is the
            corresponding spec (excluding batch size) of the element of `nest`
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
    """Convert the DistributionSpecs in nests to their parameter specs.

    Args:
        nests (nested DistributionSpec of TensorSpec):  Each DistributionSpec
            will be converted to a dictionary of the spec of its input Tensor
            parameters.
    Returns:
        A nest of TensorSpec/dict[TensorSpec]. Each leaf is a TensorSpec or a
        dict corresponding to one distribution, with keys as parameter name and
        values as TensorSpecs for the parameters.
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
    """Convert distribution parameters to Distribution, keep Tensors unchanged.
    Args:
        nests (nested tf.Tensor): nested Tensor and dictionary of the Tensor
            parameters of Distribution. Typically, `nest` is obtained using
            `distributions_to_params()`
        nest_spec (nested DistributionSpec and TensorSpec): The distribution
            params will be converted to Distribution according to the
            corresponding DistributionSpec in nest_spec
    Returns:
        nested Distribution/Tensor
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
    """Convert distributions to its parameters, keep Tensors unchanged.
    Only returns parameters that have tf.Tensor values.
    Args:
        nests (nested Distribution and Tensor): Each Distribution will be
            converted to dictionary of its Tensor parameters.
    Returns:
        A nest of Tensor/Distribution parameters. Each leaf is a Tensor or a
        dict corresponding to one distribution, with keys as parameter name and
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
        A Tensor representing the log probability of each action in the batch.
    """

    def _compute_log_prob(single_distribution, single_action):
        single_log_prob = single_distribution.log_prob(single_action)
        return single_log_prob

    nest.assert_same_structure(distributions, actions)
    log_probs = nest.map_structure(_compute_log_prob, distributions, actions)
    total_log_probs = sum(nest.flatten(log_probs))
    return total_log_probs


def rsample_action_distribution(nested_distributions):
    """Sample actions from distributions with reparameterization-based sampling
        (rsample) to enable backpropagation.
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
        eps (float): a floating value in [0,1], representing the chance of
            action sampling instead of taking argmax. This can help prevent
            a dead loop in some deterministic environment like Breakout.
    Returns:
        (nested) Tensor
    """

    def greedy_fn(dist):
        # pytorch distribution has no 'mode' operation
        sample_action = dist.sample()
        greedy_mask = torch.rand(sample_action.shape[0]) > eps
        if isinstance(dist, td.categorical.Categorical):
            greedy_action = torch.argmax(dist.logits, -1)
        elif isinstance(dist, td.normal.Normal):
            greedy_action = dist.mean
        else:
            raise NotImplementedError("Mode sampling not implemented for "
                                      "{cls}".format(cls=type(dist)))
        sample_action[greedy_mask] = greedy_action[greedy_mask]
        return sample_action

    if eps >= 1.0:
        return sample_action_distribution(nested_distributions)
    else:
        return nest.map_structure(greedy_fn, nested_distributions)


def get_base_dist(dist):
    """Get the base distribution.

    Args:
        dist (td.Distribution)
    Returns:
        the base distribution if dist is td.Independent or
            td.TransformedDistribution, and dist if dist is td.Normal
    Raises:
        NotImplementedError if dist or its based distribution is not
            td.Normal, td.Independent or td.TransformedDistribution
    """
    if isinstance(dist, td.Normal) or isinstance(dist, td.Categorical):
        return dist
    elif isinstance(dist, (td.Independent, td.TransformedDistribution)):
        return get_base_dist(dist.base_dist)
    else:
        raise NotImplementedError(
            "Distribution type %s is not supported" % type(dist))


@gin.configurable
def estimated_entropy(dist, num_samples=1, check_numerics=False):
    """Estimate entropy by sampling.

    Use sampling to calculate entropy. The unbiased estimator for entropy is
    -log(p(x)) where x is an unbiased sample of p. However, the gradient of
    -log(p(x)) is not an unbiased estimator of the gradient of entropy. So we
    also calculate a value whose gradient is an unbiased estimator of the
    gradient of entropy. See docs/subtleties_of_estimating_entropy.py for
    detail.

    Args:
        dist (torch.distributions.Distribution): concerned distribution
        num_samples (int): number of random samples used for estimating entropy.
        check_numerics (bool): If true, adds tf.debugging.check_numerics to
            help find NaN / Inf values. For debugging only.
    Returns:
        tuple of (entropy, entropy_for_gradient). entropy_for_gradient is for
        calculating gradient
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
def entropy_with_fallback(distributions, action_spec):
    """Computes total entropy of nested distribution.
    If entropy() of a distribution is not implemented, this function will
    fallback to use sampling to calculate the entropy. It returns two values:
    (entropy, entropy_for_gradient).
    There are two situations:
    * entropy() is implemented. entropy is same as entropy_for_gradient.
    * entropy() is not implemented. We use sampling to calculate entropy. The
        unbiased estimator for entropy is -log(p(x)). However, the gradient of
        -log(p(x)) is not an unbiased estimator of the gradient of entropy. So
        we also calculate a value whose gradient is an unbiased estimator of
        the gradient of entropy. See estimated_entropy() for detail.
    Example:
        ent, ent_for_grad = entropy_with_fall_back(dist, action_spec)
        alf.summary.scalar("entropy", ent)
        ent_for_grad.backward()
    Args:
        distributions (nested Distribution): A possibly batched tuple of
            distributions.
        action_spec (nested BoundedTensorSpec): A nested tuple representing the
            action spec.
    Returns:
        tuple of (entropy, entropy_for_gradient). You should use entropy in
        situations where its value is needed, and entropy_for_gradient where
        you need to calculate the gradient of entropy.
    """

    def _compute_entropy(dist: td.Distribution, action_spec):
        if isinstance(dist, td.TransformedDistribution):
            # TransformedDistribution is used by NormalProjectionNetwork with
            # scale_distribution=True, in which case we estimate with sampling.
            entropy, entropy_for_gradient = estimated_entropy(dist)
        else:
            entropy = dist.entropy()
            entropy_for_gradient = entropy
        return entropy, entropy_for_gradient

    entropies = list(
        map(_compute_entropy, nest.flatten(distributions),
            nest.flatten(action_spec)))
    entropies, entropies_for_gradient = zip(*entropies)

    return sum(entropies), sum(entropies_for_gradient)


def calc_default_target_entropy(spec):
    """Calc default target entropy
    Args:
        spec (TensorSpec): action spec
    Returns:
    """
    zeros = np.zeros(spec.shape)
    min_max = np.broadcast(spec.minimum, spec.maximum, zeros)
    cont = spec.is_continuous
    min_prob = 0.01
    log_mp = np.log(min_prob)
    # continuous: suppose the prob concentrates on a delta of 0.01*(M-m)
    # discrete: ignore the entry of 0.99 and uniformly distribute probs on rest
    e = np.sum([(np.log(M - m) + log_mp
                 if cont else min_prob * (np.log(M - m) - log_mp))
                for m, M, _ in min_max])
    return e


def calc_default_max_entropy(spec, fraction=0.8):
    """Calc default max entropy
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
