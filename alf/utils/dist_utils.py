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
import numpy as np
import torch
import torch.distributions as td

import alf.nest as nest
from alf.tensor_specs import TensorSpec


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
    elif type(obj) == td.TransformedDistribution:
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

    def _compute_entropy(dist: torch.distributions.Distribution):
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
        if isinstance(dist, torch.distributions.categorical.Categorical):
            greedy_action = torch.argmax(dist.logits, -1)
        elif isinstance(dist, torch.distributions.normal.Normal):
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
def estimated_entropy(dist,
                      seed=None,
                      assume_reparametrization=False,
                      num_samples=1,
                      check_numerics=False):
    """Estimate entropy by sampling.

    Use sampling to calculate entropy. The unbiased estimator for entropy is
    -log(p(x)) where x is an unbiased sample of p. However, the gradient of
    -log(p(x)) is not an unbiased estimator of the gradient of entropy. So we
    also calculate a value whose gradient is an unbiased estimator of the
    gradient of entropy. See docs/subtleties_of_estimating_entropy.py for
    detail.

    Args:
        dist (tfp.distributions.Distribution): concerned distribution
        seed (Any): Any Python object convertible to string, supplying the
            initial entropy.
        assume_reparametrization (bool): assume the sample from continuous
            distribution is generated by transforming a fixed distribution
            by a parameterized function. If we can assume this,
            entropy_for_gradient will have lower variance. We make the default
            to be False to be safe.
        num_samples (int): number of random samples used for estimating entropy.
        check_numerics (bool): If true, adds tf.debugging.check_numerics to
            help find NaN / Inf values. For debugging only.
    Returns:
        tuple of (entropy, entropy_for_gradient). entropy_for_gradient is for
        calculating gradient
    """
    sample_shape = (num_samples, )
    single_action = dist.sample(sample_shape=sample_shape, seed=seed)
    if single_action.dtype.is_floating and assume_reparametrization:
        entropy = -dist.log_prob(single_action)
        if check_numerics:
            entropy = tf.debugging.check_numerics(entropy, 'entropy')
        entropy = tf.reduce_mean(entropy, axis=0)
        entropy_for_gradient = entropy
    else:
        entropy = -dist.log_prob(tf.stop_gradient(single_action))
        if check_numerics:
            entropy = tf.debugging.check_numerics(entropy, 'entropy')
        entropy_for_gradient = -0.5 * tf.math.square(entropy)
        entropy = tf.reduce_mean(entropy, axis=0)
        entropy_for_gradient = tf.reduce_mean(entropy_for_gradient, axis=0)
    return entropy, entropy_for_gradient


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
    cont = tensor_spec.is_continuous(spec)
    # use uniform distributions to compute upper bounds
    e = np.sum([(np.log(M - m) * (fraction if M - m > 1 else 1.0 / fraction)
                 if cont else np.log(M - m + 1) * fraction)
                for m, M, _ in min_max])
    return e
