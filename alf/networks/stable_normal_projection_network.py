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
"""Project inputs to a normal distribution object."""
import math

import gin.tf

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import utils as network_utils
from tf_agents.networks.normal_projection_network import NormalProjectionNetwork
from tf_agents.networks.normal_projection_network import tanh_squash_to_spec


@gin.configurable
class StableNormalProjectionNetwork(NormalProjectionNetwork):
    """Generates a tfp.distribution.Normal by predicting a mean and std.

    It parameterizes the normal distributions as std=c0+1/(c1+softplus(b))
    and mean=a*std where a and b are outputs from means_projection_layer and
    stds_projectin_layer respectively. c0 and c1 are chosen so that
    min_std <= std <= max_std. The advantage of this parameterization is that
    its second order derivatives with respect to a and b are bounded even when
    the standard deviations become very small so that the optimization is
    more stable. See docs/stable_gradient_descent_for_gaussian_distribution.py
    for detail.
    """

    def __init__(self,
                 sample_spec,
                 activation_fn=None,
                 init_means_output_factor=1e-10,
                 mean_transform=tanh_squash_to_spec,
                 inverse_std_transform='softplus',
                 state_dependent_std=False,
                 scale_distribution=False,
                 init_std=1.0,
                 min_std=0.0,
                 max_std=None):
        """Creates an instance of StableNormalProjectionNetwork.

        Args:
            sample_spec (BoundedTensorSpec): detailing the shape and dtypes of
                samples pulled from the output distribution.
            activation_fn (Callable): Activation function to use in dense layer.
            init_means_output_factor (float): Output factor for initializing
                action means weights.
            mean_transform (Callable): Transform to apply to the calculated
                means. Uses `tanh_squash_to_spec` by default.
            inverse_std_transform (Callable): Currently supports tf.math.exp and
                tf.nn.softplus. Transformation to obtain inverse std. The
                transformed values are further transformed according to min_std
                and max_std.
            state_dependent_std (bool): If true, stddevs will be produced by MLP
                from state. else, stddevs will be an independent variable.
            scale_distribution (bool): Whether or not to use a bijector chain to
                scale distributions to match the sample spec. Note the
                TransformedDistribution does not support certain operations
                required by some agents or policies such as KL divergence
                calculations or Mode.
            init_std (float): Initial value for standard deviation.
            min_std (float): Minimum value for standard deviation.
            max_std (float): Maximum value for standard deviation. If None, no
                maximum is enforced.
        """
        self._min_std = min_std
        self._max_std = max_std

        c = 1 / (init_std - min_std)
        if max_std is not None:
            c -= 1 / (max_std - min_std)

        if inverse_std_transform == 'exp':
            std_transform = tf.math.exp
            std_bias_initializer_value = math.log(c)
        elif inverse_std_transform == 'softplus':
            std_transform = tf.nn.softplus
            std_bias_initializer_value = math.log(math.exp(c) - 1)
        else:
            raise TypeError(
                "Unsupported std_transform %s" % inverse_std_transform)

        super().__init__(
            sample_spec=sample_spec,
            activation_fn=activation_fn,
            init_means_output_factor=init_means_output_factor,
            std_bias_initializer_value=std_bias_initializer_value,
            mean_transform=mean_transform,
            std_transform=std_transform,
            state_dependent_std=state_dependent_std,
            scale_distribution=scale_distribution)

    def call(self, inputs, outer_rank):
        if inputs.dtype != self._sample_spec.dtype:
            raise ValueError(
                'Inputs to NormalProjectionNetwork must match the sample_spec.dtype.'
            )
        # outer_rank is needed because the projection is not done on the raw
        # observations so getting the outer rank is hard as there is no spec to
        # compare to.
        batch_squash = network_utils.BatchSquash(outer_rank)
        inputs = batch_squash.flatten(inputs)

        means = self._means_projection_layer(inputs)
        means = tf.reshape(means, [-1] + self._sample_spec.shape.as_list())

        if self._state_dependent_std:
            stds = self._stddev_projection_layer(inputs)
        else:
            stds = self._bias(tf.zeros_like(means))
            stds = tf.reshape(stds, [-1] + self._sample_spec.shape.as_list())

        inv_stds = self._std_transform(stds)
        if self._max_std is not None:
            inv_stds += 1 / (self._max_std - self._min_std)
        stds = 1. / inv_stds
        if self._min_std > 0:
            stds += self._min_std
        stds = tf.cast(stds, self._sample_spec.dtype)

        means = means * stds

        # If not scaling the distribution later, use a normalized mean.
        if not self._scale_distribution and self._mean_transform is not None:
            means = self._mean_transform(means, self._sample_spec)
        means = tf.cast(means, self._sample_spec.dtype)

        means = batch_squash.unflatten(means)
        stds = batch_squash.unflatten(stds)

        return self.output_spec.build_distribution(loc=means, scale=stds)
