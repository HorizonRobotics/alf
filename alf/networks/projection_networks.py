# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import gin
import math
import numpy as np
from typing import Callable

import torch
import torch.nn as nn
import torch.distributions as td

import alf.layers as layers
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks.network import DistributionNetwork, Network
from alf.utils import dist_utils


def DiagMultivariateNormal(loc, scale_diag):
    return td.Independent(td.Normal(loc, scale_diag), 1)


@gin.configurable
class CategoricalProjectionNetwork(DistributionNetwork):
    def __init__(self,
                 input_size,
                 action_spec,
                 logits_init_output_factor=0.1,
                 name="CategoricalProjectionNetwork"):
        """Creates a categorical projection network that outputs a discrete
        distribution over a number of classes.

        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        Args:
            input_size (int): the input vector size
            action_spec (BounedTensorSpec): a tensor spec containing the information
                of the output distribution.
            name (str):
        """
        super(CategoricalProjectionNetwork, self).__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        unique_num_actions = np.unique(action_spec.maximum -
                                       action_spec.minimum + 1)
        if len(unique_num_actions) > 1 or np.any(unique_num_actions <= 0):
            raise ValueError(
                'Bounds on discrete actions must be the same for all '
                'dimensions and have at least 1 action. Projection '
                'Network requires num_actions to be equal across '
                'action dimensions. Implement a more general '
                'categorical projection if you need more flexibility.')

        output_shape = action_spec.shape + (int(unique_num_actions), )
        self._output_shape = output_shape

        self._projection_layer = layers.FC(
            input_size,
            np.prod(output_shape),
            kernel_init_gain=logits_init_output_factor)

    def forward(self, inputs, state=()):
        inputs, state = Network.forward(self, inputs, state)
        logits = self._projection_layer(inputs)
        logits = logits.reshape(inputs.shape[0], *self._output_shape)
        return td.Independent(
            td.Categorical(logits=logits),
            reinterpreted_batch_ndims=len(self._output_shape) - 1), state


@gin.configurable
class NormalProjectionNetwork(DistributionNetwork):
    def __init__(self,
                 input_size,
                 action_spec,
                 activation=layers.identity,
                 projection_output_init_gain=0.1,
                 std_bias_initializer_value=0.0,
                 squash_mean=True,
                 state_dependent_std=False,
                 std_transform=nn.functional.softplus,
                 scale_distribution=False,
                 name="NormalProjectionNetwork"):
        """Creates an instance of NormalProjectionNetwork.

        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            activation (torch.nn.Functional): activation function to use in
                dense layers.
            projection_output_init_gain (float): Output gain for initializing
                action means and std weights.
            std_bias_initializer_value (float): Initial value for the bias of the
                `std_projection_layer`.
            squash_mean (bool): If True, squash the output mean to fit the
                action spec. If `scale_distribution` is also True, this value
                will be ignored.
            state_dependent_std (bool): If True, std will be generated depending
                on the current state; otherwise a global std will be generated
                regardless of the current state.
            std_transform (Callable): Transform to apply to the std, on top of
                `activation`.
            scale_distribution (bool): Whether or not to scale the output
                distribution to ensure that the output aciton fits within the
                `action_spec`. Note that this is different from `mean_transform`
                which merely squashes the mean to fit within the spec.
            name (str):
        """
        super(NormalProjectionNetwork, self).__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(action_spec, TensorSpec)
        assert len(action_spec.shape) == 1, "Only support 1D action spec!"

        self._action_spec = action_spec
        self._mean_transform = layers.identity
        self._scale_distribution = scale_distribution

        if squash_mean or scale_distribution:
            assert isinstance(action_spec, BoundedTensorSpec), \
                ("When squashing the mean or scaling the distribution, bounds "
                 + "are required for the action spec!")

            action_high = torch.as_tensor(action_spec.maximum)
            action_low = torch.as_tensor(action_spec.minimum)
            self._action_means = (action_high + action_low) / 2
            self._action_magnitudes = (action_high - action_low) / 2
            # Do not transform mean if scaling distribution
            if not scale_distribution:
                self._mean_transform = (
                    lambda inputs: self._action_means + self._action_magnitudes
                    * inputs.tanh())

        self._std_transform = layers.identity
        if std_transform is not None:
            self._std_transform = std_transform

        self._means_projection_layer = layers.FC(
            input_size,
            action_spec.shape[0],
            activation=activation,
            kernel_init_gain=projection_output_init_gain)

        if state_dependent_std:
            self._std_projection_layer = layers.FC(
                input_size,
                action_spec.shape[0],
                activation=activation,
                kernel_init_gain=projection_output_init_gain,
                bias_init_value=std_bias_initializer_value)
        else:
            self._std = nn.Parameter(
                action_spec.constant(std_bias_initializer_value),
                requires_grad=True)
            self._std_projection_layer = lambda _: self._std

    def _normal_dist(self, means, stds):
        normal_dist = DiagMultivariateNormal(loc=means, scale_diag=stds)
        if self._scale_distribution:
            # The transformed distribution can also do reparameterized sampling
            # i.e., `.has_rsample=True`
            # Note that in some cases kl_divergence might no longer work for this
            # distribution! Assuming the same `transforms`, below will work:
            # ````
            # kl_divergence(Independent, Independent)
            #
            # kl_divergence(TransformedDistribution(Independent, transforms),
            #               TransformedDistribution(Independent, transforms))
            # ````
            squashed_dist = td.TransformedDistribution(
                base_distribution=normal_dist,
                transforms=[
                    td.SigmoidTransform(),
                    td.AffineTransform(
                        loc=self._action_means - self._action_magnitudes,
                        scale=2 * self._action_magnitudes)
                ])
            return squashed_dist
        else:
            return normal_dist

    def forward(self, inputs, state=()):
        inputs, state = Network.forward(self, inputs, state)
        means = self._mean_transform(self._means_projection_layer(inputs))
        stds = self._std_transform(self._std_projection_layer(inputs))
        return self._normal_dist(means, stds), state


@gin.configurable
class StableNormalProjectionNetwork(NormalProjectionNetwork):
    """Generates a Multi-variate normal by predicting a mean and std.

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
                 input_size,
                 action_spec,
                 activation=layers.identity,
                 projection_output_init_gain=0.1,
                 squash_mean=True,
                 state_dependent_std=False,
                 inverse_std_transform='softplus',
                 scale_distribution=False,
                 init_std=1.0,
                 min_std=0.0,
                 max_std=None,
                 name="StableNormalProjectionNetwork"):
        """Creates an instance of StableNormalProjectionNetwork.

        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            activation (torch.nn.Functional): activation function to use in
                dense layers.
            projection_output_init_gain (float): Output gain for initializing
                action means and std weights.
            squash_mean (bool): If True, squash the output mean to fit the
                action spec. If `scale_distribution` is also True, this value
                will be ignored.
            state_dependent_std (bool): If True, std will be generated depending
                on the current state; otherwise a global std will be generated
                regardless of the current state.
            inverse_std_transform (torch.nn.functional): Currently supports
                "exp" and "softplus". Transformation to obtain inverse std. The
                transformed values are further transformed according to min_std
                and max_std.
            scale_distribution (bool): Whether or not to scale the output
                distribution to ensure that the output aciton fits within the
                `action_spec`. Note that this is different from `mean_transform`
                which merely squashes the mean to fit within the spec.
            init_std (float): Initial value for standard deviation.
            min_std (float): Minimum value for standard deviation.
            max_std (float): Maximum value for standard deviation. If None, no
                maximum is enforced.
            name (str):
        """
        self._min_std = min_std
        self._max_std = max_std
        assert init_std > min_std

        c = 1 / (init_std - min_std)
        if max_std is not None:
            assert init_std < max_std
            c -= 1 / (max_std - min_std)

        if inverse_std_transform == 'exp':
            std_transform = torch.exp
            std_bias_initializer_value = math.log(c)
        elif inverse_std_transform == 'softplus':
            std_transform = nn.functional.softplus
            std_bias_initializer_value = math.log(math.exp(c) - 1)
        else:
            raise ValueError(
                "Unsupported inverse_std_transform %s" % inverse_std_transform)

        super().__init__(
            input_size=input_size,
            action_spec=action_spec,
            activation=activation,
            projection_output_init_gain=projection_output_init_gain,
            std_bias_initializer_value=std_bias_initializer_value,
            squash_mean=squash_mean,
            state_dependent_std=state_dependent_std,
            std_transform=std_transform,
            scale_distribution=scale_distribution,
            name=name)

    def forward(self, inputs, state=()):
        inputs, state = Network.forward(self, inputs, state)
        inv_stds = self._std_transform(self._std_projection_layer(inputs))
        if self._max_std is not None:
            inv_stds += 1 / (self._max_std - self._min_std)
        stds = 1. / inv_stds
        if self._min_std > 0:
            stds += self._min_std

        means = self._mean_transform(
            self._means_projection_layer(inputs) * stds)

        return self._normal_dist(means, stds), state
