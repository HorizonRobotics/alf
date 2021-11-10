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

from functools import partial
import math
import numpy as np
from typing import Callable

import torch
import torch.nn as nn
import torch.distributions as td

import alf
import alf.layers as layers
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.networks.network import Network
from alf.utils import dist_utils
import alf.utils.math_ops as math_ops


@alf.configurable
class CategoricalProjectionNetwork(Network):
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
        logits = self._projection_layer(inputs)
        logits = logits.reshape(inputs.shape[0], *self._output_shape)
        if len(self._output_shape) > 1:
            return td.Independent(
                td.Categorical(logits=logits),
                reinterpreted_batch_ndims=len(self._output_shape) - 1), state
        else:
            return td.Categorical(logits=logits), state

    def make_parallel(self, n):
        """Creates a ``ParallelCategoricalProjectionNetwork`` using ``n``
        replicas of ``self``. The initialized layer parameters will
        be different.
        """
        parallel_proj_net_args = dict(**self.saved_args)
        parallel_proj_net_args.update(n=n, name="parallel_" + self.name)
        return ParallelCategoricalProjectionNetwork(**parallel_proj_net_args)


@alf.configurable
class ParallelCategoricalProjectionNetwork(Network):
    def __init__(self,
                 input_size,
                 action_spec,
                 n,
                 logits_init_output_factor=0.1,
                 name="ParallelCategoricalProjectionNetwork"):
        """Creates an instance of ParallelCategoricalProjectionNetwork.


        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            n (int): number of the parallel networks
            name (str): name of this network.
        """
        super(ParallelCategoricalProjectionNetwork, self).__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(action_spec, TensorSpec)

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

        self._projection_layer = layers.ParallelFC(
            input_size,
            np.prod(output_shape),
            n,
            kernel_init_gain=logits_init_output_factor)
        self._n = n

    def forward(self, inputs, state=()):
        logits = self._projection_layer(inputs)
        logits = logits.reshape(inputs.shape[0], self._n, *self._output_shape)
        if len(self._output_shape) > 1:
            return td.Independent(
                td.Categorical(logits=logits),
                reinterpreted_batch_ndims=len(self._output_shape) - 1), state
        else:
            return td.Categorical(logits=logits), state


@alf.configurable
class NormalProjectionNetwork(Network):
    def __init__(self,
                 input_size,
                 action_spec,
                 activation=math_ops.identity,
                 projection_output_init_gain=0.3,
                 std_bias_initializer_value=0.0,
                 squash_mean=True,
                 state_dependent_std=False,
                 std_transform=nn.functional.softplus,
                 scale_distribution=False,
                 dist_squashing_transform=dist_utils.StableTanh(),
                 name="NormalProjectionNetwork"):
        """Creates an instance of NormalProjectionNetwork.

        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            activation (Callable): activation function to use in
                dense layers.
            projection_output_init_gain (float): Output gain for initializing
                action means and std weights.
            std_bias_initializer_value (float): Initial value for the bias of the
                ``std_projection_layer``.
            squash_mean (bool): If True, squash the output mean to fit the
                action spec. If ``scale_distribution`` is also True, this value
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
            dist_squashing_transform (td.Transform):  A distribution Transform
                which transforms values into :math:`(-1, 1)`. Default to ``dist_utils.StableTanh()``
            name (str): name of this network.
        """
        super(NormalProjectionNetwork, self).__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(action_spec, TensorSpec)
        assert len(action_spec.shape) == 1, "Only support 1D action spec!"

        self._action_spec = action_spec
        self._mean_transform = math_ops.identity
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
            else:
                self._transforms = [
                    dist_squashing_transform,
                    dist_utils.AffineTransform(
                        loc=self._action_means, scale=self._action_magnitudes)
                ]

        self._std_transform = math_ops.identity
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
        normal_dist = dist_utils.DiagMultivariateNormal(loc=means, scale=stds)
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
                base_distribution=normal_dist, transforms=self._transforms)
            return squashed_dist
        else:
            return normal_dist

    def forward(self, inputs, state=()):
        means = self._mean_transform(self._means_projection_layer(inputs))
        stds = self._std_transform(self._std_projection_layer(inputs))
        return self._normal_dist(means, stds), state

    def make_parallel(self, n):
        """Create a ``ParallelNormalProjectionNetwork`` using ``n``
        replicas of ``self``. The initialized layer parameters will
        be different.
        """
        parallel_proj_net_args = dict(**self.saved_args)
        parallel_proj_net_args.update(n=n, name="parallel_" + self.name)
        return ParallelNormalProjectionNetwork(**parallel_proj_net_args)


@alf.configurable
class ParallelNormalProjectionNetwork(Network):
    def __init__(self,
                 input_size,
                 action_spec,
                 n,
                 activation=math_ops.identity,
                 projection_output_init_gain=0.3,
                 std_bias_initializer_value=0.0,
                 squash_mean=True,
                 state_dependent_std=False,
                 std_transform=nn.functional.softplus,
                 scale_distribution=False,
                 dist_squashing_transform=dist_utils.StableTanh(),
                 name="ParallelNormalProjectionNetwork"):
        """Creates an instance of ParallelNormalProjectionNetwork.


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
            dist_squashing_transform (td.Transform):  A distribution Transform
                which transform values to fall in (-1, 1). Default to `dist_utils.StableTanh()`
            name (str): name of this network.
        """
        super(ParallelNormalProjectionNetwork, self).__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(action_spec, TensorSpec)
        assert len(action_spec.shape) == 1, "Only support 1D action spec!"

        self._action_spec = action_spec
        self._mean_transform = math_ops.identity
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
            else:
                self._transforms = [
                    dist_squashing_transform,
                    dist_utils.AffineTransform(
                        loc=self._action_means, scale=self._action_magnitudes)
                ]

        self._std_transform = math_ops.identity
        if std_transform is not None:
            self._std_transform = std_transform

        self._means_projection_layer = layers.ParallelFC(
            input_size,
            action_spec.shape[0],
            n,
            activation=activation,
            kernel_init_gain=projection_output_init_gain)

        if state_dependent_std:
            self._std_projection_layer = layers.ParallelFC(
                input_size,
                action_spec.shape[0],
                n,
                activation=activation,
                kernel_init_gain=projection_output_init_gain,
                bias_init_value=std_bias_initializer_value)
        else:
            self._std = nn.Parameter(
                action_spec.constant(
                    std_bias_initializer_value, outer_dims=(n, )),
                requires_grad=True)
            self._std_projection_layer = lambda _: self._std

    def _normal_dist(self, means, stds):
        normal_dist = dist_utils.DiagMultivariateNormal(loc=means, scale=stds)
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
                base_distribution=normal_dist, transforms=self._transforms)
            return squashed_dist
        else:
            return normal_dist

    def forward(self, inputs, state=()):
        means = self._mean_transform(self._means_projection_layer(inputs))
        stds = self._std_transform(self._std_projection_layer(inputs))
        return self._normal_dist(means, stds), state


@alf.configurable
class StableNormalProjectionNetwork(NormalProjectionNetwork):
    r"""Generates a Multi-variate normal by predicting a mean and std.

    It parameterizes the normal distributions as :math:`\sigma=c_0+\frac{1}{c_1+softplus(b)}`
    and :math:`\mu=a\cdot\sigma` where a and b are outputs from means_projection_layer and
    stds_projectin_layer respectively. :math:`c_0` and :math:`c_1` are chosen so that
    :math:`\sigma_{min} <= \sigma <= \sigma_{max}`. The advantage of this parameterization is that
    its second order derivatives with respect to a and b are bounded even when
    the standard deviations become very small so that the optimization is
    more stable. See ``docs/stable_gradient_descent_for_gaussian_distribution.py``
    for detail.
    """

    def __init__(self,
                 input_size,
                 action_spec,
                 activation=math_ops.identity,
                 projection_output_init_gain=1e-5,
                 squash_mean=True,
                 state_dependent_std=False,
                 inverse_std_transform='softplus',
                 scale_distribution=False,
                 init_std=1.0,
                 min_std=0.0,
                 max_std=None,
                 dist_squashing_transform=dist_utils.StableTanh(),
                 name="StableNormalProjectionNetwork"):
        """Creates an instance of StableNormalProjectionNetwork.

        Currently there seems no need for this class to handle nested inputs;
        If necessary, extend the argument list to support it in the future.

        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            activation (Callable): activation function to use in
                dense layers.
            projection_output_init_gain (float): Output gain for initializing
                action means and std weights.
            squash_mean (bool): If True, squash the output mean to fit the
                action spec. If `scale_distribution` is also True, this value
                will be ignored.
            state_dependent_std (bool): If True, std will be generated depending
                on the current state; otherwise a global std will be generated
                regardless of the current state.
            inverse_std_transform (str): Currently supports
                "exp" and "softplus". Transformation to obtain inverse std. The
                transformed values are further transformed according to min_std
                and max_std.
            scale_distribution (bool): Whether or not to scale the output
                distribution to ensure that the output aciton fits within the
                `action_spec`. Note that this is different from 'mean_transform'
                which merely squashes the mean to fit within the spec.
            init_std (float): Initial value for standard deviation.
            min_std (float): Minimum value for standard deviation.
            max_std (float): Maximum value for standard deviation. If None, no
                maximum is enforced.
            dist_squashing_transform (td.Transform):  A distribution Transform
                which transforms values into :math:`(-1, 1)`. Default to ``dist_utils.StableTanh()``
            name (str): name of this network.
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
            dist_squashing_transform=dist_squashing_transform,
            name=name)

    def forward(self, inputs, state=()):
        inv_stds = self._std_transform(self._std_projection_layer(inputs))
        if self._max_std is not None:
            inv_stds = inv_stds + 1 / (self._max_std - self._min_std)
        stds = 1. / inv_stds
        if self._min_std > 0:
            stds = stds + self._min_std

        means = self._mean_transform(
            self._means_projection_layer(inputs) * stds)

        return self._normal_dist(means, stds), state


@alf.configurable
class CauchyProjectionNetwork(NormalProjectionNetwork):
    def __init__(self,
                 input_size,
                 action_spec,
                 squash_median=True,
                 scale_bias_initializer_value=0.0,
                 state_dependent_scale=False,
                 scale_transform=nn.functional.softplus,
                 scale_distribution=False,
                 dist_squashing_transform=dist_utils.StableTanh(),
                 name='CauchyProjectionNetwork'):
        """Similar to ``NormalProjectionNetwork`` except that the output
        distribution is a ``DiagMultivariateCauchy``. Also since Cauchy doesn't
        have mean or std, we provide parameters for its median and scale instead.
        But the median and scale will just reuse the code for handling mean and std
        in ``NormalProjectionNetwork``.

        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            squash_median (bool): If True, squash the output median to fit the
                action spec. If ``scale_distribution`` is also True, this value
                will be ignored.
            scale_bias_initializer_value (float): Initial value for the bias of the
                scale projection layer.
            state_dependent_scale (bool): If True, scale will be generated depending
                on the current state; otherwise a global scale will be generated
                regardless of the current state.
            scale_transform (Callable): Transform to apply to the scale, on top of
                `activation`.
            scale_distribution (bool): Whether or not to scale the output
                distribution to ensure that the output aciton fits within the
                `action_spec`. Note that this is different from `mean_transform`
                which merely squashes the mean to fit within the spec.
            dist_squashing_transform (td.Transform):  A distribution Transform
                which transform values to fall in (-1, 1). Default to `dist_utils.StableTanh()`
            name (str): name of this network.
        """
        super(CauchyProjectionNetwork, self).__init__(
            input_size=input_size,
            action_spec=action_spec,
            squash_mean=squash_median,
            std_bias_initializer_value=scale_bias_initializer_value,
            state_dependent_std=state_dependent_scale,
            std_transform=scale_transform,
            scale_distribution=scale_distribution,
            dist_squashing_transform=dist_squashing_transform,
            name=name)

    def forward(self, inputs, state=()):
        median = self._mean_transform(self._means_projection_layer(inputs))
        scale = self._std_transform(self._std_projection_layer(inputs))
        return self._cauchy_dist(median, scale), state

    def _cauchy_dist(self, median, scale):
        cauchy_dist = dist_utils.DiagMultivariateCauchy(
            loc=median, scale=scale)
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
                base_distribution=cauchy_dist, transforms=self._transforms)
            return squashed_dist
        else:
            return cauchy_dist


def _get_transformer(action_spec):
    """Transform from [0,1] to [action_low, action_high]."""
    action_high = torch.as_tensor(action_spec.maximum)
    action_low = torch.as_tensor(action_spec.minimum)
    if ((action_low == 0) & (action_high == 1)).all():
        return lambda x: x
    else:
        return lambda x: dist_utils.AffineTransformedDistribution(
            base_dist=x, loc=action_low, scale=action_high - action_low)


@alf.configurable
class BetaProjectionNetwork(Network):
    """Beta projection network.

    Its output is a distribution with independent beta distribution for each
    action dimension. Since the support of beta distribution is [0, 1], we also
    apply an affine transformation so the support fill the range specified by
    ``action_spec``.
    """

    def __init__(self,
                 input_size,
                 action_spec,
                 activation=nn.functional.softplus,
                 min_concentration=0.,
                 projection_output_init_gain=0.0,
                 bias_init_value=0.541324854612918,
                 grad_clip=0.01,
                 name="BetaProjectionNetwork"):
        """
        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            activation (Callable): activation function to use in
                dense layers.
            bias_init_value (float): the default value is chosen so that, for softplus
                activation, the initial concentration will be close 1, which
                corresponds to uniform distribution.
            grad_clip (float): if provided, the L2-norm of the gradient of concentration
                will be clipped to be no more than ``grad_clip``.
            min_concentration (float): there may be issue of numerical stability
                if the calculated concentration is very close to 0. A positive
                value of this may help to alleviate it.
        """
        super().__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)
        assert action_spec.ndim == 1, "Only support 1D action spec!"

        self._transformer = _get_transformer(action_spec)

        self._concentration_projection_layer = layers.FC(
            input_size,
            2 * action_spec.shape[0],
            activation=activation,
            bias_init_value=bias_init_value,
            kernel_init_gain=projection_output_init_gain)
        self._grad_clip = grad_clip
        self._min_concentration = min_concentration

    def forward(self, inputs, state=()):
        concentration = self._concentration_projection_layer(inputs)
        if self._min_concentration != 0:
            concentration = concentration + self._min_concentration
        if self._grad_clip is not None and inputs.requires_grad:
            concentration.register_hook(lambda x: x / (x.norm(
                dim=1, keepdim=True) * (1 / self._grad_clip)).clamp(1.))
        concentration10 = concentration.split(
            concentration.shape[-1] // 2, dim=-1)
        return self._transformer(
            dist_utils.DiagMultivariateBeta(*concentration10)), state


@alf.configurable
class TruncatedProjectionNetwork(Network):
    def __init__(self,
                 input_size,
                 action_spec,
                 activation=math_ops.identity,
                 projection_output_init_gain=0.3,
                 scale_bias_initializer_value=0.0,
                 state_dependent_scale=False,
                 loc_transform=torch.tanh,
                 scale_transform=nn.functional.softplus,
                 min_scale=None,
                 max_scale=None,
                 dist_ctor=dist_utils.TruncatedNormal,
                 name="TruncatedProjectionNetwork"):
        """Creates an instance of TruncatedProjectionNetwork.

        Its output is a TruncatedDistribution with bounds given by the action
        bounds specified in ``action_spec``.

        Args:
            input_size (int): input vector dimension
            action_spec (TensorSpec): a tensor spec containing the information
                of the output distribution.
            activation (Callable): activation function to use in
                dense layers.
            projection_output_init_gain (float): Output gain for initializing
                action means and std weights.
            std_bias_initializer_value (float): Initial value for the bias of the
                ``std_projection_layer``.
            state_dependent_scale (bool): If True, std will be generated depending
                on the current state (i.e. inputs); otherwise a global scale will
                be generated regardless of the current state.
            loc_transform (Callable): Tranform to apply to the loc, on top of
                `activation` to make it within [-1, 1].
            scale_transform (Callable): Transform to apply to the std, on top of
                `activation` to make it positive.
            min_scale (float): Minimum value for scale. If None, no maximum is
                enforced.
            max_scale (float): Maximum value for scale. If None, no maximum is
                enforced.
            dist_ctor(Callable): constructor for the distribution called as:
                `dist_ctor(loc=loc, scale=scale, lower_bound=lower_bound, upper_bound=upper_bound)`.
            name (str): name of this network.
        """
        super().__init__(
            input_tensor_spec=TensorSpec((input_size, )), name=name)

        assert isinstance(action_spec, TensorSpec)
        assert len(action_spec.shape) == 1, "Only support 1D action spec!"

        self._scale_transform = math_ops.identity
        if scale_transform is not None:
            self._scale_transform = scale_transform

        self._loc_projection_layer = layers.FC(
            input_size,
            action_spec.shape[0],
            activation=activation,
            kernel_init_gain=projection_output_init_gain)

        if state_dependent_scale:
            self._scale_projection_layer = layers.FC(
                input_size,
                action_spec.shape[0],
                activation=activation,
                kernel_init_gain=projection_output_init_gain,
                bias_init_value=scale_bias_initializer_value)
        else:
            self._scale = nn.Parameter(
                action_spec.constant(scale_bias_initializer_value),
                requires_grad=True)
            self._scale_projection_layer = lambda _: self._scale

        self._action_high = torch.as_tensor(action_spec.maximum).broadcast_to(
            action_spec.shape)
        self._action_low = torch.as_tensor(action_spec.minimum).broadcast_to(
            action_spec.shape)
        self._dist_ctor = dist_ctor

        action_means = (self._action_high + self._action_low) / 2
        action_magnitudes = (self._action_high - self._action_low) / 2

        # Although the TruncatedDistribution will ensure the actions are within
        # the bound, we still make sure the loc parameter to be within the bound
        # for better numerical stability
        self._loc_transform = (lambda inputs: action_means + action_magnitudes
                               * loc_transform(inputs))

        self._min_scale = min_scale
        self._max_scale = max_scale

    def forward(self, inputs, state=()):
        loc = self._loc_transform(self._loc_projection_layer(inputs))
        scale = self._scale_transform(self._scale_projection_layer(inputs))
        if self._min_scale is not None or self._max_scale is not None:
            scale = scale.clamp(min=self._min_scale, max=self._max_scale)
        dist = self._dist_ctor(
            loc=loc,
            scale=scale,
            lower_bound=self._action_low,
            upper_bound=self._action_high)

        return dist, state
