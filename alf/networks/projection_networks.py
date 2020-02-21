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
import numpy as np
import functools
from typing import Callable

import torch
import torch.nn as nn
import torch.distributions as td

import alf.layers as layers
from alf.tensor_specs import TensorSpec, BoundedTensorSpec


def DiagMultivariateNormal(loc, scale_diag):
    return td.Independent(td.Normal(loc, scale_diag), 1)


@gin.configurable
class CategoricalProjectionNetwork(nn.Module):
    def __init__(self, input_size, num_actions, logits_init_output_factor=0.1):
        """Creates a categorical projection network that outputs a discrete
        distribution over a number of classes.

        Args:
            input_size (int): the input vector size
            num_actions (int): number of actions
        """
        super(CategoricalProjectionNetwork, self).__init__()

        self._projection_layer = layers.FC(
            input_size,
            num_actions,
            kernel_init_gain=logits_init_output_factor)

    def forward(self, inputs):
        logits = self._projection_layer(inputs)
        return torch.distributions.Categorical(logits=logits)


@gin.configurable
class NormalProjectionNetwork(nn.Module):
    def __init__(self,
                 input_size,
                 action_spec,
                 activation=layers.identity,
                 projection_output_init_gain=0.1,
                 std_bias_initializer_value=0.0,
                 squash_mean=True,
                 state_dependent_std=False,
                 std_transform=nn.functional.softplus,
                 scale_distribution=False):
        """Creates an instance of NormalProjectionNetwork.

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
        """
        super(NormalProjectionNetwork, self).__init__()

        assert isinstance(action_spec, TensorSpec)
        assert len(action_spec.shape) == 1, "Only support 1D action spec!"

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

    def forward(self, inputs):
        means = self._mean_transform(self._means_projection_layer(inputs))
        stds = self._std_transform(self._std_projection_layer(inputs))
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
