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
"""Collection of spec utility functions."""

import numpy as np
import torch
from typing import Iterable

import alf.nest as nest
from alf.nest.utils import get_outer_rank
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from . import dist_utils


class BatchSquash(object):
    """Facilitates flattening and unflattening batch dims of a tensor. Copied
    from `tf_agents`.

    Exposes a pair of matched flatten and unflatten methods. After flattening
    only 1 batch dimension will be left. This facilitates evaluating networks
    that expect inputs to have only 1 batch dimension.
    """

    def __init__(self, batch_dims):
        """Create two tied ops to flatten and unflatten the front dimensions.

        Args:
            batch_dims (int): Number of batch dimensions the flatten/unflatten
                ops should handle.

        Raises:
            ValueError: if batch dims is negative.
        """
        if batch_dims < 0:
            raise ValueError('Batch dims must be non-negative.')
        self._batch_dims = batch_dims
        self._original_tensor_shape = None

    def flatten(self, tensor):
        """Flattens and caches the tensor's batch_dims."""
        if self._batch_dims == 1:
            return tensor
        self._original_tensor_shape = tensor.shape
        return torch.reshape(tensor,
                             (-1, ) + tuple(tensor.shape[self._batch_dims:]))

    def unflatten(self, tensor):
        """Unflattens the tensor's batch_dims using the cached shape."""
        if self._batch_dims == 1:
            return tensor

        if self._original_tensor_shape is None:
            raise ValueError('Please call flatten before unflatten.')

        return torch.reshape(
            tensor, (tuple(self._original_tensor_shape[:self._batch_dims]) +
                     tuple(tensor.shape[1:])))


def spec_means_and_magnitudes(spec: BoundedTensorSpec):
    """Get the center and magnitude of the ranges for the input spec.

    Args:
        spec (BoundedTensorSpec): the spec used to compute mean and magnitudes.

    Returns:
        spec_means (Tensor): the mean value of the spec bound.
        spec_magnitudes (Tensor): the magnitude of the spec bound.
    """

    spec_means = (spec.maximum + spec.minimum) / 2.0
    spec_magnitudes = (spec.maximum - spec.minimum) / 2.0
    return torch.as_tensor(spec_means).to(
        spec.dtype), torch.as_tensor(spec_magnitudes).to(spec.dtype)


def scale_to_spec(tensor, spec: BoundedTensorSpec):
    """Shapes and scales a batch into the given spec bounds.

    Args:
        tensor: A tensor with values in the range of [-1, 1].
        spec: (BoundedTensorSpec) to use for scaling the input tensor.

    Returns:
        A batch scaled the given spec bounds.
    """
    bs = BatchSquash(get_outer_rank(tensor, spec))
    tensor = bs.flatten(tensor)
    means, magnitudes = spec_means_and_magnitudes(spec)
    tensor = means + magnitudes * tensor
    tensor = bs.unflatten(tensor)
    return tensor


def clip_to_spec(value, spec: BoundedTensorSpec):
    """Clips value to a given bounded tensor spec.
    Args:
        value: (tensor) value to be clipped.
        spec: (BoundedTensorSpec) spec containing min and max values for clipping.
    Returns:
        clipped_value: (tensor) `value` clipped to be compatible with `spec`.
    """
    return torch.max(
        torch.min(value, torch.as_tensor(spec.maximum)),
        torch.as_tensor(spec.minimum))


def zeros_from_spec(nested_spec, batch_size):
    """Create nested zero Tensors or Distributions.

    A zero tensor with shape[0]=`batch_size is created for each TensorSpec and
    A distribution with all the parameters as zero Tensors is created for each
    DistributionSpec.

    Args:
        nested_spec (nested TensorSpec or DistributionSpec):
        batch_size (int|tuple|list): batch size/shape added as the first dimension to the shapes
             in TensorSpec
    Returns:
        nested Tensor or Distribution
    """
    if isinstance(batch_size, Iterable):
        shape = batch_size
    else:
        shape = [batch_size]

    def _zero_tensor(spec):
        return spec.zeros(shape)

    param_spec = dist_utils.to_distribution_param_spec(nested_spec)
    params = nest.map_structure(_zero_tensor, param_spec)
    return dist_utils.params_to_distributions(params, nested_spec)


def is_same_spec(spec1, spec2):
    """Whether two nested specs are same.

    Args:
        spec1 (nested TensorSpec): the first spec
        spec2 (nested TensorSpec): the second spec
    Returns:
        bool
    """
    try:
        nest.assert_same_structure(spec1, spec2)
    except:
        return False
    same = nest.map_structure(lambda s1, s2: s1 == s2, spec1, spec2)
    return all(nest.flatten(same))
