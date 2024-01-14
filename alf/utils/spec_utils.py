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
import sys
import torch
from typing import Iterable

import alf.nest as nest
from alf.nest.utils import get_outer_rank
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from . import dist_utils
from alf.utils.tensor_utils import BatchSquash


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


def consistent_with_spec(nested, spec, from_dim=0):
    """Check whether the nested structure is consistent with the spec.

    Besides return the bool value, it will also print out the mismatching
    information.

    Args:
        nested (nested Tensor): the nested structure to be checked
        spec (nested TensorSpec): the spec
    Returns:
        bool: True if the nested structure is consistent with the spec
    """
    nested_paths = nest.flatten(
        nest.py_map_structure_with_path(lambda path, x: path, nested))
    spec_paths = nest.flatten(
        nest.py_map_structure_with_path(lambda path, x: path, spec))
    if set(nested_paths) != set(spec_paths):
        print("The nest does not match the spec!", file=sys.stderr)
        not_in_spec = set(nested_paths) - set(spec_paths)
        not_in_obs = set(spec_paths) - set(nested_paths)
        if not_in_spec:
            print("Not in spec: ", not_in_spec, file=sys.stderr)
        if not_in_obs:
            print("Not in nested: ", not_in_obs, file=sys.stderr)
        return False

    def _check_spec(path, x, s):
        if not (len(x.shape) - from_dim == len(s.shape)
                and x.shape[from_dim:] == s.shape
                and x.dtype == np.dtype(s.dtype_str)):
            print(
                f"Spec mismatch at path: {path}, "
                f"tensor shape={x.shape} tensor dtype={x.dtype} "
                f"spec shape={s.shape} spec dtype={s.dtype}",
                file=sys.stderr)
            return False
        else:
            return True

    ok = nest.py_map_structure_with_path(_check_spec, nested, spec)
    return all(nest.flatten(ok))
