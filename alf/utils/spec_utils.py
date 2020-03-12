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
"""Collection of spec utility functions."""

import numpy as np
import torch

from alf.layers import BatchSquash
import alf.nest as nest
from alf.nest.utils import get_outer_rank
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import tensor_utils


def spec_means_and_magnitudes(spec: BoundedTensorSpec):
    """Get the center and magnitudes of the ranges for the input spec.

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
    tensor = tensor_utils.scale_and_shift(
        tensor, means=means, magnitudes=magnitudes)
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
