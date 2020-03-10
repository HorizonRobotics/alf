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

import alf.nest as nest
from alf.tensor_specs import TensorSpec, BoundedTensorSpec


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
    return torch.tensor(spec_means).to(
        torch.float32), torch.tensor(spec_magnitudes).to(torch.float32)


def scale_to_spec(tensor, spec: BoundedTensorSpec):
    """Shapes and scales a batch into the given spec bounds.

    Args:
        tensor: A [batch x n] tensor with values in the range of [-1, 1].
        spec: (BoundedTensorSpec) to use for scaling the input tensor.

    Returns:
        A batch scaled the given spec bounds.
    """
    means, magnitudes = spec_means_and_magnitudes(spec)
    tensor = means + magnitudes * tensor
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
        torch.min(value, torch.tensor(spec.maximum)),
        torch.tensor(spec.minimum))
