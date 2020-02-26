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
"""Some nest utils functions."""

import gin
import torch

from . import nest


def stack_nests(nests):
    """Stack tensors to a sequence.

    All the nest should have same structure and shape. In the resulted nest,
    each tensor has shape of [T,...] and is the concat of all the corresponding
    tensors in nests
    Args:
        nests (list[nest]): list of nests with same structure and shape
    Returns:
        a nest with same structure as nests[0]
    """

    def _stack(*tensors):
        return torch.cat([t.unsqueeze(0) for t in tensors])

    return nest.map_structure(_stack, *nests)


@gin.configurable
def nest_concatenate(nested, dim=-1):
    """Concatenate all elements in a nest along the specified axis. It assumes
    that all elements have the same tensor shape. Can be used as a preprocessing
    combiner in `EncodingNetwork`.

    Args:
        nested (nest): a nested structure
        dim (int): the dim along which the elements are concatenated

    Returns:
        tensor (torch.Tensor): the concat result
    """
    return torch.cat(nest.flatten(nested), dim=dim)


def nest_reduce_sum(nested):
    """Add all elements in a nest together. It assumes that all elements have
    the same tensor shape. Can be used as a preprocessing combiner in
    `EncodingNetwork`.

    Args:
        nested (nest): a nested structure

    Returns:
        tensor (torch.Tensor):
    """
    return torch.sum(torch.stack(nest.flatten(nested), dim=0), dim=0)
