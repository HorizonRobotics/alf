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

import abc
import gin

import torch
import torch.nn as nn

from . import nest
from alf.tensor_specs import TensorSpec


class NestCombiner(abc.ABC):
    """A base class for combining all elements in a nested structure."""

    def __init__(self, name):
        self._name = name

    @abc.abstractmethod
    def _combine_flat(self, tensors):
        """Given a list of tensors flattened from the nest, this function defines
        the combining method.

        Args:
            tensors (list[Tensor]): a flat list of tensors

        Returns:
            tensor (Tensor): the combined result
        """
        pass

    def __call__(self, nested):
        """Combine all elements according to the method defined in `combine_flat`.

        Args:
            nested (nest): a nested structure; each element can be either a
                `Tensor` or a `TensorSpec`.

        Returns:
            Tensor or TensorSpec: if `Tensor`, the returned is the concatenated
                result; otherwise it's the tensor spec of the result.
        """
        flat = nest.flatten(nested)
        assert len(flat) > 0, "The nest is empty!"
        if isinstance(flat[0], TensorSpec):
            tensors = nest.map_structure(
                lambda spec: spec.zeros(outer_dims=(1, )), flat)
        else:
            tensors = flat
        ret = self._combine_flat(tensors)
        if isinstance(flat[0], TensorSpec):
            return TensorSpec.from_tensor(ret, from_dim=1)
        return ret


@gin.configurable
class NestConcat(NestCombiner):
    def __init__(self, dim=-1, name="NestConcat"):
        """A combiner for concatenating all elements in a nest along the specified
        axis. It assumes that all elements have the same tensor spec. Can be used
        as a preprocessing combiner in `EncodingNetwork`.

        Args:
            dim (int): the dim along which the elements are concatenated
            name (str):
        """
        super(NestConcat, self).__init__(name)
        self._dim = dim

    def _combine_flat(self, tensors):
        return torch.cat(tensors, dim=self._dim)


@gin.configurable
class NestSum(NestCombiner):
    def __init__(self, average=False, name="NestSum"):
        """Add all elements in a nest together. It assumes that all elements have
        the same tensor shape. Can be used as a preprocessing combiner in
        `EncodingNetwork`.

        Args:
            average (bool): If True, the elements are averaged instead of summed.
            name (str):
        """
        super(NestSum, self).__init__(name)
        self._average = average

    def _combine_flat(self, tensors):
        ret = sum(tensors)
        if self._average:
            ret *= 1 / float(len(tensors))
        return ret


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
    return nest.map_structure(lambda *tensors: torch.stack(tensors), *nests)


def get_outer_rank(tensors, specs):
    """Compares tensors to specs to determine the number of batch dimensions.

    For each tensor, it checks the dimensions with respect to specs and
    returns the number of batch dimensions if all nested tensors and
    specs agree with each other.

    Args:
        tensors (nested Tensors): Nested list/tuple/dict of Tensors.
        specs (nested TensorSpecs): Nested list/tuple/dict of TensorSpecs,
            describing the shape of unbatched tensors.
    Returns:
        The number of outer dimensions for all Tensors (zero if all are
        unbatched or empty).
    Raises:
        AssertionError: If
        1. The shape of Tensors are not compatible with specs, or
        2. A mix of batched and unbatched tensors are provided.
        3. The tensors are batched but have an incorrect number of outer dims.
    """

    def _get_outer_rank(tensor, spec):
        outer_rank = len(tensor.shape) - len(spec.shape)
        assert outer_rank >= 0
        assert list(tensor.shape[outer_rank:]) == list(spec.shape)
        return outer_rank

    outer_ranks = nest.map_structure(_get_outer_rank, tensors, specs)
    outer_ranks = nest.flatten(outer_ranks)
    outer_rank = outer_ranks[0]
    assert all([r == outer_rank
                for r in outer_ranks]), ("Tensors have different "
                                         "outer_ranks %s" % outer_ranks)
    return outer_rank
