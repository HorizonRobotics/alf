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

import alf
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
        """Combine all elements according to the method defined in
        ``combine_flat``.

        Args:
            nested (nest): a nested structure; each element can be either a
                ``Tensor` or a `TensorSpec``.

        Returns:
            Tensor or TensorSpec: if ``Tensor``, the returned is the concatenated
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
        """A combiner for concatenating all tensors in a nest along the specified
        axis. It assumes that all tensors have the same tensor spec. Can be used
        as a preprocessing combiner in ``EncodingNetwork``.

        Args:
            dim (int): the dim along which the tensors are concatenated
            name (str):
        """
        super(NestConcat, self).__init__(name)
        self._dim = dim

    def _combine_flat(self, tensors):
        return torch.cat(tensors, dim=self._dim)


@gin.configurable
class NestSum(NestCombiner):
    def __init__(self, average=False, activation=None, name="NestSum"):
        """Add all tensors in a nest together. It assumes that all tensors have
        the same tensor shape. Can be used as a preprocessing combiner in
        ``EncodingNetwork``.

        Args:
            average (bool): If True, the tensors are averaged instead of summed.
            activation (Callable): activation function.
            name (str):
        """
        super(NestSum, self).__init__(name)
        self._average = average
        if activation is None:
            activation = lambda x: x
        self._activation = activation

    def _combine_flat(self, tensors):
        ret = sum(tensors)
        if self._average:
            ret *= 1 / float(len(tensors))
        return self._activation(ret)


@gin.configurable
class NestMultiply(NestCombiner):
    def __init__(self, activation=None, name="NestMultiply"):
        """Element-wise multiply all tensors in a nest. It assumes that all
        tensors have the same shape. Can be used as a preprocessing combiner in
        ``EncodingNetwork``.

        Args:
            activation (Callable): optional activation function applied after
                the multiplication.
            name (str):
        """
        super(NestMultiply, self).__init__(name)
        if activation is None:
            activation = lambda x: x
        self._activation = activation

    def _combine_flat(self, tensors):
        ret = alf.utils.math_ops.mul_n(tensors)
        return self._activation(ret)


def stack_nests(nests):
    """Stack tensors to a sequence.

    All the nest should have same structure and shape. In the resulted nest,
    each tensor has shape of :math:`[T,...]` and is the concat of all the
    corresponding tensors in nests.

    Args:
        nests (list[nest]): list of nests with same structure and shape.
    Returns:
        a nest with same structure as ``nests[0]``.
    """
    if len(nests) == 1:
        return nest.map_structure(lambda tensor: tensor.unsqueeze(0), nests[0])
    else:
        return nest.map_structure(lambda *tensors: torch.stack(tensors),
                                  *nests)


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
        int: The number of outer dimensions for all tensors (zero if all are
        unbatched or empty).

    Raises:
        AssertionError: If the shape of Tensors are not compatible with specs,
            a mix of batched and unbatched tensors are provided, or the tensors
            are batched but have an incorrect number of outer dims.
    """

    outer_ranks = []

    def _get_outer_rank(tensor, spec):
        outer_rank = len(tensor.shape) - len(spec.shape)
        assert outer_rank >= 0
        assert tensor.shape[outer_rank:] == spec.shape
        outer_ranks.append(outer_rank)

    nest.map_structure(_get_outer_rank, tensors, specs)
    outer_rank = outer_ranks[0]
    assert all([r == outer_rank
                for r in outer_ranks]), ("Tensors have different "
                                         "outer_ranks %s" % outer_ranks)
    return outer_rank


def transform_nest(nested, field, func):
    """Transform the node of a nested structure indicated by ``field`` using
    ``func``.

    This function can be used to update our ``namedtuple`` structure conveniently,
    comparing the following two methods:

        .. code-block:: python

            info = info._replace(rl=info.rl._replace(sac=info.rl.sac * 0.5))

    vs.

        .. code-block:: python

            info = transform_nest(info, 'rl.sac', lambda x: x * 0.5)

    The second method is usually shorter, more intuitive, and less error-prone
    when ``field`` is a long string.

    Args:
        nested (nested Tensor): the structure to be applied the transformation.
        field (str): If a string, it's the field to be transformed, multi-level
            path denoted by "A.B.C". If ``None``, then the root object is
            transformed.
        func (Callable): transform func, the function will be called as
            ``func(nested)`` and should return a new nest.
    Returns:
        transformed nest
    """

    def _traverse_transform(nested, levels):
        if not levels:
            return func(nested)
        level = levels[0]
        if nest.is_namedtuple(nested):
            new_val = _traverse_transform(
                nested=getattr(nested, level), levels=levels[1:])
            return nested._replace(**{level: new_val})
        elif isinstance(nested, dict):
            new_val = nested.copy()
            _val = _traverse_transform(nested=nested[level], levels=levels[1:])
            if _val is not None:
                new_val[level] = _val
            else:
                del new_val[level]
            return new_val
        else:
            raise TypeError("If value is a nest, it must be either " +
                            "a dict or namedtuple!")

    return _traverse_transform(
        nested=nested, levels=field.split('.') if field else [])


def convert_device(nests):
    """Convert the device of the tensors in nests to default device."""

    def _convert_cuda(tensor):
        if tensor.device.type != 'cuda':
            return tensor.cuda()
        else:
            return tensor

    def _convert_cpu(tensor):
        if tensor.device.type != 'cpu':
            return tensor.cpu()
        else:
            return tensor

    d = alf.get_default_device()
    if d == 'cpu':
        return nest.map_structure(_convert_cpu, nests)
    elif d == 'cuda':
        return nest.map_structure(_convert_cuda, nests)
    else:
        raise NotImplementedError("Unknown device %s" % d)


def grad(nested, objective):
    """Compute the gradients of an ``objective`` `w.r.t` each variable in
    ``nested``. It will simply call ``torch.autograd.grad`` after flattening the
    nest, and then pack the flat list back to a structure like ``nested``.

    Args:
        nested (nest): a nest of variables that require grads.
        objective (Tensor): a tensor whose gradients will be computed.
    """
    return nest.pack_sequence_as(
        nested, list(torch.autograd.grad(objective, nest.flatten(nested))))


def zeros_like(nested):
    """Create a new nest with all zeros like the reference ``nested``.

    Args:
        nested (nested Tensor): a nested structure

    Returns:
        nested Tensor: a nest with all zeros
    """
    return nest.map_structure(torch.zeros_like, nested)
