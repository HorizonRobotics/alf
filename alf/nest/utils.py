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
"""Some nest utils functions."""

import abc

import torch
import torch.nn as nn

from functools import reduce
import numpy as np
from typing import Callable

import alf
from . import nest
from .nest import get_field, map_structure
from alf.tensor_specs import TensorSpec


class NestCombiner(abc.ABC, nn.Module):
    """A base class for combining all elements in a nested structure."""

    def __init__(self, name: str, batch_dims: int = 1):
        """
        Args:
            name: name of the combiner
            batch_dims: number of batch dims (default 1). This argument is only
                necessary for combiners that are not batch-dim invariant (combined
                results depending on the definition of batch dims, e.g., outer
                product).
        """
        super().__init__()
        self._name = name
        self._batch_dims = batch_dims

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
                lambda spec: spec.zeros(outer_dims=(1, ) * self._batch_dims),
                flat)
        else:
            tensors = flat
        ret = self._combine_flat(tensors)
        if isinstance(flat[0], TensorSpec):
            return TensorSpec.from_tensor(ret, from_dim=self._batch_dims)
        return ret


@alf.configurable
@alf.repr_wrapper
class NestConcat(NestCombiner):
    def __init__(self, nest_mask=None, dim=-1, name="NestConcat"):
        """A combiner for selecting from the tensors in a nest and then
        concatenating them along a specified axis. If nest_mask is None,
        then all the tensors from the nest will be selected.
        It assumes that all the selected tensors have the same tensor spec.
        Can be used as a preprocessing combiner of a network.

        Note that batch dimension is not considered for concat. This means that
        dim=0 means the first dimension after batch dimension.

        Args:
            nest_mask (nest|None): nest structured mask indicating which of the
                tensors in the nest to be selected or not, indicated by a
                value of True/False (1/0). Note that the structure of the mask
                should be the same as the nest of data to apply this operator on.
                If is None, then all the tensors from the nest will be selected.
            dim (int): the dim along which the tensors are concatenated
            name (str):
        """
        super(NestConcat, self).__init__(name)
        self._nest_mask = nest_mask
        self._flat_mask = nest.flatten(nest_mask) if nest_mask else nest_mask
        self._dim = dim if dim < 0 else dim + 1

    def _combine_flat(self, tensors):
        if self._flat_mask is not None:
            assert len(self._flat_mask) == len(tensors), (
                "incompatible structures "
                "between mask and data nest")
            selected_tensors = []
            for i, mask_value in enumerate(self._flat_mask):
                if mask_value:
                    selected_tensors.append(tensors[i])
            return torch.cat(selected_tensors, dim=self._dim)
        else:
            return torch.cat(tensors, dim=self._dim)

    def make_parallel(self, n):
        """Create a ``NestConcat`` layer to handle parallel batch.

        It is assumed that a parallel batch has shape [B, n, ...] and both the
        batch dimension and replica dimension are not considered for concat.

        Args:
            n (int): the number of replicas.
        Returns:
            a ``NestConcat`` layer to handle parallel batch.
        """
        return NestConcat(self._nest_mask, self._dim, "parallel_" + self._name)


@alf.configurable
class NestSum(NestCombiner):
    def __init__(self, average=False, activation=None, name="NestSum"):
        """Add all tensors in a nest together. It assumes that all tensors have
        the same tensor shape. Can be used as a preprocessing combiner of
        a network.

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

    def make_parallel(self, n):
        return NestSum(self._average, self._activation,
                       "parallel_" + self._name)


@alf.configurable
class NestMultiply(NestCombiner):
    def __init__(self, activation=None, name="NestMultiply"):
        """Element-wise multiply all tensors in a nest. It assumes that all
        tensors have the same shape. Can be used as a preprocessing combiner of
        a network.

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

    def make_parallel(self, n):
        return NestMultiply(self._activation, "parallel_" + self._name)


@alf.configurable
@alf.repr_wrapper
class NestOuterProduct(NestCombiner):
    def __init__(self,
                 activation: Callable = None,
                 batch_dims: int = 1,
                 padding: bool = False,
                 name: str = "NestOuterProduct"):
        """Perform outer-product operations across a nested structure. Can be used
        as a preprocessing combiner of a network.

        Sometimes combining tensors using outer product might be more expressive
        than concatenating, e.g., when one tensor is one-hot. See the discussions in

        ::

            "STOCHASTIC NEURAL NETWORKS FOR HIERARCHICAL REINFORCEMENT LEARNING",
            Florensa, et al., ICLR 2017, https://arxiv.org/pdf/1704.03012.pdf.

        In this implementation, we also support padding 1s to the tensors before
        doing the outer product, essentially combining outer product and
        concatenation together in one combiner.

        .. warning::

            Due to outer product, this combiner might result in a very long
            output vector. Make sure to do the calculation before using it.

        Args:
            activation: optional activation function applied after the outer product.
            batch_dims: number of batch dims. Default to 1. If the total input dim
                is ``N``, then the last ``N-batch_dims`` will be flattened for
                outer product.
            padding: if True, each tensor will be padded by 1 before performing
                outer product. When this flag is enabled, essentially it has
                the effect of concatenation of all tensors in the output tensor.
            name: name of the combiner
        """
        super(NestOuterProduct, self).__init__(name, batch_dims=batch_dims)
        if activation is None:
            activation = alf.layers.identity
        self._activation = activation
        self._padding = padding

    def _combine_flat(self, tensors):
        batch_shape = tensors[0].shape[:self._batch_dims]

        for t in tensors:
            assert batch_shape == t.shape[:self._batch_dims], (
                "Different batch shapes %s vs. %s" %
                (batch_shape, t.shape[:self._batch_dims]))

        B = int(np.prod(batch_shape))

        tensors = [t.reshape(B, -1) for t in tensors]
        if self._padding:
            tensors = [
                torch.cat([t, torch.ones((B, 1), dtype=t.dtype)], dim=1)
                for t in tensors
            ]

        out = reduce(
            lambda x, y: torch.einsum('bn,bm->bnm', x, y).reshape(B, -1),
            tensors)
        out = out.reshape(*batch_shape, -1)
        return self._activation(out)

    def make_parallel(self, n):
        return NestOuterProduct(self._activation, self._batch_dims + 1,
                                self._padding, "parallel_" + self._name)


def stack_nests(nests, dim=0):
    """Stack tensors to a sequence.

    All the nest should have same structure and shape. In the resulted nest,
    each tensor has shape of :math:`[T,...]` and is the concat of all the
    corresponding tensors in nests.

    Args:
        nests (list[nest]): list of nests with same structure and shape.
        dim (int): dimension to insert. Has to be between 0 and the number of
            dimensions of concatenated tensors (inclusive)
    Returns:
        a nest with same structure as ``nests[0]``.
    """
    if len(nests) == 1:
        return nest.map_structure(lambda tensor: tensor.unsqueeze(dim),
                                  nests[0])
    else:
        return nest.map_structure(lambda *tensors: torch.stack(tensors, dim),
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


def convert_device(nests, device=None):
    """Convert the device of the tensors in nests to the specified
        or to the default device.
    Args:
        nests (nested Tensors): Nested list/tuple/dict of Tensors.
        device (None|str): the target device, should either be `cuda` or `cpu`.
            If None, then the default device will be used as the target device.
    Returns:
        nests (nested Tensors): Nested list/tuple/dict of Tensors after device
            conversion.

    Raises:
        NotImplementedError if the target device is not one of
            None, `cpu` or `cuda` when cuda is available, or AssertionError
            if target device is `cuda` but cuda is unavailable.


    """

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

    if device is None:
        d = alf.get_default_device()
    else:
        d = device

    if d == 'cpu':
        return nest.map_structure(_convert_cpu, nests)
    elif d == 'cuda':
        assert torch.cuda.is_available(), "cuda is unavailable"
        return nest.map_structure(_convert_cuda, nests)
    else:
        raise NotImplementedError("Unknown device %s" % d)


def grad(nested, objective, retain_graph=False):
    """Compute the gradients of an ``objective`` `w.r.t` each variable in
    ``nested``. It will simply call ``torch.autograd.grad`` after flattening the
    nest, and then pack the flat list back to a structure like ``nested``.

    Args:
        nested (nest): a nest of variables that require grads.
        objective (Tensor): a tensor whose gradients will be computed.
        retain_graph (bool): if True, after autograd the computational graph
            won't be freed
    """
    return nest.pack_sequence_as(
        nested,
        list(
            torch.autograd.grad(
                objective, nest.flatten(nested), retain_graph=retain_graph)))


def zeros_like(nested):
    """Create a new nest with all zeros like the reference ``nested``.

    Args:
        nested (nested Tensor): a nested structure

    Returns:
        nested Tensor: a nest with all zeros
    """
    return nest.map_structure(torch.zeros_like, nested)


def make_nested_module(nested, ignore_non_module_element=True):
    """Convert a nest of modules to nn.Module using nn.ModuleList or nn.ModuleDict.

    The reason to use this function is that nest of Modules will not be trained
    or checkpointed. We need to use nn.ModuleList or nn.ModuleDict to hold the
    individual modules in the nest.

    Args:
        nested (nested nn.Module): a nest of nn.Module
        ignore_non_module_element (bool): If True, will ignore the non-module element
            and replace them with None. If False, will raise error if there are
            any non-module elements.
    Returns:
        nn.Module
    """
    if isinstance(nested, (tuple, list)):
        module = torch.nn.ModuleList()
        for m in nested:
            module.append(make_nested_module(m))
    elif nest.is_namedtuple(nested) or isinstance(nested, dict):
        module = torch.nn.ModuleDict()
        for field, value in nest.extract_fields_from_nest(nested):
            module[field] = make_nested_module(value)
    else:
        module = nested
        if not ignore_non_module_element:
            assert isinstance(
                nested,
                torch.nn.Module), ("Unsupported type %s" % type(nested))
        elif not isinstance(nested, torch.nn.Module):
            module = None
    return module


def get_nested_field(nested, nest_fields):
    """Get nested fields from a nest.

    Example:

        x = get_nested_field(nest, ('a.b', 'c'))
        y = (get_field(nest, 'a.b')), get_field(nest, 'c'))
        # y and x are same

    Args:
        nested (nest): a nested structure
        nest_fields (nested str): nested strings. Each string indicates a path
            to retrieve the value from ``nest``
    Returns:
        a nest with same structure as ``nest_fields``.
    """
    return map_structure(lambda f: get_field(nested, f), nest_fields)
