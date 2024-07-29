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
"""TensorSpec with PyTorch types; adapted from Tensorflow's tensor_spec.py:

https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/framework/tensor_spec.py
"""
from __future__ import annotations
from typing import Optional, Union, Tuple, Dict, List

import numpy as np

import torch

import alf


def torch_dtype_to_str(dtype):
    assert isinstance(dtype, torch.dtype)
    return dtype.__str__()[6:]


@alf.configurable
class TensorSpec(object):
    """Describes a torch.Tensor.

    A TensorSpec allows an API to describe the Tensors that it accepts or
    returns, before that Tensor exists. This allows dynamic and flexible graph
    construction and configuration.
    """

    __slots__ = ["_shape", "_dtype"]

    def __init__(self, shape, dtype=torch.float32):
        """
        Args:
            shape (tuple[int]): The shape of the tensor.
            dtype (str or torch.dtype): The type of the tensor values,
                e.g., "int32" or torch.int32
        """
        self._shape = tuple(shape)
        if isinstance(dtype, str):
            self._dtype = getattr(torch, dtype)
        else:
            assert isinstance(dtype, torch.dtype)
            self._dtype = dtype

    @classmethod
    def from_spec(cls, spec):
        assert isinstance(spec, TensorSpec)
        return cls(spec.shape, spec.dtype)

    @classmethod
    def from_tensor(cls, tensor, from_dim=0):
        """Create TensorSpec from tensor.

        Args:
            tensor (Tensor): tensor from which the spec is extracted
            from_dim (int): use tensor.shape[from_dim:] as shape
        Returns:
            TensorSpec
        """
        assert isinstance(tensor, torch.Tensor)
        return TensorSpec(tensor.shape[from_dim:], tensor.dtype)

    @classmethod
    def from_array(cls, array, from_dim=0):
        """Create TensorSpec from numpy array.

        Args:
            array (np.ndarray|np.number): array from which the spec is extracted
            from_dim (int): use ``array.shape[from_dim:]`` as shape
        Returns:
            TensorSpec
        """
        assert isinstance(array, (np.ndarray, np.number))
        return TensorSpec(array.shape[from_dim:], str(array.dtype))

    def replace(self,
                shape: Union[None, tuple, torch.Size] = None,
                dtype: Optional[torch.dtype] = None) -> TensorSpec:
        """Create a new TensorSpec with part of the properties replaced.

        For example, if we have a TensorSpec like

        .. code-block:: python

            spec = TensorSpec((3, 5), torch.int32)

        You can explicitly create a similar spec with a different dtype by

        .. code-block:: python

            new_spec = spec.replace(dtype=torch.float32)

        """
        new_shape = shape or self.shape
        new_dtype = dtype or self.dtype
        return TensorSpec(shape=new_shape, dtype=new_dtype)

    @classmethod
    def is_bounded(cls):
        del cls
        return False

    @property
    def shape(self):
        """Returns the `TensorShape` that represents the shape of the tensor."""
        return self._shape

    @property
    def numel(self):
        """Returns the number of elements."""
        return int(np.prod(self._shape))

    @property
    def ndim(self):
        """Return the rank of the tensor."""
        return len(self._shape)

    @property
    def dtype(self):
        """Returns the `dtype` of elements in the tensor."""
        return self._dtype

    @property
    def dtype_str(self):
        """The str representation of dtype

        It can be used to construct a numpy array.
        """
        return torch_dtype_to_str(self._dtype)

    @property
    def is_discrete(self):
        """Whether spec is discrete."""
        return not self.dtype.is_floating_point

    @property
    def is_continuous(self):
        """Whether spec is continuous."""
        return self.dtype.is_floating_point

    def __repr__(self):
        return "TensorSpec(shape={}, dtype={})".format(self.shape,
                                                       repr(self.dtype))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.shape == other.shape and self.dtype == other.dtype

    def __ne__(self, other):
        return not self == other

    def __reduce__(self):
        return TensorSpec, (self._shape, self._dtype)

    def _calc_shape(self, outer_dims):
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return shape

    def constant(self, value, outer_dims=None):
        """Create a constant tensor from the spec.

        Args:
            value : a scalar
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        return self.ones(outer_dims) * value

    def zeros(self, outer_dims=None):
        """Create a zero tensor from the spec.

        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        return torch.zeros(self._calc_shape(outer_dims), dtype=self._dtype)

    def numpy_constant(self, value, outer_dims=None):
        """Create a constant np.ndarray from the spec.

        Args:
            value (Number) : a scalar
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            np.ndarray: an array of ``self._dtype``.
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return np.full(shape, value, dtype=self.dtype_str)

    def numpy_zeros(self, outer_dims=None):
        """Create a zero numpy.ndarray from the spec.

        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            np.ndarray: an array of ``self._dtype``.
        """
        return self.numpy_constant(0, outer_dims)

    def ones(self, outer_dims=None):
        """Create an all-one tensor from the spec.

        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        return torch.ones(self._calc_shape(outer_dims), dtype=self._dtype)

    def randn(self, outer_dims=None):
        """Create a tensor filled with random numbers from a std normal dist.

        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return torch.randn(*shape, dtype=self._dtype)

    def rand(self, outer_dims: Tuple[int] = None):
        """Create a tensor filled with random numbers in :math:`[0,1]`.

        Args:
            outer_dims: an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            torch.Tensor: a tensor of ``self._dtype``.
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return torch.rand(*shape, dtype=self._dtype)


@alf.configurable
class BoundedTensorSpec(TensorSpec):
    """A `TensorSpec` that specifies minimum and maximum values.
    Example usage:

    .. code-block:: python

        spec = BoundedTensorSpec((1, 2, 3), torch.float32, 0, (5, 5, 5))
        torch_minimum = torch.as_tensor(spec.minimum, dtype=spec.dtype)
        torch_maximum = torch.as_tensor(spec.maximum, dtype=spec.dtype)

    Bounds are meant to be inclusive. This is especially important for
    integer types. The following spec will be satisfied by tensors
    with values in the set {0, 1, 2}:

    .. code-block:: python

        spec = BoundedTensorSpec((3, 5), torch.int32, 0, 2)

    """

    __slots__ = ("_minimum", "_maximum")

    def __init__(self, shape, dtype=torch.float32, minimum=0, maximum=1):
        """

        Args:
            shape (tuple[int]): The shape of the tensor.
            dtype (str or torch.dtype): The type of the tensor values,
                e.g., "int32" or torch.int32
            minimum: numpy number or sequence specifying the minimum element
                bounds (inclusive). Must be broadcastable to `shape`.
            maximum: numpy number or sequence specifying the maximum element
                bounds (inclusive). Must be broadcastable to `shape`.
        """
        super(BoundedTensorSpec, self).__init__(shape, dtype)

        try:
            min_max = np.broadcast(minimum, maximum, np.zeros(self.shape))
            for m, M, _ in min_max:
                assert m <= M, "Min {} is greater than Max {}".format(m, M)
        except ValueError as exception:
            raise ValueError(
                "minimum or maximum is not compatible with shape. "
                "Message: {!r}.".format(exception))

        self._minimum = np.array(
            minimum, dtype=torch_dtype_to_str(self._dtype))
        self._minimum.setflags(write=False)

        self._maximum = np.array(
            maximum, dtype=torch_dtype_to_str(self._dtype))
        self._maximum.setflags(write=False)

    def replace(self,
                shape: Union[None, tuple, torch.Size] = None,
                dtype: Optional[torch.dtype] = None,
                minimum: Union[None, float, np.ndarray] = None,
                maximum: Union[None, float, np.ndarray] = None
                ) -> BoundedTensorSpec:
        """Create a new BoundedTensorSpec with part of the properties replaced.

        For example, if we have a BoundedTensorSpec like

        .. code-block:: python

            spec = BoundedTensorSpec((3, 5), torch.int32, 0, 2)

        You can explicitly create a similar spec with a different shape and minimum by

        .. code-block:: python

            new_spec = spec.replace(shape=(4, 8), minimum=-1)

        """
        new_shape = shape or self.shape
        new_dtype = dtype or self.dtype
        new_minimum = minimum if minimum is not None else self.minimum
        new_maximum = maximum if maximum is not None else self.maximum
        return BoundedTensorSpec(
            shape=new_shape,
            dtype=new_dtype,
            minimum=new_minimum,
            maximum=new_maximum)

    @classmethod
    def is_bounded(cls):
        del cls
        return True

    @classmethod
    def from_spec(cls, spec):
        assert isinstance(spec, BoundedTensorSpec)
        minimum = getattr(spec, "minimum")
        maximum = getattr(spec, "maximum")
        return BoundedTensorSpec(spec.shape, spec.dtype, minimum, maximum)

    @property
    def minimum(self):
        """Returns a NumPy array specifying the minimum bounds (inclusive)."""
        return self._minimum

    @property
    def maximum(self):
        """Returns a NumPy array specifying the maximum bounds (inclusive)."""
        return self._maximum

    def __repr__(self):
        s = "BoundedTensorSpec(shape={}, dtype={}, minimum={}, maximum={})"
        return s.format(self.shape, repr(self.dtype), repr(self.minimum),
                        repr(self.maximum))

    def __eq__(self, other):
        tensor_spec_eq = super(BoundedTensorSpec, self).__eq__(other)
        return (tensor_spec_eq and np.allclose(self.minimum, other.minimum)
                and np.allclose(self.maximum, other.maximum))

    def __reduce__(self):
        return BoundedTensorSpec, (self._shape, self._dtype, self._minimum,
                                   self._maximum)

    def sample(self, outer_dims=None):
        """Sample uniformly given the min/max bounds.

        Args:
            outer_dims (list[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            tensor (torch.Tensor): a tensor of `self._dtype`
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape

        if self.is_continuous:
            uniform = torch.rand(shape, dtype=self._dtype)
            return ((1 - uniform) * torch.tensor(self._minimum) +
                    torch.tensor(self._maximum) * uniform)
        else:
            # torch.randint cannot have multi-dim lows and highs; currently only
            # support a scalar minimum and maximum
            assert (np.shape(self._minimum) == ()
                    and np.shape(self._maximum) == ())
            return torch.randint(
                low=self._minimum.item(),
                high=self._maximum.item() + 1,
                size=shape,
                dtype=self._dtype)

    def numpy_sample(self, outer_dims=None, rng=np.random):
        """Sample numpy arrays uniformly given the min/max bounds.

        Args:
            outer_dims (list[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
            rng (numpy.random.RandomState): random number generator

        Returns:
            np.ndarray: an array of ``self._dtype``
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape

        if self.is_continuous:
            uniform = rng.rand(*shape).astype(self.dtype_str)
            return (1 - uniform) * self._minimum + self._maximum * uniform
        else:
            return rng.randint(
                low=self._minimum,
                high=self._maximum + 1,
                size=shape,
                dtype=self.dtype_str)


# yapf: disable
NestedTensorSpec = Union[
    TensorSpec,
    List['NestedTensorSpec'],
    # An empty tuple is also considered a NestedTensorSpec
    Tuple[()],
    # Though Tuple['NestedTensorSpec', ...] is not the tightest specification, it is
    # here to cover the case of "(named) tuple of NestedTensorSpec".
    Tuple['NestedTensorSpec', ...],
    Dict[str, 'NestedTensorSpec']
]

NestedBoundedTensorSpec = Union[
    BoundedTensorSpec,
    List['NestedBoundedTensorSpec'],
    # An empty tuple is also considered a NestedBoundedTensorSpec
    Tuple[()],
    # Though Tuple['NestedBoundedTensorSpec', ...] is not the tightest specification,
    # it is here to cover the case of "(named) tuple of NestedBoundedTensorSpec".
    Tuple['NestedBoundedTensorSpec', ...],
    Dict[str, 'NestedBoundedTensorSpec']
]
# yapf: enable


def concat_specs(specs: Union[NestedTensorSpec, NestedBoundedTensorSpec]):
    """Create a new spec which is the concatenation of the flattened input specs.

    Args:
        specs: a nested structure of TensorSpec or BoundedTensorSpec.
    Returns:
        a new spec which is the concatenation of the flattened input specs.
    """
    specs = alf.nest.flatten(specs)
    dtype = specs[0].dtype
    spec_type = type(specs[0])
    assert all([spec.dtype == dtype for spec in specs
                ]), ("All action specs should have the same dtype")
    assert all([type(spec) == spec_type for spec in specs
                ]), ("All action specs should have the same type")
    if spec_type == alf.BoundedTensorSpec:
        minimum = np.concatenate([
            np.broadcast_to(spec.minimum, spec.shape).reshape(-1)
            for spec in specs
        ],
                                 axis=0)
        maximum = np.concatenate([
            np.broadcast_to(spec.maximum, spec.shape).reshape(-1)
            for spec in specs
        ],
                                 axis=0)
        return alf.BoundedTensorSpec(
            shape=(sum(spec.numel for spec in specs), ),
            minimum=minimum,
            maximum=maximum,
            dtype=dtype)
    else:
        return alf.TensorSpec(
            shape=(sum(spec.numel for spec in specs), ), dtype=dtype)
