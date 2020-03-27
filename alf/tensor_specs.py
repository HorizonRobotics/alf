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
"""TensorSpec with PyTorch types; adapted from Tensorflow's tensor_spec.py:

https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/framework/tensor_spec.py
"""

import gin
import numpy as np

import torch


def torch_dtype_to_str(dtype):
    assert isinstance(dtype, torch.dtype)
    return dtype.__str__()[6:]


@gin.configurable
class TensorSpec(object):
    """Describes a torch.Tensor.
    A TensorSpec allows an API to describe the Tensors that it accepts or
    returns, before that Tensor exists. This allows dynamic and flexible graph
    construction and configuration.
    """

    __slots__ = ["_shape", "_dtype"]

    def __init__(self, shape, dtype=torch.float32):
        """Creates a TensorSpec.
        Args:
            shape (tuple[int]): The shape of the tensor.
            dtype (str or torch.dtype): The type of the tensor values,
                e.g., "int32" or torch.int32
        """
        self._shape = shape
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
    def is_bounded(cls):
        del cls
        return False

    @property
    def shape(self):
        """Returns the `TensorShape` that represents the shape of the tensor."""
        return self._shape

    @property
    def dtype(self):
        """Returns the `dtype` of elements in the tensor."""
        return self._dtype

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

    def constant(self, value, outer_dims=None):
        """Create a constant tensor from the spec.

        Args:
            value : a scalar
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            tensor (torch.Tensor): a tensor of `self._dtype`
        """
        value = torch.as_tensor(value).to(self._dtype)
        assert len(value.size()) == 0, "The input value must be a scalar!"
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return torch.ones(size=shape, dtype=self._dtype) * value

    def zeros(self, outer_dims=None):
        """Create a zero tensor from the spec.

        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            tensor (torch.Tensor): a tensor of `self._dtype`
        """
        return self.constant(0, outer_dims)

    def ones(self, outer_dims=None):
        """Create an all-one tensor from the spec.

        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            tensor (torch.Tensor): a tensor of `self._dtype`
        """
        return self.constant(1, outer_dims)

    def randn(self, outer_dims=None):
        """Create a tensor filled with random numbers from a std normal dist.

        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.

        Returns:
            tensor (torch.Tensor): a tensor of `self._dtype`
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return torch.randn(*shape, dtype=self._dtype)


@gin.configurable
class BoundedTensorSpec(TensorSpec):
    """A `TensorSpec` that specifies minimum and maximum values.
    Example usage:
    ```python
    spec = BoundedTensorSpec((1, 2, 3), torch.float32, 0, (5, 5, 5))
    torch_minimum = torch.as_tensor(spec.minimum, dtype=spec.dtype)
    torch_maximum = torch.as_tensor(spec.maximum, dtype=spec.dtype)
    ```
    Bounds are meant to be inclusive. This is especially important for
    integer types. The following spec will be satisfied by tensors
    with values in the set {0, 1, 2}:
    ```python
    spec = BoundedTensorSpec((3, 5), torch.int32, 0, 2)
    ```
    """

    __slots__ = ("_minimum", "_maximum")

    def __init__(self, shape, dtype=torch.float32, minimum=0, maximum=1):
        """Initializes a new `BoundedTensorSpec`.
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

        if self._dtype.is_floating_point:
            uniform = torch.rand(shape, dtype=self._dtype)
            return ((1 - uniform) * torch.as_tensor(self._minimum) +
                    torch.as_tensor(self._maximum) * uniform)
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
