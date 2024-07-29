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
"""SegmentTree."""

import math
import torch
import torch.nn as nn

import alf
from alf.nest.utils import convert_device


class SegmentTree(nn.Module):
    """
    Data structure to allow efficient calculation of the summary statistics over a
    segment of elements.
    See https://en.wikipedia.org/wiki/Segment_tree for detail.

    In this implementation, ``values[1]`` is the root. ``values[capacity: 2*capacity]``
    are the leaves. The two children of an internal node ``values[i]`` are ``values[2*i]``
    and ``values[2*i+1]``. And ``values[i]`` is set to ``op(values[2*i], values[2*i+1])``.
    Each leaf represent a value set through ``__setitem__``. All the nodes of
    tree are initialized to be zeros.
    """

    def __init__(self,
                 capacity,
                 op,
                 dtype=torch.float32,
                 device="cpu",
                 name="SegmentTree"):
        super().__init__()
        self._name = name
        self._device = device
        with alf.device(self._device):
            self.register_buffer("_values",
                                 torch.zeros((2 * capacity, ), dtype=dtype))
        self._op = op
        self._capacity = capacity
        self._leftmost_leaf = 1
        self._depth = 0
        while self._leftmost_leaf < capacity:
            self._leftmost_leaf *= 2
            if self._leftmost_leaf < capacity:
                self._depth += 1

    def __setitem__(self, indices, values):
        """Set the value of leaves and update the internal nodes.

        Args:
            indices (Tensor): 1-D int64 Tensor. Its values should be in range
                [0, capacity).
            values (Tensor): 1-D Tensor with the same shape as ``indices``
        """

        def _step(indices):
            """
            Calculate the parent value from its children.
            """
            indices = torch.unique(indices >> 1)
            left = self._values[indices * 2]
            right = self._values[indices * 2 + 1]
            self._values[indices] = op(left, right)
            return indices

        with alf.device(self._device):
            indices = convert_device(indices)
            values = convert_device(values)

            assert indices.ndim == 1
            assert values.ndim == 1
            assert indices.shape == values.shape, (
                "indices and values should be 1-D tensor with the same length. "
                "Got %s and %s." % (indices.shape, values.shape))
            op = self._op
            indices, order = torch.sort(indices)
            values = values[order]
            assert indices[-1] < self._capacity
            indices = self._index_to_leaf(indices)
            self._values[indices] = values

            num_large = (indices >= self._leftmost_leaf).to(torch.int64).sum()
            if num_large > 0:
                large_indices = indices[:num_large]
                small_indices = indices[num_large:]
                large_indices = _step(large_indices)
                indices = torch.cat([large_indices, small_indices])

            for _ in range(self._depth):
                indices = _step(indices)

    def __getitem__(self, idx):
        """Get the values of leaves.

        Args:
            idx (Tensor): 1-D int64 Tensor. Its values should be in range
                [0, capacity).
        Returns:
            Tensor: with same shaps as idx.
        """
        with alf.device(self._device):
            idx = convert_device(idx)
            assert 0 <= idx.min()
            assert idx.max() < self._capacity
            result = self._values[self._index_to_leaf(idx)]
        return convert_device(result)

    def _index_to_leaf(self, idx):
        """
        Make sure idx=0 is the leftmost leaf.
        """
        idx = idx + self._leftmost_leaf
        idx = torch.where(idx >= 2 * self._capacity, idx - self._capacity, idx)
        return idx

    def _leaf_to_index(self, leaf):
        idx = leaf - self._leftmost_leaf
        idx = torch.where(idx < 0, idx + self._capacity, idx)
        return idx

    def summary(self):
        """The summary of the tree.

        If ``op`` is ``torch.add``, it's the sum of all values.
        If ``op`` is ``torch.min``, it's the min of all values.
        If ``op`` is ``torch.max``, it's the max of all values.

        Returns:
            a scalar
        """
        return convert_device(self._values[1])


class SumSegmentTree(SegmentTree):
    """SegmentTree with sum operation."""

    def __init__(self,
                 capacity,
                 dtype=torch.float32,
                 device="cpu",
                 name="SumSegmentTree"):
        super().__init__(
            capacity, torch.add, dtype=dtype, device=device, name=name)
        self._nnz = 0

    def __setitem__(self, indices, values):
        assert values.min() >= 0
        leaves = self._index_to_leaf(indices)
        nnz = (values != 0).sum() - (self._values[leaves] != 0).sum()
        self._nnz += int(nnz.cpu().numpy())

        super().__setitem__(indices, values)

    @property
    def nnz(self):
        """The number of non-zeros."""
        return self._nnz

    def find_sum_bound(self, thresholds):
        """
        The result is an int64 Tensor with the same shape as `thresholds`.
        result[i] is the minimum idx such that
            thresholds[i] < values[0] + ... + values[idx]

        values[result[i]] will never be 0.

        Args:
            thresholds (Tensor): 1-D Tensor. All the elements in `thresholds`
                should be smaller than self.summary()
        Returns:
            Tensor: 1-D int64 Tensor with the same shape as ``thresholds``.
                Note that if thresholds[i] == root,  result[i] will be
                the index of the non-zero value with the largest index.
        Raises:
            ValueError:  If one or more of ``thresholds`` is greater than ``summary()``.
        """

        def _step(indices, thresholds):
            """Choose one of the children of each index based on threshold.

            If threshold is greater than or equal to the
            left child, choose the right child and update threhsold to threhsold - left_child.
            Otherwise choose left child and keep threshold unchanged.
            """
            indices *= 2
            left = self._values[indices]
            right = self._values[indices + 1]
            # The condition (thresholds >= left) * (right == 0) is only possible
            # if the original threshold == summary(), we want to make sure we
            # still get an index corresponding to non-zero value.
            greater = (thresholds >= left) * (right > 0)
            indices = torch.where(greater, indices + 1, indices)
            thresholds = torch.where(greater, thresholds - left, thresholds)
            return indices, thresholds

        with alf.device(self._device):
            if not torch.all(thresholds <= self.summary()):
                raise ValueError("thresholds cannot "
                                 "be greater than summary(): got %s vs. %s" %
                                 (thresholds.max(), self.summary()))
            thresholds = convert_device(thresholds)
            indices = torch.ones_like(thresholds, dtype=torch.int64)
            for _ in range(self._depth):
                indices, thresholds = _step(indices, thresholds)

            is_small = indices < self._capacity
            num_small = is_small.to(torch.int64).sum()
            if num_small > 0:
                i = torch.where(is_small)[0]
                small_indices = indices[i]
                small_thresholds = thresholds[i]
                small_indices, _ = _step(small_indices, small_thresholds)
                indices[i] = small_indices

        return convert_device(self._leaf_to_index(indices))


class MinSegmentTree(SegmentTree):
    """SegmentTree with min operation."""

    def __init__(self,
                 capacity,
                 dtype=torch.float32,
                 device="cpu",
                 name="MinSegmentTree"):
        super().__init__(capacity, torch.min, dtype, device=device, name=name)


class MaxSegmentTree(SegmentTree):
    """SegmentTree with max operation."""

    def __init__(self,
                 capacity,
                 dtype=torch.float32,
                 device="cpu",
                 name="MaxSegmentTree"):
        super().__init__(capacity, torch.max, dtype, device=device, name=name)
