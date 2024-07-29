# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Various memory classes.

Currently, all the memory classes implemented here only supports memory in one
episode, which means that the memory is reset at the beginning of an episode.
"""

import abc
import math
import six
from typing import Callable
import torch
import torch.nn.functional as F
import torch.nn as nn

import alf
from alf.utils.common import expand_dims_as
from alf.utils.math_ops import argmin


@six.add_metaclass(abc.ABCMeta)
class Memory(object):
    """Abstract base class for Memory."""

    def __init__(self, dim, size, state_spec, name="Memory"):
        """

        Args:
            dim (int): dimension of memory content
            size (int): number of memory slots
            state_spec (nested TensorSpec): the spec for the states
            name (str): name of this memory
        """
        super(Memory, self).__init__()
        self._dim = dim
        self._size = size
        self._state_spec = state_spec
        self._name = name

    @property
    def dim(self):
        """Get the dimension of each content vector."""
        return self._dim

    @property
    def size(self):
        """Get the size of the memory (i.e. the number of memory slots)."""
        return self._size

    @property
    def state_spec(self):
        """Get the state tensor specs."""
        return self._state_spec

    @abc.abstractmethod
    def read(self, keys):
        """Read out memory vectors for the given keys.

        Args:
            keys (Tensor): shape is (b, dim) or (b, k, dim) where b is batch
              size, k is the number of read keys, and dim is memory content
              dimension
        Returns:
            result (Tensor): shape is same as keys. result[..., i] is the read
              result for the corresponding key.
        """
        pass

    @abc.abstractmethod
    def write(self, content):
        """Write content to memory.

        The way how it is written to the memory buffer is decided by the
        subclass.

        Args:
            content (Tensor): shape should be (b, dim)
        """
        pass


class MemoryWithUsage(Memory):
    """Memory with usage indicator.

    MemoryWithUsage stores memory in a matrix. During memory `write`, the memory
    slot with the smallest usage is replaced by the new memory content. The
    memory content can be retrieved through attention mechanism using `read`.

    This implementation follows the one described in arXiv:1803.10760.
    """

    def __init__(self,
                 dim,
                 size,
                 snapshot_only=False,
                 normalize=True,
                 scale=None,
                 usage_decay=None,
                 name='MemoryWithUsage'):
        """

        See Methods 2.3 of `Unsupervised Predictive Memory in a Goal-Directed
        Agent <https://arxiv.org/abs/1803.10760>`_

        Args:
            dim (int): dimension of memory content
            size (int): number of memory slots
            snapshot_only (bool): If True, only keeps the last snapshot of the
              memory instead of keeping all the memory snapshot at every steps.
              If True, gradient cannot be propagated to the writer.
            normalize (bool): If True, use cosine similarity, otherwise use dot
              product.
            scale (None|float): Scale the similarity by this. If scale is None,
              a default value is used based ``normalize``. If ``normalize`` is True,
              ``scale`` is default to 5.0. If ``normalize`` is False, ``scale`` is
              default to ``1/sqrt(dim)``.
            usage_decay (None|float): The usage will be scaled by this factor
              at every ``write`` call. If None, it is default to ``1 - 1 / size``
        """
        self._normalize = normalize
        if scale is None:
            if normalize:
                scale = 5.0
            else:
                scale = 1. / math.sqrt(dim)
        self._scale = scale
        self._built = False
        self._snapshot_only = snapshot_only
        if usage_decay is None:
            usage_decay = 1. - 1. / size
        self._usage_decay = usage_decay
        state_spec = (alf.TensorSpec((size, dim), dtype=torch.float32),
                      alf.TensorSpec((size, ), dtype=torch.float32))
        super(MemoryWithUsage, self).__init__(
            dim, size, state_spec=state_spec, name=name)

    def build(self, batch_size):
        """Build the memory for batch_size.

        User does not need to call this explicitly. `read` and `write` will
        automatically call this if the memory has not been built yet.

        Note: Subsequent `write` and `read` must match this `batch_size`
        Args:
            batch_size (int): batch size of the model.
        """
        self._batch_size = batch_size
        self._memory = torch.zeros(batch_size, self.size, self.dim)
        self._usage = torch.zeros(batch_size, self.size)
        self._built = True

    def create_keynet(self, query_spec, num_keys):
        """Create a net which can be used to generate keys.

        The created keynet can be used with ``genkey_and_read``.

        Args:
            query_spec (alf.TensorSpec): the spec for the query
            num_keys (int): the number of keys to be generated.
        Returns:
            Callable: a function which calculates ``num_keys`` keys given query.
        """
        assert isinstance(
            query_spec, alf.TensorSpec), ("Wrong type for "
                                          "query_spec: %s" % type(query_spec))
        assert query_spec.ndim == 1, (
            "Query mush be a rank-1 tensor. Got: %s" % query_spec.ndim)
        return alf.layers.FC(query_spec.shape[0], num_keys * (self.dim + 1))

    def genkey_and_read(self, keynet: Callable, query, flatten_result=True):
        """Generate key and read.

        Args:
            keynet (Callable): ``keynet(query)`` is a tensor of shape
                (batch_size, num_keys * (dim + 1)). ``keynet`` can be created
                using ``create_keynet``.
            query (Tensor): the query from which the keys are generated
            flatten_result (bool): If True, the result shape will be
                (batch_size, num_keys * dim), otherwise it is
                (batch_size, num_keys, dim)
        Returns:
            Tensor: If flatten_result is True, its shape is ``(batch_size, num_keys * dim)``,
                otherwise it is ``(batch_size, num_keys, dim)``

        """
        batch_size = query.shape[0]
        keys_and_scales = keynet(query)
        num_keys = keys_and_scales.shape[-1] // (self.dim + 1)
        assert num_keys * (self.dim + 1) == keys_and_scales.shape[-1]
        keys = keys_and_scales[:, :num_keys * self.dim]
        scales = keys_and_scales[:, num_keys * self.dim:]
        keys = keys.reshape(batch_size, num_keys, self.dim)
        scales = F.softplus(scales)

        r = self.read(keys, scales)
        if flatten_result:
            r = r.reshape(batch_size, num_keys * self.dim)
        return r

    def read(self, keys, scale=None):
        r"""Read from memory.

        Read the memory for given the keys. For each key in keys we will get one
        result as :math:`r = \sum_i M_i a_i` where :math:`M_i` is the memory content
        at location i and :math:`a_i` is the attention weight for key at location i.
        :math:`a` is calculated as softmax of a scaled similarity between key and
        each memory content: :math:`a_i = \exp(\frac{scale*sim_i}{\sum_i scale*sim_i})`

        Args:
            keys (Tensor): shape[-1] is dim.
              For single key read, the shape is (batch_size, dim).
              For multiple key read, the shape is (batch_szie, k, dim), where
              k is the number of keys.
            scale (None|float|Tensor): shape is () or keys.shape[:-1]. The
              cosine similarities are multiplied with ``scale`` before softmax
              is applied. If None, use the scale provided at constructor.
        Returns:
            result Tensor: shape is same as keys. result[..., i] is the read
              result for the corresponding key.

        """
        if not self._built:
            self.build(keys.shape[0])
        assert 2 <= keys.ndim <= 3
        assert keys.shape[0] == self._batch_size
        assert keys.shape[-1] == self.dim

        multikey = keys.ndim == 3
        if not multikey:
            keys = keys.unsqueeze(1)

        # B: batch size, K: number of keys, N: memory size, D: dimension of the memory
        sim = torch.bmm(keys, self._memory.transpose(1, 2))  # [B, K, N]
        if self._normalize:
            key_norm = 1 / (1e-30 + keys.norm(dim=2))  # [B, K]
            mem_norm = 1 / (1e-30 + self._memory.norm(dim=2))  # [B, N]
            key_norm = key_norm.unsqueeze(-1)  # [B, K, 1]
            mem_norm = mem_norm.unsqueeze(1)  # [B, 1, N]
            sim = sim * key_norm * mem_norm

        if scale is None:
            scale = self._scale
        else:
            if isinstance(scale, (int, float)):
                pass
            else:  # assuming it's Tensor
                scale = expand_dims_as(scale, sim)

        sim = sim * scale  # [B, K, N]

        attention = F.softmax(sim, dim=2)
        result = torch.bmm(attention, self._memory)  # [B, K, D]

        if multikey:
            usage = attention.sum(1)  # [B, N]
        else:
            usage = attention.squeeze(1)

        if self._snapshot_only:
            self._usage.add_(usage.detach())
        else:
            self._usage = self._usage + usage

        if not multikey:
            result = result.squeeze(1)
        return result

    def write(self, content):
        """Write content to memory.

        Append the content to memory. If the memory is full, the slot with the
        smallest usage will be overridden. The usage is calculated during read as
        the sum of past attentions.

        Args:
            content (Tensor): shape should be (b, dim)
        """
        if not self._built:
            self.build(content.shape[0])
        assert len(content.shape) == 2
        assert content.shape[0] == self._batch_size
        assert content.shape[1] == self.dim

        location = argmin(self._usage)  # [B]
        loc_weight = F.one_hot(location, num_classes=self._size)  # [B, N]

        # reset usage for at the new location
        usage = self._usage * (1 - loc_weight) + loc_weight  # [B, N]

        # update content at the new location
        loc_weight = loc_weight.unsqueeze(2)  # [B, N, 1]
        memory = (self._usage_decay * (1 - loc_weight) * self._memory +
                  loc_weight * content.unsqueeze(1))
        if self._snapshot_only:
            self._usage = usage.detach()
            self._memory = memory.detach()
        else:
            self._usage = usage
            self._memory = memory

    def reset(self):
        """Reset the the memory to the initial state.

        Both memory and uage are set to zeros.
        """
        batch_size = self._batch_size
        self._memory = torch.zeros(batch_size, self.size, self.dim)
        self._usage = torch.zeros(batch_size, self.size)

    @property
    def usage(self):
        """Get the usage for each memory slots.

        Returns:
            usage (Tensor) of shape (batch_size, size)

        """
        return self._usage

    @property
    def states(self):
        """Get the states of the memory.

        Returns:
            memory states: tuple of memory content and usage tensor.

        """
        assert not self._snapshot_only, (
            "states() is not supported for snapshot_only memory")
        return (self._memory, self._usage)

    def from_states(self, states):
        """Restore the memory from states.

        Args:
            states (tuple of Tensor): It is should be obtained from states().
        """
        assert not self._snapshot_only, (
            "from_states() is not supported for snapshot_only memory")
        if states is None:
            self._memory = None
            self._usage = None
            self._built = False
        else:
            alf.nest.assert_same_structure(states, self.state_spec)
            self._memory, self._usage = states
            self._batch_size = self._memory.shape[0]
            self._built = True


class FIFOMemory(Memory):
    """A Simple FIFO memory.

    When new memory slots are written, the oldest memory slots are removed.
    """

    def __init__(self, dim, size, name="FIFOMemory"):
        """
        Args:
            dim (int): dimension of memory content
            size (int): number of memory slots
        """
        self._built = False
        state_spec = (alf.TensorSpec((size, dim), dtype=torch.float32),
                      alf.TensorSpec((), dtype=torch.int64))
        self._range = torch.arange(size).unsqueeze(0)
        super().__init__(dim, size, state_spec=state_spec, name=name)

    def build(self, batch_size):
        """Build the memory for batch_size.

        User does not need to call this explicitly. `read` and `write` will
        automatically call this if the memory has not been built yet.

        Note: Subsequent `write` and `read` must match this `batch_size`
        Args:
            batch_size (int): batch size of the model.
        """
        self._batch_size = batch_size
        self._memory = torch.zeros(batch_size, self.size, self.dim)
        self._current_size = torch.zeros(batch_size, dtype=torch.int64)
        self._built = True

    def write(self, content):
        """Write content to memory.

        Append the content to memory. If the memory is full, the oldest slot
        will be removed.

        Args:
            content (Tensor): shape should be [b, dim] or [b, k, dim] where k
                means the number of memory slots to be written
        """
        if not self._built:
            self.build(content.shape[0])
        if content.ndim == 2:
            content = content.unsqueeze(1)
        assert content.shape[0] == self._batch_size
        assert content.shape[2] == self.dim
        k = content.shape[1]

        self._memory = torch.cat([content, self._memory[:, :-k, :]], dim=1)
        self._current_size = self._current_size + k

    def read(self, keys):
        raise NotImplementedError()

    @property
    def states(self):
        """Get the states of the memory.

        Returns:
            memory states: tuple of memory content and usage tensor.

        """
        return (self._memory, self._current_size)

    def from_states(self, states):
        """Restore the memory from states.

        Args:
            states (tuple of Tensor): It is should be obtained from states().
        """
        if states is None:
            self._memory = None
            self._current_size = None
            self._built = False
        else:
            alf.nest.assert_same_structure(states, self.state_spec)
            self._memory, self._current_size = states
            self._batch_size = self._memory.shape[0]
            self._built = True

    def memory(self):
        return self._memory

    def mask(self):
        """Get the mask for the stored memory.

        Returns:
            Tensor: shape=(batch_size, size), dtype=torch.bool
        """
        return self._range < self._current_size.unsqueeze(-1)
