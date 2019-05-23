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
"""Implementation of a simple Memory module described in arXiv:1803.10760."""

import abc
import math
from typing import Callable

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations

from tf_agents.networks import network

from alf.utils.common import expand_dims_as


class Memory(object):
    """Abstract base class for Memory."""

    def __init__(self, dim, size, state_spec, name="Memory"):
        """Create an instance of `Memory`.

        Args:
            dim (int): dimension of memory content
            size (int): number of memory slots
            state_spec (nested TensorSpec): the spec for the states
            name (str): name of this memory
        """
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
            resutl (Tensor): shape is same as keys. result[..., i] is the read
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
    memory content can be retrived thrugh attention mechanism using `read`.
    """

    def __init__(self,
                 dim,
                 size,
                 snapshot_only=False,
                 normalize=True,
                 scale=None,
                 usage_decay=None,
                 name='MemoryWithUsage'):
        """Create an instance of `MemoryWithUsage`.

        See Methods 2.3 of "Unsupervised Predictive Memory in a Goal-Directed
        Agent"

        Args:
            dim (int): dimension of memory content
            size (int): number of memory slots
            snapshot_only (bool): If True, only keeps the last snapshot of the
              memory instead of keeping all the memory snapshot at every steps.
              If True, gradient cannot be propagated to the writer.
            normalize (bool): If True, use cosine similarity, otherwise use dot
              product.
            scale (None|float): Scale the similarity by this. If scale is None,
              a default value is used based `normalize`. If `normalize` is True,
              `scale` is default to 5.0. If `normalize` is False, `scale` is
              default to `1/sqrt(dim)`.
            usage_decay (None|float): The usage will be scaled by this factor
              at every `write` call. If None, it is default to `1 - 1 / size`
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
        state_spec = (tf.TensorSpec([size, dim], dtype=tf.float32),
                      tf.TensorSpec([size], dtype=tf.float32))
        super(MemoryWithUsage, self).__init__(
            dim, size, state_spec=state_spec, name=name)

    def build(self, batch_size):
        """Build the memory for batch_size.

        User does not need to call this explictly. `read` and `write` will
        automatically call this if the memory has not been built yet.

        Note: Subsequent `write` and `read` must match this `batch_size`
        Args:
            batch_size (int): batch size of the model.
        """
        self._batch_size = batch_size
        self._initial_memory = tf.zeros((batch_size, self.size, self.dim))
        self._initial_usage = tf.zeros((batch_size, self.size))
        if self._snapshot_only:
            self._memory = tf.Variable(self._initial_memory, trainable=False)
            self._usage = tf.Variable(self._initial_usage, trainable=False)
        else:
            self._memory = self._initial_memory
            self._usage = self._initial_usage
        self._built = True

    def genkey_and_read(self, keynet: Callable, query, flatten_result=True):
        """Generate key and read.

        Args:
            keynet (Callable): keynet(query) is a tensor of shape
              (batch_size, num_keys * (dim + 1))
            query (Tensor): the query from which the keys are generated
            flatten_result (bool): If True, the result shape will be
               (batch_size, num_keys * dim), otherwise it is
               (batch_size, num_keys, dim)
        Returns:
            resutl Tensor: If flatten_result is True,
              its shape is (batch_size, num_keys * dim), otherwise it is
              (batch_size, num_keys, dim)

        """
        batch_size = query.shape[0]
        keys_and_scales = keynet(query)
        num_keys = keys_and_scales.shape[-1] // (self.dim + 1)
        assert num_keys * (self.dim + 1) == keys_and_scales.shape[-1]
        keys, scales = tf.split(
            keys_and_scales,
            num_or_size_splits=[num_keys * self.dim, num_keys],
            axis=-1)
        keys = tf.reshape(keys, keys.shape[:-1].concatenate((num_keys,
                                                             self.dim)))
        scales = tf.math.softplus(tf.reshape(scales, keys.shape[:-1]))

        r = self.read(keys, scales)
        if flatten_result:
            r = tf.reshape(r, (batch_size, num_keys * self.dim))
        return r

    def read(self, keys, scale=None):
        """Read from memory.

        Read the memory for given the keys. For each key in keys we will get one
        result as `r = sum_i M[i] a[i]` where `M[i]` is the memory content
        at location i and `a[i]` is the attention weight for key at location i.
        `a` is calculated as softmax of a scaled similarity between key and
        each memory content: `a[i] = exp(scale*sim[i])/(sum_i scale*sim[i])`

        Args:
            keys (Tensor): shape[-1] is dim.
              For single key read, the shape is (batch_size, dim).
              For multiple key read, the shape is (batch_szie, k, dim), where
              k is the number of keys.
            scale (None|float|Tensor): shape is () or keys.shape[:-1]. The
              cosine similarities are multiplied with `scale` before softmax
              is applied. If None, use the scale provided at constructor.
        Returns:
            resutl Tensor: shape is same as keys. result[..., i] is the read
              result for the corresponding key.

        """
        if not self._built:
            self.build(keys.shape[0])
        assert 2 <= len(keys.shape) <= 3
        assert keys.shape[0] == self._batch_size
        assert keys.shape[-1] == self.dim

        if scale is None:
            scale = self._scale
        else:
            if isinstance(scale, (int, float)):
                pass
            else:  # assuming it's Tensor
                scalar = expand_dims_as(scalar, keys)
        sim = layers.dot([keys, self._memory],
                         axes=-1,
                         normalize=self._normalize)
        sim = sim * scale

        attention = activations.softmax(sim)
        result = layers.dot([attention, self._memory], axes=(-1, 1))

        if len(sim.shape) > 2:  # multiple read keys
            usage = tf.reduce_sum(
                attention, axis=tf.range(1,
                                         len(sim.shape) - 1))
        else:
            usage = attention

        if self._snapshot_only:
            self._usage.assign_add(usage)
        else:
            self._usage = self._usage + usage

        return result

    def write(self, content):
        """Write content to memory.

        Append the content to memory. If the memory is full, the slot with the
        smallest usage will be overriden. The usage is calculated during read as
        the sum of past attentions.
        
        Args:
            content (Tensor): shape should be (b, dim)
        """
        if not self._built:
            self.build(content.shape[0])
        assert len(content.shape) == 2
        assert content.shape[0] == self._batch_size
        assert content.shape[1] == self.dim

        location = tf.argmin(self._usage, -1)
        loc_weight = tf.one_hot(location, depth=self._size)

        # reset usage for at the new location
        usage = self._usage * (1 - loc_weight) + loc_weight

        # update content at the new location
        loc_weight = tf.expand_dims(loc_weight, 2)
        memory = self._usage_decay * (1 - loc_weight) * self._memory \
            + loc_weight * tf.expand_dims(content, 1)
        if self._snapshot_only:
            self._usage.assign(usage)
            self._memory.assign(memory)
        else:
            self._usage = usage
            self._memory = memory

    def reset(self):
        """Reset the the memory to the initial state.

        Both memory and uage are set to zeros.
        """
        if self._snapshot_only:
            self._usage.assign(self._initial_usage)
            self._memory.assign(self._initial_memory)
        else:
            self._usage = self._initial_usage
            self._memory = self._initial_memory

    @property
    def usage(self):
        """Get the usage for each memory slots.

        Returns:
            usage (Tensor) of shape (batch_size, size)

        """
        return self._usage

    def __str__(self):
        s = "MemoryWithUsage: size=%s dim=%s" % (self.size, self.dim) + "\n" \
            + " memory: " + str(self._memory) + "\n" \
            + " usage: " + str(self._usage)
        return s

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
            tf.nest.assert_same_structure(states, self.state_spec)
            self._memory, self._usage = states
            self._batch_size = self._memory.shape[0]
            self._built = True
