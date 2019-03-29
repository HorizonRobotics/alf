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

import abc

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations

from tf_agents.networks import network


class Memory(network.Network):
    def __init__(self, dim, size):
        """Creat an instance of `Memory`

        Args:
            dim (int): dimension of memory content
            size (int): number of memory slots
        """
        self._dim = dim
        self._size = size

    @property
    def dim(self):
        return self._dim

    @property
    def size(self):
        return self._size

    @abc.abstractmethod
    def read(self, keys):
        """Read out memory vectors given keys

        Args:
            keys (Tensor): shape is (b, dim) or (b, k, dim) where b is batch
              size, k is the number of read keys, and dim is memory content
              dimension
        Returns:
            result (Tensor): shape is same as `keys`.
        """
        pass

    @abc.abstractmethod
    def write(self, content):
        """Write content to memory
        """
        pass


class MemoryWithUsage(Memory):
    def __init__(self,
                 dim,
                 size,
                 snapshot_only=False,
                 normalize=True,
                 scale=None):
        """Creat an instance of `MemoryWithUsage`
        See Methods 2.3 of "Unsupervised Predictive Memory in a Goal-Directed
        Agent"

        Args:
            dim (int): dimension of memory content
            size (int): number of memory slots
            snapshot_only (bool): If True, only keeps the last snapshot of the
               memory instead of keeping all the memory snapshot at every steps.
               If True, gradient cannot be propagated to the writer.
            normalize (bool): If True, use cosine similarity, otherwise use dot
                product
            scale (float): scale the similarity by this
        """
        super(MemoryWithUsage, self).__init__(dim, size)
        self._normalize = normalize
        self._scale = scale
        self._built = False
        self._snapshot_only = snapshot_only
        if snapshot_only:
            # TODO: support this by using variables for _memory and _usage
            raise ValueError("Not supported yet")

    def build(self, batch_size):
        self._batch_size = batch_size
        self._memory = tf.zeros((batch_size, self.size, self.dim))
        self._usage = tf.zeros((batch_size, self.size))
        self._built = True

    def read(self, keys, scale=None):
        """
        Read the memory given the keys. For each key in keys we will get one
        result as `r = sum_i M[i] a[i]` where `M[i]` is the memory content
        at location i and `a[i]` is the attention weight for key at location i.
        `a` is calculated as softmax of a scaled similarity between key and
        each memory content: `a[i] = exp(scale*sim[i])/(sum_i scale*sim[i])`

        Args:
            keys (Tensor): shape[-1] is d
            scale (None|float|Tensor): shape is () or keys.shape[:-1]. The
             cosine similarities are multiplied with temperature before softmax
             is applied. If None, use the scale provided at constructor.
        Returns:
            resutl Tensor: shape is same as keys. result[..., i] is the read
              result for the corresponding key.
        """
        if not self._built:
            self.build(keys.shape[0])
        assert len(keys.shape) >= 2
        assert keys.shape[0] == self._batch_size
        assert keys.shape[-1] == self.dim

        if scale is None:
            scale = self._scale
        else:
            if isinstance(scale, (int, float)):
                pass
            else:  # assuming it's Tensor
                assert len(scale.shape) < len(keys.shape)
                assert scale.shape == keys.shape[:len(scale.shape)]
                k = len(keys.shape) - len(scale.shape) - 1
                if k > 0:
                    scale = tf.reshape(scale, keys.shape.concatenate(
                        (1, ) * k))

        sim = layers.dot([keys, self._memory],
                         axes=-1,
                         normalize=self._normalize)
        if scale is not None:
            sim = sim * scale

        attention = activations.softmax(sim)
        result = layers.dot([attention, self._memory], axes=(-1, 1))

        if len(sim.shape) > 2:  # multiple read keys
            usage = tf.reduce_sum(attention, 1)
        else:
            usage = attention
        self._usage = self._usage + usage

        return result

    def write(self, content):
        """
        Append the content into memory. If the memory is full, the slot with
        the smallest usage will be overriden. The usage is calculated during
        read as the sum of past attentions.
        
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
        self._usage = self._usage * (1 - loc_weight) + loc_weight

        # update content at the new location
        loc_weight = tf.expand_dims(loc_weight, 2)
        self._memory = self._memory * (1 - loc_weight) \
            +  loc_weight * tf.expand_dims(content, 1)

    @property
    def usage(self):
        """Get the usage for each memory slots
        Returns:
            usage (tf.Tensor) of shape (batch_size, size)
        """
        return self._usage

    def __str__(self):
        s = "MemoryWithUsage: size=%s dim=%s" % (self.size, self.dim) + "\n" \
            + " memory: " + str(self._memory) + "\n" \
            + " usage: " + str(self._usage)
        return s
