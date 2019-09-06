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
"""Classes for storing data for sampling."""

import tensorflow as tf

from tf_agents.utils import common as tfa_common


class DataBuffer(tf.Module):
    """A simple circular buffer supporting random sampling.
    """

    def __init__(self, tensor_spec: tf.TensorSpec, capacity,
                 name="DataBuffer"):
        """Create a DataBuffer.

        Args:
            tensor_spec (TensorSpec): spec for the data item (without batch
                dimension) to be stored.
            capacity (int): capacity of the buffer.
            name (str): name of the buffer
        """
        super().__init__()
        self._capacity = capacity
        shape = shape = [capacity] + tensor_spec.shape.as_list()
        self._buffer = tfa_common.create_variable(
            name=name + "/buffer",
            initializer=tf.zeros(shape, dtype=tensor_spec.dtype),
            dtype=tensor_spec.dtype,
            shape=shape,
            trainable=False)
        self._current_size = tfa_common.create_variable(
            name=name + "/size",
            initializer=0,
            dtype=tf.int64,
            shape=(),
            trainable=False)
        self._current_pos = tfa_common.create_variable(
            name=name + "/pos",
            initializer=0,
            dtype=tf.int64,
            shape=(),
            trainable=False)

    def add_batch(self, batch):
        """Add a batch of items to the buffer.

        Args:
            batch (Tensor): shape should be [batch_size] + tensor_space.shape
        """
        batch_size = batch.shape[0]
        if batch_size >= self._capacity:
            self._current_size.assign(self._capacity)
            self._current_pos.assign(0)
            self._buffer.assign(batch[-self._capacity:, ...])
        elif batch_size + self._current_pos <= self._capacity:
            new_pos = self._current_pos + batch_size
            self._buffer.scatter_update(
                tf.IndexedSlices(batch, tf.range(self._current_pos, new_pos)))
            self._current_pos.assign(new_pos)
            self._current_size.assign(
                tf.minimum(self._current_size + batch_size, self._capacity))
        else:
            first = self._capacity - self._current_pos
            self._buffer.scatter_update(
                tf.IndexedSlices(batch[0:first, ...],
                                 tf.range(self._current_pos, self._capacity)))
            new_pos = batch_size - first
            self._buffer.scatter_update(
                tf.IndexedSlices(batch[first:, ...], tf.range(0, new_pos)))
            self._current_pos.assign(new_pos)
            self._current_size.assign(self._capacity)

    def get_batch(self, batch_size):
        """Get batsh_size random samples in the buffer.

        Args:
            batch_size (int): batch size
        Returns:
            Tensor of shape [batch_size] + tensor_spec.shape
        """
        indices = tf.random.uniform(
            shape=(batch_size, ),
            dtype=tf.int64,
            minval=0,
            maxval=self._current_size)
        return tf.gather(self._buffer, indices, axis=0)
