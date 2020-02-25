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
"""Replay buffer."""

import gin
import torch.nn as nn

import alf


@gin.configurable
class ReplayBuffer(nn.Module):
    """Replay buffer.

    Different from tf_agents.replay_buffers.tf_uniform_replay_buffer, this
    replay buffer allow users to specify the environment id when adding batch.
    """

    def __init__(self,
                 data_spec,
                 num_environments,
                 max_length=1024,
                 device="cpu:*",
                 name="ReplayBuffer"):
        """Create a ReplayBuffer.

        Args:
            data_spec (nested TensorSpec): spec describing a single item that
                can be stored in this buffer.
            num_environments (int): number of environments.
            max_length (int): The maximum number of items that can be stored
                for a single environment.
            device (str): A TensorFlow device to place the Variables and ops.
            name (str): name of the replay buffer.
        """
        super().__init__(name=name)
        self._max_length = max_length
        self._num_envs = num_environments
        self._device = device

        def _create_buffer(tensor_spec):
            shape = [num_environments, max_length
                     ] + tensor_spec.shape.as_list()
            return tf.Variable(
                name=name + "/buffer",
                initial_value=tf.zeros(shape, dtype=tensor_spec.dtype),
                trainable=False)

        with tf.device(self._device):
            self._current_size = tf.Variable(
                name=name + "/size",
                initial_value=tf.zeros((num_environments, ), tf.int64),
                trainable=False)
            self._current_pos = tf.Variable(
                name=name + "/pos",
                initial_value=tf.zeros((num_environments, ), tf.int64),
                trainable=False)
            self._buffer = tf.nest.map_structure(_create_buffer, data_spec)
            # TF 2.0 checkpoint does not handle tuple. We have to convert
            # _buffer to a flattened list in order to make the checkpointer save the
            # content in self._buffer. This seems to be fixed in
            # tf-nightly-gpu 2.1.0.dev20191119
            # TODO: remove this after upgrading tensorflow
            self._flattened_buffer = tf.nest.flatten(self._buffer)

    def add_batch(self, batch, env_ids=None):
        """Add a batch of items to the buffer.

        Args:
            batch (Tensor): shape should be [batch_size] + tensor_space.shape
            env_ids (Tensor): If None, batch_size must be num_environments.
                If not None, its shape should be [batch_size]. We assume there
                is no duplicate ids in `env_id`. batch[i] is generated by
                environment env_ids[i].
        """
        batch_size = get_nest_batch_size(batch, tf.int32)
        with tf.device(self._device):
            if env_ids is None:
                env_ids = tf.range(self._num_envs)

            assert len(
                env_ids.shape.as_list()) == 1, "env_ids should be an 1D tensor"
            tf.Assert(batch_size == tf.shape(env_ids)[0], [
                "batch and env_ids do not have same length", batch_size, "vs.",
                tf.shape(env_ids)[0]
            ])

            # Make sure that there is no duplicate in `env_id`
            _, _, env_id_count = tf.unique_with_counts(tf.sort(env_ids))
            tf.Assert(
                tf.reduce_max(env_id_count) == 1,
                ["There are duplicated ids in env_ids", env_ids])
            current_pos = tf.gather(self._current_pos, env_ids, axis=0)
            indices = tf.concat([
                tf.cast(tf.expand_dims(env_ids, -1), tf.int64),
                tf.expand_dims(current_pos, -1)
            ],
                                axis=-1)

            tf.nest.map_structure(
                lambda buf, bat: buf.scatter_nd_update(indices, bat),
                self._buffer, batch)

            self._current_pos.scatter_nd_update(
                tf.expand_dims(env_ids, -1),
                (current_pos + 1) % self._max_length)
            current_size = tf.gather(self._current_size, env_ids, axis=0)
            self._current_size.scatter_nd_update(
                tf.expand_dims(env_ids, -1),
                tf.minimum(current_size + 1, self._max_length))

    def get_batch(self, batch_size, batch_length):
        """Randomly get `batch_size` trajectories from the buffer.

        Note: The environments where the sampels are from are ordered in the
            returned batch.

        Args:
            batch_size (int): get so many trajectories
            batch_length (int): the length of each trajectory
        Returns:
            nested Tensors. The shapes are [batch_size, batch_length, ...]
        """
        with tf.device(self._device):
            min_size = tf.reduce_min(self._current_size)
            tf.Assert(min_size >= batch_length, [
                "Not all environments has enough data. The smallest data "
                "size is: ", min_size, "Try storing more data before "
                "calling get_batch"
            ])

            batch_size_per_env = batch_size // self._num_envs
            remaining = batch_size % self._num_envs
            if batch_size_per_env > 0:
                env_ids = tf.tile(
                    tf.range(self._num_envs, dtype=tf.int64),
                    [batch_size_per_env])
            else:
                env_ids = tf.zeros((0, ), tf.int64)
            if remaining > 0:
                eids = tf.range(self._num_envs, dtype=tf.int64)
                eids = tf.random.shuffle(eids)[:remaining]
                env_ids = tf.concat([env_ids, eids], axis=0)

            r = tf.random.uniform(tf.shape(env_ids))
            num_positions = self._current_size - batch_length + 1
            num_positions = tf.gather(num_positions, env_ids)
            pos = tf.cast(r * tf.cast(num_positions, tf.float32), tf.int64)
            pos += tf.gather(self._current_pos - self._current_size, env_ids)
            pos = tf.reshape(pos, [-1, 1])  # [B, 1]
            pos = pos + tf.expand_dims(
                tf.range(batch_length, dtype=tf.int64), axis=0)  # [B, T]
            pos = pos % self._max_length
            pos = tf.expand_dims(pos, -1)  # [B, T, 1]
            env_ids = tf.reshape(env_ids, [-1, 1])  # [B, 1]
            env_ids = tf.tile(env_ids, [1, batch_length])  # [B, T]
            env_ids = tf.reshape(env_ids, [-1, batch_length, 1])  # [B, T, 1]
            indices = tf.concat([env_ids, pos], axis=-1)  # [B, T, 2]
            return tf.nest.map_structure(
                lambda buffer: tf.gather_nd(buffer, indices), self._buffer)

    def clear(self):
        with tf.device(self._device):
            self._current_size.assign(tf.zeros_like(self._current_size))
            self._current_pos.assign(tf.zeros_like(self._current_pos))

    def gather_all(self):
        """Returns all the items in buffer.

        Returns:
            Returns all the items currently in the buffer. The shapes of the
            tensors are [B, T, ...] where B=num_environments, T=current_size
        Raises:
            tf.errors.InvalidArgumentError: if the current_size is not same for
                all the environments
        """
        size = tf.reduce_min(self._current_size)
        max_size = tf.reduce_max(self._current_size)
        tf.Assert(size == max_size, [
            "Not all environment have the same size. min_size:", size,
            "max_size:", max_size
        ])

        if size == self._max_length:
            return tf.nest.map_structure(lambda buf: buf.value(), self._buffer)
        else:
            return tf.nest.map_structure(lambda buf: buf[:, :size, ...],
                                         self._buffer)

    @property
    def num_environments(self):
        return self._num_envs
