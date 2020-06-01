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
"""Replay buffer."""

import gin
import torch
import torch.nn as nn

import alf
from alf.data_structures import namedtuple
from alf.nest.utils import convert_device
from alf.utils.common import warning_once
from alf.utils.data_buffer import atomic, RingBuffer

from .segment_tree import SumSegmentTree, MaxSegmentTree, MinSegmentTree

BatchInfo = namedtuple(
    "BatchInfo", ["env_ids", "positions", "importance_weights"],
    default_value=())


class ReplayBuffer(RingBuffer):
    """Replay buffer with RingBuffer as implementation."""

    def __init__(self,
                 data_spec,
                 num_environments,
                 max_length=1024,
                 prioritized_sampling=False,
                 initial_priority=1.0,
                 device="cpu",
                 allow_multiprocess=False,
                 name="ReplayBuffer"):
        """
        If ``prioritized_sampling`` is set to True, instead of sampling experiences
        uniformly from the replay buffer, ``get_batch()`` samples experiences
        with probability proportional to the priority of each experience. The
        initial priority of a new expeirence added to the buffer is set to
        ``initial_priority``. The priorities can be updated using ``update_priority()``.

        Args:
            data_spec (nested TensorSpec): spec describing a single item that
                can be stored in this buffer.
            num_environments (int): number of environments or total batch size.
            max_length (int): The maximum number of items that can be stored
                for a single environment.
            prioritized_sampling (bool): Use prioritized sampling if this is True.
            initial_priority (float): initial priority used for new experiences.
                The actual initial priority used for new experience is the maximum
                of this value and the current maximum priority of all experiences.
            device (str): A torch device to place the Variables and ops.
            allow_multiprocess (bool): if ``True``, allows multiple processes
                to write and read the buffer asynchronously.
            name (str): name of the replay buffer.
        """
        super().__init__(
            data_spec,
            num_environments,
            max_length=max_length,
            device=device,
            allow_multiprocess=allow_multiprocess,
            name=name)

        self._prioritized_sampling = prioritized_sampling
        if prioritized_sampling:
            self._mini_batch_length = 1
            tree_size = self._max_length * num_environments
            self._sum_tree = SumSegmentTree(tree_size, device=device)
            self._max_tree = MaxSegmentTree(tree_size, device=device)
            self._initial_priority = torch.tensor(
                initial_priority, dtype=torch.float32, device=device)

    def add_batch(self, batch, env_ids=None, blocking=False):
        with alf.device(self._device):
            env_ids = self.check_convert_env_ids(env_ids)
            self.enqueue(batch, env_ids, blocking=blocking)
            if self._prioritized_sampling:
                self._initialize_priority(env_ids)

    @property
    def initial_priority(self):
        """The initial priority used for newly added experiences.

        We use a large value for initial priority so that a new experience can
        be used for training sooner. We make it at least 1.0 so that it can never
        be very small.
        """
        return convert_device(
            torch.max(self._max_tree.summary(), self._initial_priority))

    def _initialize_priority(self, env_ids):
        last_pos = (self._current_pos[env_ids] -
                    self._mini_batch_length) % self._max_length
        indices = self._env_id_pos_to_index(env_ids, last_pos)
        values = torch.full(indices.shape, self.initial_priority)

        if self._mini_batch_length > 1:
            last_pos = (self._current_pos[env_ids] - 1) % self._max_length
            indices = torch.cat(
                [indices,
                 self._env_id_pos_to_index(env_ids, last_pos)])
            values = torch.cat(
                [values,
                 torch.zeros_like(last_pos, dtype=torch.float32)])

        self._update_segment_tree(indices, values)

    def _update_segment_tree(self, indices, values):
        self._sum_tree[indices] = values
        self._max_tree[indices] = values

    def _env_id_pos_to_index(self, env_ids, positions):
        """Convert env_id, pos to indices used by SegmentTree."""
        return env_ids * self._max_length + positions

    def _index_to_env_id_pos(self, indices):
        """Convert indices used by SegmentTree to (env_id, pos)."""
        return indices / self._max_length, indices % self._max_length

    def _change_mini_batch_length(self, mini_batch_length):
        env_ids = torch.arange(self._num_envs, dtype=torch.int64)
        if mini_batch_length > self._mini_batch_length:
            for i in range(self._mini_batch_length, mini_batch_length):
                last_pos = (self._current_pos - i) % self._max_length
                indices = self._env_id_pos_to_index(env_ids, last_pos)
                valid = torch.where(self._current_size >= i)[0]
                indices = indices[valid]
                values = torch.zeros_like(indices, dtype=torch.float32)
                self._update_segment_tree(indices, values)
        elif mini_batch_length < self._mini_batch_length:
            for i in range(mini_batch_length, self._mini_batch_length):
                last_pos = (self._current_pos - i) % self._max_length
                indices = self._env_id_pos_to_index(env_ids, last_pos)
                valid = torch.where(self._current_size >= i)[0]
                indices = indices[valid]
                values = torch.full(indices.shape, self.initial_priority)
                self._update_segment_tree(indices, values)
        self._mini_batch_length = mini_batch_length

    def update_priority(self, env_ids, positions, priorities):
        """Update the priorities for the given experiences.

        Args:
            env_ids (Tensor): 1-D int64 Tensor.
            positions (Tensor): 1-D int64 Tensor with same shape as ``env_ids``.
                This position should be obtained the BatchInfo returned by
                ``get_batch()``
        """
        indices = self._env_id_pos_to_index(env_ids, positions)
        self._update_segment_tree(indices, priorities)

    @atomic
    def get_batch(self, batch_size, batch_length):
        """Randomly get `batch_size` trajectories from the buffer.

        Note: The environments where the sampels are from are ordered in the
            returned batch.

        Args:
            batch_size (int): get so many trajectories
            batch_length (int): the length of each trajectory
        Returns:
            tuple:
                - nested Tensors: The samples. Its shapes are [batch_size, batch_length, ...]
                - BatchInfo: Information about the batch. Its shapes are [batch_size].
                    - env_ids: environment id for each sequence
                    - positions: starting position in the replay buffer for each sequence.
                    - importance_weights: importance weight divided by the average of
                        all non-zero importance weights in the buffer.
        """
        with alf.device(self._device):
            if self._prioritized_sampling:
                env_ids, pos = self._prioritized_sample(
                    batch_size, batch_length)
            else:
                env_ids, pos = self._uniform_sample(batch_size, batch_length)

            info = BatchInfo(env_ids=env_ids, positions=pos)

            pos = pos.reshape(-1, 1)  # [B, 1]
            pos = pos + torch.arange(batch_length).unsqueeze(0)  # [B, T]
            pos = pos % self._max_length
            env_ids = env_ids.reshape(-1, 1).expand(batch_size,
                                                    batch_length)  # [B, T]
            result = alf.nest.map_structure(lambda b: b[(env_ids, pos)],
                                            self._buffer)

            if self._prioritized_sampling:
                indices = self._env_id_pos_to_index(info.env_ids,
                                                    info.positions)
                avg_weight = self._sum_tree.nnz / self._sum_tree.summary()
                info = info._replace(
                    importance_weights=self._sum_tree[indices] * avg_weight)

        return convert_device(result), convert_device(info)

    def _uniform_sample(self, batch_size, batch_length):
        min_size = self._current_size.min()
        assert min_size >= batch_length, (
            "Not all environments have enough data. The smallest data "
            "size is: %s Try storing more data before calling get_batch" %
            min_size)

        batch_size_per_env = batch_size // self._num_envs
        remaining = batch_size % self._num_envs
        if batch_size_per_env > 0:
            env_ids = torch.arange(self._num_envs).repeat(batch_size_per_env)
        else:
            env_ids = torch.zeros(0, dtype=torch.int64)
        if remaining > 0:
            eids = torch.randperm(self._num_envs)[:remaining]
            env_ids = torch.cat([env_ids, eids], dim=0)

        r = torch.rand(*env_ids.shape)

        num_positions = self._current_size - batch_length + 1
        num_positions = num_positions[env_ids]
        pos = (r * num_positions).to(torch.int64)
        pos += (self._current_pos - self._current_size)[env_ids]
        pos = pos % self._max_length

        return env_ids, pos

    def _prioritized_sample(self, batch_size, batch_length):
        if batch_length != self._mini_batch_length:
            if self._mini_batch_length > 1:
                warning_once(
                    "It is not advisable to use different batch_length"
                    "for different calls to get_batch(). Previous batch_length=%d "
                    "new batch_length=%d" % (self._mini_batch_length,
                                             batch_length))
            self._change_mini_batch_length(batch_length)

        assert self._sum_tree.summary() > 0, (
            "There is no data in the "
            "buffer or the data of all the environments are shorter than "
            "batch_length=%s" % batch_length)

        r = torch.rand((batch_size, ))
        r = (r + torch.arange(batch_size, dtype=torch.float32)) / batch_size
        r = r * self._sum_tree.summary()
        indices = self._sum_tree.find_sum_bound(r)
        return self._index_to_env_id_pos(indices)

    @atomic
    def gather_all(self):
        """Returns all the items in the buffer.

        Returns:
            Tensors of shape [B, T, ...], B=num_environments, T=current_size
        Raises:
            AssertionError: if the current_size is not same for all the
            environments.
        """
        size = self._current_size.min()
        max_size = self._current_size.max()
        assert size == max_size, (
            "Not all environments have the same size. min_size: %s "
            "max_size: %s" % (size, max_size))
        if size < self._max_length:
            pos = self._current_pos.min()
            max_pos = self._current_pos.max()
            assert pos == max_pos, (
                "Not all environments have the same ending position. "
                "min_pos: %s max_pos: %s" % (pos, max_pos))
            assert size == pos, (
                "When buffer not full, ending position of the data in the "
                "buffer current_pos coincides with current_size")

        # NOTE: this is not the proper way to gather all from a ring
        # buffer whose data can start from the middle, so this is limited
        # to the case where clear() is the only way to remove data from
        # the buffer.
        if size == self._max_length:
            result = self._buffer
        else:
            # Assumes that non-full buffer always stores data starting from 0
            result = alf.nest.map_structure(lambda buf: buf[:, :size, ...],
                                            self._buffer)
        return convert_device(result)

    def dequeue(self, env_ids=None):
        raise NotImplementedError("gather is not compatible with dequeue.")

    @property
    def total_size(self):
        """Total size from all environments."""
        return convert_device(self._current_size.sum())
