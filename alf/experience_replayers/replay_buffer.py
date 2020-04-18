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
from alf.nest.utils import convert_device
from alf.utils.data_buffer import atomic, RingBuffer


class ReplayBuffer(RingBuffer):
    """Replay buffer with RingBuffer as implementation.
    """

    def __init__(self,
                 data_spec,
                 num_environments,
                 max_length=1024,
                 device="cpu",
                 allow_multiprocess=False,
                 name="ReplayBuffer"):
        super().__init__(
            data_spec,
            num_environments,
            max_length=max_length,
            device=device,
            allow_multiprocess=allow_multiprocess,
            name=name)

    def add_batch(self, batch, env_ids=None, blocking=False):
        self.enqueue(batch, env_ids, blocking=blocking)

    @atomic
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
        with alf.device(self._device):
            min_size = self._current_size.min()
            assert min_size >= batch_length, (
                "Not all environments have enough data. The smallest data "
                "size is: %s Try storing more data before calling get_batch" %
                min_size)

            batch_size_per_env = batch_size // self._num_envs
            remaining = batch_size % self._num_envs
            if batch_size_per_env > 0:
                env_ids = torch.arange(
                    self._num_envs).repeat(batch_size_per_env)
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
            pos = pos.reshape(-1, 1)  # [B, 1]
            pos = pos + torch.arange(batch_length).unsqueeze(0)  # [B, T]
            pos = pos % self._max_length
            env_ids = env_ids.reshape(-1, 1).repeat(1, batch_length)  # [B, T]
            result = alf.nest.map_structure(lambda b: b[(env_ids, pos)],
                                            self._buffer)
        return convert_device(result)

    @atomic
    def gather_all(self):
        """Returns all the items in the buffer.

        Returns:
            Tensors of shape [B, T, ...], B=num_environments, T=current_size
        Raises:
            AssertionError: if the current_size is not same for
                all the environments
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
