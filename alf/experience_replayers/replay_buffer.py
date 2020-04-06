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

import functools
import gin
from multiprocessing import Lock
import torch
import torch.nn as nn

import alf
from alf.layers import BatchSquash


def _convert_device(nests):
    """Convert the device of the tensors in nests to default device."""

    def _convert_cuda(tensor):
        if tensor.device.type != 'cuda':
            return tensor.cuda()
        else:
            return tensor

    def _convert_cpu(tensor):
        if tensor.device.type != 'cpu':
            return tensor.cpu()
        else:
            return tensor

    d = alf.get_default_device()
    if d == 'cpu':
        return alf.nest.map_structure(_convert_cpu, nests)
    elif d == 'cuda':
        return alf.nest.map_structure(_convert_cuda, nests)
    else:
        raise NotImplementedError("Unknown device %s" % d)


def atomic(func):
    """Make class member function atomic by checking class._lock."""

    def atomic_deco(func):
        @functools.wraps(func)
        def atomic_wrapper(self, *args, **kwargs):
            lock = getattr(self, '_lock')
            if lock:
                with lock:
                    return func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return atomic_wrapper

    return atomic_deco(func)


class RingBuffer(nn.Module):
    """Batched Ring Buffer.

    To be made multiprocessing safe, optionally.
    Can be used to implement ReplayBuffer and Queue.

    Different from tf_agents.replay_buffers.tf_uniform_replay_buffer, this
    replay buffer allows users to specify the environment id when adding batch.
    Thus, multiple actors can store experience in the same buffer.
    """

    def __init__(self,
                 data_spec,
                 num_environments,
                 max_length=1024,
                 device="cpu",
                 allow_multiprocess=False,
                 name="RingBuffer"):
        """
        Args:
            data_spec (nested TensorSpec): spec describing a single item that
                can be stored in this buffer.
            num_environments (int): number of environments or total batch size.
            max_length (int): The maximum number of items that can be stored
                for a single environment.
            device (str): A torch device to place the Variables and ops.
            allow_multiprocess (bool): if true, allows multiple processes to
                write and read the buffer asynchronously.
            name (str): name of the replay buffer.
        """
        super().__init__()
        self._name = name
        self._max_length = max_length
        self._num_envs = num_environments
        self._device = device
        self._allow_multiprocess = allow_multiprocess
        if allow_multiprocess:
            self._lock = Lock()
        else:
            self._lock = None

        buffer_id = [0]

        def _create_buffer(tensor_spec):
            buf = tensor_spec.zeros((num_environments, max_length))
            self.register_buffer("_buffer%s" % buffer_id[0], buf)
            buffer_id[0] += 1
            return buf

        with alf.device(self._device):
            self.register_buffer(
                "_current_size",
                torch.zeros(num_environments, dtype=torch.int64))
            # Current *ending* position of data in the buffer
            self.register_buffer(
                "_current_pos", torch.zeros(
                    num_environments, dtype=torch.int64))
            self._buffer = alf.nest.map_structure(_create_buffer, data_spec)
            bs = BatchSquash(batch_dims=2)
            self._flattened_buffer = alf.nest.map_structure(
                lambda x: bs.flatten(x), self._buffer)

    def check_convert_env_ids(self, env_ids):
        with alf.device(self._device):
            if env_ids is None:
                env_ids = torch.arange(self._num_envs)
            else:
                env_ids = env_ids.to(torch.int64)
            env_ids = _convert_device(env_ids)
            return env_ids

    @atomic
    def enqueue(self, batch, env_ids=None):
        """Add a batch of items to the buffer.

        Args:
            batch (Tensor): shape should be [batch_size] + tensor_space.shape
            env_ids (Tensor): If None, batch_size must be num_environments.
                If not None, its shape should be [batch_size]. We assume there
                is no duplicate ids in `env_id`. batch[i] is generated by
                environment env_ids[i].
        """
        batch_size = alf.nest.get_nest_batch_size(batch)
        with alf.device(self._device):
            batch = _convert_device(batch)
            env_ids = self.check_convert_env_ids(env_ids)

            assert len(env_ids.shape) == 1, "env_ids should be an 1D tensor"
            assert batch_size == env_ids.shape[0], (
                "batch and env_ids do not have same length %s vs. %s" %
                (batch_size, env_ids.shape[0]))

            # Make sure that there is no duplicate in `env_id`
            _, env_id_count = torch.unique(env_ids, return_counts=True)
            assert torch.max(env_id_count) == 1, (
                "There are duplicated ids in env_ids %s" % env_ids)

            current_pos = self._current_pos[env_ids]
            indices = env_ids * self._max_length + current_pos
            alf.nest.map_structure(
                lambda buf, bat: buf.__setitem__(indices, bat),
                self._flattened_buffer, batch)

            self._current_pos[env_ids] = (current_pos + 1) % self._max_length
            current_size = self._current_size[env_ids]
            self._current_size[env_ids] = torch.min(
                current_size + 1, torch.tensor(self._max_length))

    @atomic
    def dequeue(self, env_ids=None):
        """Return earliest batch and remove it from buffer.

        Args:
            env_ids (Tensor): If None, batch_size must be num_environments.
                If not None, only return data from these envs. We assume there
                is no duplicate ids in `env_id`. result[i] is generated by
                environment env_ids[i].
        Returns:
            nested Tensors. The shapes are [batch_size, ...]
        """
        with alf.device(self._device):
            env_ids = self.check_convert_env_ids(env_ids)
            assert len(env_ids.shape) == 1, "env_ids should be an 1D tensor"

            current_size = self._current_size[env_ids]
            min_size = current_size.min()
            assert min_size >= 1, (
                "Not all environments have enough data. The smallest data "
                "size is: %s Try storing more data before calling dequeue" %
                min_size)
            pos = self._current_pos[env_ids] - current_size
            pos = pos % self._max_length
            indices = env_ids * self._max_length + pos  # shape [B*1]

            batch_size = env_ids.shape[0]
            result = alf.nest.map_structure(
                lambda buffer: buffer[indices].reshape(
                    batch_size, 1, *buffer.shape[1:]), self._flattened_buffer)

            self._current_size[env_ids] = (current_size - 1)
        return _convert_device(result)

    @atomic
    def clear(self):
        """Clear the whole buffer."""
        self._current_size.fill_(0)
        self._current_pos.fill_(0)

    @property
    def num_environments(self):
        return self._num_envs


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

    def add_batch(self, batch, env_ids=None):
        self.enqueue(batch, env_ids)

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
            indices = env_ids * self._max_length + pos  # [B, T]
            indices = indices.reshape(-1)  # [B*T]
            result = alf.nest.map_structure(
                lambda buffer: buffer[indices].reshape(
                    batch_size, batch_length, *buffer.shape[1:]),
                self._flattened_buffer)
        return _convert_device(result)

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
        return _convert_device(result)

    def dequeue(self, env_ids=None):
        raise NotImplementedError("gather is not compatible with dequeue.")

    @property
    def total_size(self):
        """Total size from all environments."""
        return _convert_device(self._current_size.sum())
