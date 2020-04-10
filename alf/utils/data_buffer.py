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

import functools
import gin
from multiprocessing import Event, RLock
import time

import torch
import torch.nn as nn

import alf
from alf.layers import BatchSquash
from alf.utils import common
from alf.nest import get_nest_batch_size
from alf.tensor_specs import TensorSpec
from alf.nest.utils import convert_device


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

    Multiprocessing safe, optionally via allow_multiprocess flag, blocking
    modes to enqueue and dequeue, a stop event to terminate blocked
    processes, and putting buffer into shared memory.

    This is the underlying implementation of ReplayBuffer and Queue.

    Different from tf_agents.replay_buffers.tf_uniform_replay_buffer, this
    buffer allows users to specify the environment id when adding batch.
    Thus, multiple actors can store experience in the same buffer.

    Once stop event is set, all blocking enqueue and dequeue calls that
    happen afterwards will be skipped, unless the operation already started.
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
        # allows outside to stop enqueue and dequeue processes from waiting
        self._stop = Event()
        if allow_multiprocess:
            self._lock = RLock()  # re-entrant lock
            # notify a finished dequeue event, so blocked enqueues may start
            self._dequeued = Event()
            self._dequeued.set()
            # notify a finished enqueue event, so blocked dequeues may start
            self._enqueued = Event()
            self._enqueued.clear()
        else:
            self._lock = None
            self._dequeued = None
            self._enqueued = None

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

        if allow_multiprocess:
            self.share_memory()

    def check_convert_env_ids(self, env_ids):
        with alf.device(self._device):
            if env_ids is None:
                env_ids = torch.arange(self._num_envs)
            else:
                env_ids = env_ids.to(torch.int64)
            env_ids = convert_device(env_ids)
            assert len(env_ids.shape) == 1, "env_ids should be a 1D tensor"
            return env_ids

    def has_space(self, env_ids):
        """Check free space for one batch of data for env_ids.

        Args:
            env_ids (Tensor): Assumed not None, properly checked by
                check_convert_env_ids().
        
        Returns:
            bool
        """
        current_size = self._current_size[env_ids]
        max_size = current_size.max()
        return max_size < self._max_length

    def enqueue(self, batch, env_ids=None, blocking=False):
        """Add a batch of items to the buffer.

        Note, when blocking == False, it always succeeds, overwriting oldest
        data if there is no free slot.

        Args:
            batch (Tensor): shape should be [batch_size] + tensor_space.shape
            env_ids (Tensor): If None, batch_size must be num_environments.
                If not None, its shape should be [batch_size]. We assume there
                is no duplicate ids in `env_id`. batch[i] is generated by
                environment env_ids[i].
            blocking (bool): If True, blocks if there is no free slot to add
                data.  If False, enqueue can overwrite oldest data.

        Returns:
            True on success, False only in blocking mode when queue is stopped.
        """
        if blocking:
            assert self._allow_multiprocess, (
                "Set allow_multiprocess to enable blocking mode.")
            env_ids = self.check_convert_env_ids(env_ids)
            while not self._stop.is_set():
                with self._lock:
                    if self.has_space(env_ids):
                        self._enqueue(batch, env_ids)
                        return True
                # The wait here is outside the lock, so multiple dequeue and
                # enqueue could theoretically happen before the wait.  The
                # wait only acts as a more responsive sleep, and the return
                # value is not used.  We anyways need to check has_space after
                # the wait timed out.
                self._dequeued.wait(timeout=0.2)
            return False
        else:
            self._enqueue(batch, env_ids)
            return True

    @atomic
    def _enqueue(self, batch, env_ids=None):
        """Add a batch of items to the buffer (atomic).

        Args:
            batch (Tensor): shape should be [batch_size] + tensor_space.shape
            env_ids (Tensor): If None, batch_size must be num_environments.
                If not None, its shape should be [batch_size]. We assume there
                is no duplicate ids in `env_id`. batch[i] is generated by
                environment env_ids[i].
        """
        batch_size = alf.nest.get_nest_batch_size(batch)
        with alf.device(self._device):
            batch = convert_device(batch)
            env_ids = self.check_convert_env_ids(env_ids)
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
            # set flags if they exist to unblock potential consumers
            if self._enqueued:
                self._enqueued.set()
                self._dequeued.clear()

    def has_data(self, env_ids):
        """Check one batch of data available for env_ids.

        Args:
            env_ids (Tensor): Assumed not None, properly checked by
                check_convert_env_ids().
        
        Returns:
            bool
        """
        current_size = self._current_size[env_ids]
        min_size = current_size.min()
        return min_size > 0

    def dequeue(self, env_ids=None, blocking=False):
        """Return earliest batch and remove it from buffer.

        Args:
            env_ids (Tensor): If None, batch_size must be num_environments.
                If not None, dequeue from these environments. We assume there
                is no duplicate ids in `env_id`. result[i] will be from
                environment env_ids[i].
            blocking (bool): If True, blocks if there is no free slot to add
                data.
        Returns:
            nested Tensors or None when blocking dequeue gets terminated by
            stop event. The shapes of the Tensors are [batch_size, ...].
        """
        if blocking:
            assert self._allow_multiprocess, [
                "Set allow_multiprocess", "to enable blocking mode."
            ]
            env_ids = self.check_convert_env_ids(env_ids)
            while not self._stop.is_set():
                with self._lock:
                    if self.has_data(env_ids):
                        return self._dequeue(env_ids)
                # The wait here is outside the lock, so multiple dequeue and
                # enqueue could theoretically happen before the wait.  The
                # wait only acts as a more responsive sleep, and the return
                # value is not used.  We anyways need to check has_data after
                # the wait timed out.
                self._enqueued.wait(timeout=0.2)
            return None
        else:
            return self._dequeue(env_ids)

    @atomic
    def _dequeue(self, env_ids=None):
        """Return earliest batch and remove it from buffer.

        Args:
            env_ids (Tensor): If None, batch_size must be num_environments.
                If not None, dequeue from these environments. We assume there
                is no duplicate ids in `env_id`. result[i] will be from
                environment env_ids[i].
        Returns:
            nested Tensors. The shapes are [batch_size, ...]
            Raises AssertionError when not enough data is present.
        """
        with alf.device(self._device):
            env_ids = self.check_convert_env_ids(env_ids)
            current_size = self._current_size[env_ids]
            min_size = current_size.min()
            assert min_size >= 1, (
                "Not all environments have enough data. The smallest data "
                "size is: %s Try storing more data before calling dequeue" %
                min_size)
            pos = self._current_pos[env_ids] - current_size
            pos = pos % self._max_length
            indices = env_ids * self._max_length + pos  # shape [B]

            batch_size = env_ids.shape[0]
            result = alf.nest.map_structure(
                lambda buffer: buffer[indices].reshape(
                    batch_size, 1, *buffer.shape[1:]), self._flattened_buffer)

            self._current_size[env_ids] = current_size - 1
            # set flags if they exist to unblock potential consumers
            if self._dequeued:
                self._dequeued.set()
                self._enqueued.clear()
        return convert_device(result)

    @atomic
    def clear(self, env_ids=None):
        """Clear the whole buffer."""
        with alf.device(self._device):
            env_ids = self.check_convert_env_ids(env_ids)
            self._current_size.scatter_(0, env_ids, 0)
            self._current_pos.scatter_(0, env_ids, 0)
            if self._dequeued:
                self._dequeued.set()
                self._enqueued.clear()

    def stop(self):
        """Stop waiting processes from being blocked.
        
        Only checked in blocking mode of dequeue and enqueue.

        All blocking enqueue and dequeue calls that happen afterwards will
           be skipped (return None for dequeue or False for enqueue), unless
           the operation already started.
        """
        self._stop.set()

    def revive(self):
        """Clears the stop Event so blocking mode will start working again.

        Only checked in blocking mode of dequeue and enqueue.
        """
        self._stop.clear()

    @property
    def num_environments(self):
        return self._num_envs


class DataBuffer(nn.Module):
    """A simple circular buffer supporting random sampling. This buffer doesn't
    preserve temporality as data from multiple environments will be arbitrarily
    stored.
    """

    def __init__(self,
                 data_spec: TensorSpec,
                 capacity,
                 device='cpu',
                 name="DataBuffer"):
        """Create a DataBuffer.

        Args:
            data_spec (nested TensorSpec): spec for the data item (without batch
                dimension) to be stored.
            capacity (int): capacity of the buffer.
            device (str): which device to store the data
            name (str): name of the buffer
        """
        super().__init__()
        self._name = name
        self._device = device

        buffer_id = [0]

        def _create_buffer(tensor_spec):
            buf = tensor_spec.zeros((capacity, ))
            self.register_buffer("_buffer%s" % buffer_id[0], buf)
            buffer_id[0] += 1
            return buf

        with alf.device(self._device):
            self._capacity = torch.as_tensor(capacity, dtype=torch.int64)
            self.register_buffer("_current_size",
                                 torch.zeros((), dtype=torch.int64))
            self.register_buffer("_current_pos",
                                 torch.zeros((), dtype=torch.int64))
            self._buffer = alf.nest.map_structure(_create_buffer, data_spec)

    @property
    def current_size(self):
        return convert_device(self._current_size)

    def add_batch(self, batch):
        """Add a batch of items to the buffer.

        Args:
            batch (Tensor): shape should be [batch_size] + tensor_space.shape
        """
        batch_size = alf.nest.get_nest_batch_size(batch)
        with alf.device(self._device):
            batch = convert_device(batch)
            n = torch.min(self._capacity, torch.as_tensor(batch_size))
            indices = torch.arange(self._current_pos,
                                   self._current_pos + n) % self._capacity
            alf.nest.map_structure(
                lambda buf, bat: buf.__setitem__(indices, bat[-n:].detach()),
                self._buffer, batch)

            self._current_pos.copy_((self._current_pos + n) % self._capacity)
            self._current_size.copy_(
                torch.min(self._current_size + n, self._capacity))

    def get_batch(self, batch_size):
        """Get batsh_size random samples in the buffer.

        Args:
            batch_size (int): batch size
        Returns:
            Tensor of shape [batch_size] + tensor_spec.shape
        """
        with alf.device(self._device):
            indices = torch.randint(
                low=0,
                high=self._current_size,
                size=(batch_size, ),
                dtype=torch.int64)
            result = self.get_batch_by_indices(indices)
        return convert_device(result)

    def get_batch_by_indices(self, indices):
        """Get the samples by indices

        index=0 corresponds to the earliest added sample in the DataBuffer.
        Args:
            indices (Tensor): indices of the samples
        Returns:
            Tensor of shape [batch_size] + tensor_spec.shape, where batch_size
                is indices.shape[0]
        """
        with alf.device(self._device):
            indices = convert_device(indices)
            indices.copy_(
                (indices +
                 (self._current_pos - self._current_size)) % self._capacity)
            result = alf.nest.map_structure(lambda buf: buf[indices],
                                            self._buffer)
        return convert_device(result)

    def get_all(self):
        return convert_device(self._buffer)

    def clear(self):
        """Clear the buffer.

        Returns:
            None
        """
        self._current_pos.fill_(0)
        self._current_size.fill_(0)

    def pop(self, n):
        n = torch.min(torch.as_tensor(n), self._current_size)
        self._current_size.cppy_(self._current_size - n)
        self._current_pos.copy_((self._current_pos - n) % self._capacity)
