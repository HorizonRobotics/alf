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
from multiprocessing import Event, RLock
import time

import torch
import torch.nn as nn

import alf
from alf.nest import get_nest_batch_size
from alf.tensor_specs import TensorSpec
from alf.nest.utils import convert_device


def atomic(func):
    """Make class member function atomic by checking ``class._lock``.

    Can only be applied on class methods, whose containing class
    must have ``_lock`` set to ``None`` or a ``multiprocessing.Lock`` object.

    Args:
        func (callable): the function to be wrapped.

    Returns:
        the wrapped function
    """

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

    Multiprocessing safe, optionally via: ``allow_multiprocess`` flag, blocking
    modes to ``enqueue`` and ``dequeue``, a stop event to terminate blocked
    processes, and putting buffer into shared memory.

    This is the underlying implementation of ``ReplayBuffer`` and ``Queue``.

    Different from ``tf_agents.replay_buffers.tf_uniform_replay_buffer``, this
    buffer allows users to specify the environment id when adding batch.
    Thus, multiple actors can store experience in the same buffer.

    Once stop event is set, all blocking ``enqueue`` and ``dequeue`` calls that
    happen afterwards will be skipped, unless the operation already started.

    Terminology: we use ``pos`` as in ``_current_pos`` to refer to the always
    increasing position of an element in the infinitely long buffer, and ``idx``
    as the actual index of the element in the underlying store (``_buffer``).
    That means ``idx == pos % _max_length`` is always true, and one should use
    ``_buffer[idx]`` to retrieve the stored data.
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
            allow_multiprocess (bool): if ``True``, allows multiple processes
                to write and read the buffer asynchronously.
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
        self._env_ids = torch.arange(self._num_envs, device=device)

        def _create_buffer(spec_path, tensor_spec):
            buf = tensor_spec.zeros((num_environments, max_length))
            if spec_path != '':
                # buffer name cannot contain '.', which is used as the delimiter
                # by ``py_map_structure_with_path`` in the generated path
                spec_name = spec_path.replace('.', '|')
                self.register_buffer(spec_name, buf)
            else:
                self.register_buffer("_buffer%s" % buffer_id[0], buf)
                buffer_id[0] += 1
            return buf

        with alf.device(self._device):
            self.register_buffer(
                "_current_size",
                torch.zeros(num_environments, dtype=torch.int64))
            # Current *ending* positions of data in the buffer without modulo.
            # The next experience will be stored at this position after modulo.
            # These pos always increases. To get the index in the RingBuffer,
            # use ``circular()``, e.g. ``last_idx = self.circular(pos - 1)``.
            self.register_buffer(
                "_current_pos", torch.zeros(
                    num_environments, dtype=torch.int64))

            self._buffer = alf.nest.py_map_structure_with_path(
                _create_buffer, data_spec)

        if allow_multiprocess:
            self.share_memory()

    @property
    def device(self):
        """The device where the data is stored in."""
        return self._device

    def circular(self, pos):
        """Mod pos by _max_length to get the actual index in the _buffer."""
        return pos % self._max_length

    def check_convert_env_ids(self, env_ids):
        with alf.device(self._device):
            if env_ids is None:
                env_ids = torch.arange(self._num_envs)
            else:
                env_ids = env_ids.to(torch.int64)
            env_ids = convert_device(env_ids)
            assert len(env_ids.
                       shape) == 1, "env_ids {}, should be a 1D tensor".format(
                           env_ids.shape)
            return env_ids

    def has_space(self, env_ids):
        """Check free space for one batch of data for env_ids.

        Args:
            env_ids (Tensor): Assumed not ``None``, properly checked by
                ``check_convert_env_ids()``.
        Returns:
            bool
        """
        current_size = self._current_size[env_ids]
        max_size = current_size.max()
        return max_size < self._max_length

    def enqueue(self, batch, env_ids=None, blocking=False):
        """Add a batch of items to the buffer.

        Note, when ``blocking == False``, it always succeeds, overwriting
        oldest data if there is no free slot.

        Args:
            batch (Tensor): of shape ``[batch_size] + tensor_spec.shape``
            env_ids (Tensor): If ``None``, ``batch_size`` must be
                ``num_environments``. If not ``None``, its shape should be
                ``[batch_size]``. We assume there are no duplicate ids in
                ``env_id``. ``batch[i]`` is generated by environment
                ``env_ids[i]``.
            blocking (bool): If ``True``, blocks if there is no free slot to add
                data.  If ``False``, enqueue can overwrite oldest data.

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
            batch (Tensor): shape should be
                ``[batch_size] + tensor_spec.shape``.
            env_ids (Tensor): If ``None``, ``batch_size`` must be
                ``num_environments``. If not ``None``, its shape should be
                ``[batch_size]``. We assume there are no duplicate ids in
                ``env_id``. ``batch[i]`` is generated by environment
                ``env_ids[i]``.
        """
        batch_size = alf.nest.get_nest_batch_size(batch)
        with alf.device(self._device):
            batch = convert_device(batch)
            env_ids = self.check_convert_env_ids(env_ids)
            assert batch_size == env_ids.shape[0], (
                "batch and env_ids do not have same length %s vs. %s" %
                (batch_size, env_ids.shape[0]))

            current_pos = self._current_pos[0].item()
            if (env_ids.numel() == self._num_envs
                    and (self._current_pos == current_pos).all()
                    and (env_ids == self._env_ids).all()):
                # fast path for the common case when all envs have the same
                # current_pos, and when all envs unroll the same number of steps.
                pos = self.circular(current_pos)

                def _set(buf, bat):
                    buf[:, pos] = bat.detach()

                alf.nest.map_structure(_set, self._buffer, batch)
                self._current_pos.fill_(current_pos + 1)
                self._current_size.fill_(
                    min(current_pos + 1, self._max_length))
            else:
                # Make sure that there is no duplicate in `env_id`
                # torch.unique(env_ids, return_counts=True)[1] is the counts for each unique item
                assert torch.unique(
                    env_ids, return_counts=True)[1].max() == 1, (
                        "There are duplicated ids in env_ids %s" % env_ids)

                current_pos = self._current_pos[env_ids]
                indices = env_ids * self._max_length + self.circular(
                    current_pos)
                alf.nest.map_structure(
                    lambda buf, bat: buf.view(-1, *buf.shape[2:]).__setitem__(
                        indices, bat.detach()), self._buffer, batch)

                self._current_pos[env_ids] += 1
                current_size = self._current_size[env_ids]
                self._current_size[env_ids] = torch.clamp(
                    current_size + 1, max=self._max_length)
            # set flags if they exist to unblock potential consumers
            if self._enqueued:
                self._enqueued.set()
                self._dequeued.clear()

    def has_data(self, env_ids, n=1):
        """Check ``n`` steps of data available for ``env_ids``.

        Args:
            env_ids (Tensor): Assumed not ``None``, properly checked by
                ``check_convert_env_ids()``.
            n (int): Number of time steps to check.
        Returns:
            bool
        """
        current_size = self._current_size[env_ids]
        min_size = current_size.min()
        return min_size >= n

    def dequeue(self, env_ids=None, n=1, blocking=False):
        """Return earliest ``n`` steps and mark them removed in the buffer.

        Args:
            env_ids (Tensor): If None, ``batch_size`` must be num_environments.
                If not None, dequeue from these environments. We assume there
                is no duplicate ids in ``env_id``. ``result[i]`` will be from
                environment ``env_ids[i]``.
            n (int): Number of steps to dequeue.
            blocking (bool): If ``True``, blocks if there is not enough data to
                dequeue.
        Returns:
            nested Tensors or None when blocking dequeue gets terminated by
            stop event. The shape of the Tensors is ``[batch_size, n, ...]``.
        Raises:
            AssertionError: when not enough data is present, in non-blocking
            mode.
        """
        assert n <= self._max_length
        if blocking:
            assert self._allow_multiprocess, [
                "Set allow_multiprocess", "to enable blocking mode."
            ]
            env_ids = self.check_convert_env_ids(env_ids)
            while not self._stop.is_set():
                with self._lock:
                    if self.has_data(env_ids, n):
                        return self._dequeue(env_ids=env_ids, n=n)
                # The wait here is outside the lock, so multiple dequeue and
                # enqueue could theoretically happen before the wait.  The
                # wait only acts as a more responsive sleep, and the return
                # value is not used.  We anyways need to check has_data after
                # the wait timed out.
                self._enqueued.wait(timeout=0.2)
            return None
        else:
            return self._dequeue(env_ids=env_ids, n=n)

    @atomic
    def _dequeue(self, env_ids=None, n=1):
        """Return earliest ``n`` steps and mark them removed in the buffer.

        Args:
            env_ids (Tensor): If None, ``batch_size`` must be num_environments.
                If not None, dequeue from these environments. We assume there
                is no duplicate ids in ``env_id``. ``result[i]`` will be from
                environment env_ids[i].
            n (int): Number of steps to dequeue.
        Returns:
            nested Tensors of shape ``[batch_size, n, ...]``.
        Raises:
            AssertionError: when not enough data is present.
        """
        with alf.device(self._device):
            env_ids = self.check_convert_env_ids(env_ids)
            current_size = self._current_size[env_ids]
            min_size = current_size.min()
            assert min_size >= n, (
                "Not all environments have enough data. The smallest data "
                "size is: %s Try storing more data before calling dequeue" %
                min_size)
            batch_size = env_ids.shape[0]
            pos = self._current_pos[env_ids] - current_size  # mod done later
            b_indices = env_ids.reshape(batch_size, 1).expand(-1, n)
            t_range = torch.arange(n).reshape(1, -1)
            t_indices = self.circular(pos.reshape(batch_size, 1) + t_range)
            result = alf.nest.map_structure(
                lambda b: b[(b_indices, t_indices)], self._buffer)
            self._current_size[env_ids] = current_size - n
            # set flags if they exist to unblock potential consumers
            if self._dequeued:
                self._dequeued.set()
                self._enqueued.clear()
        return convert_device(result)

    @atomic
    def remove_up_to(self, n, env_ids=None):
        """Mark as removed earliest up to ``n`` steps.

        Args:
            n (int): max number of steps to mark removed from buffer.
        """
        with alf.device(self._device):
            env_ids = self.check_convert_env_ids(env_ids)
            n = torch.min(
                torch.as_tensor([n] * self._num_envs), self._current_size)
            self._current_size[env_ids] = self._current_size[env_ids] - n

    @atomic
    def clear(self, env_ids=None):
        """Clear the buffer.

        Args:
            env_ids (Tensor): optional list of environment ids to clear
        """
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
        be skipped (return ``None`` for dequeue or ``False`` for enqueue),
        unless the operation already started.
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

    def get_earliest_position(self, env_ids):
        """The earliest position that is still in the replay buffer.

        Args:
            env_ids (Tensor): int64 Tensor of environment ids
        Returns:
            Tensor with the same shape as ``env_ids``, whose each entry is the
            earliest position that is still in the replay buffer for
            corresponding environment.
        """
        return self._current_pos[env_ids] - self._current_size[env_ids]

    def get_current_position(self):
        """Get the current position for each environment.

        Returns:
            Tensor: with shape [num_environments].
        """
        return self._current_pos


class DataBuffer(RingBuffer):
    """A simple circular buffer supporting random sampling. This buffer doesn't
    preserve temporality as data from multiple environments will be arbitrarily
    stored.

    Not multiprocessing safe.
    """

    def __init__(self,
                 data_spec: TensorSpec,
                 capacity,
                 device='cpu',
                 name="DataBuffer"):
        """
        Args:
            data_spec (nested TensorSpec): spec for the data item (without batch
                dimension) to be stored.
            capacity (int): capacity of the buffer.
            device (str): which device to store the data
            name (str): name of the buffer
        """
        super().__init__(
            data_spec=data_spec,
            num_environments=1,
            max_length=capacity,
            device=device,
            allow_multiprocess=False,
            name=name)
        self._capacity = torch.as_tensor(
            self._max_length, dtype=torch.int64, device=device)
        self._derived_buffer = alf.nest.map_structure(lambda buf: buf[0],
                                                      self._buffer)

    def add_batch(self, batch):
        r"""Add a batch of items to the buffer.

        Add batch_size items along the length of the underlying RingBuffer,
        whereas RingBuffer.enqueue only adds data of length 1.
        Truncates the data if ``batch_size > capacity``.

        Args:
            batch (Tensor): of shape ``[batch_size] + tensor_spec.shape``
        """
        batch_size = alf.nest.get_nest_batch_size(batch)
        with alf.device(self._device):
            batch = convert_device(batch)
            n = torch.clamp(self._capacity, max=batch_size)
            current_pos = self.current_pos
            current_size = self.current_size
            indices = self.circular(torch.arange(current_pos, current_pos + n))
            alf.nest.map_structure(
                lambda buf, bat: buf.__setitem__(indices, bat[-n:].detach()),
                self._derived_buffer, batch)

            current_pos.copy_(current_pos + n)
            current_size.copy_(torch.min(current_size + n, self._capacity))

    def get_batch(self, batch_size):
        r"""Get batsh_size random samples in the buffer.

        Args:
            batch_size (int): batch size
        Returns:
            Tensor of shape ``[batch_size] + tensor_spec.shape``
        """
        with alf.device(self._device):
            indices = torch.randint(
                low=0,
                high=self.current_size,
                size=(batch_size, ),
                dtype=torch.int64)
            result = self.get_batch_by_indices(indices)
        return convert_device(result)

    def get_batch_by_indices(self, indices):
        r"""Get the samples by indices

        index=0 corresponds to the earliest added sample in the DataBuffer.

        Args:
            indices (Tensor): indices of the samples

        Returns:
            Tensor:
            Tensor of shape ``[batch_size] + tensor_spec.shape``, where
            ``batch_size`` is ``indices.shape[0]``
        """
        with alf.device(self._device):
            indices = convert_device(indices)
            indices = self.circular(indices + self.current_pos -
                                    self.current_size)
            result = alf.nest.map_structure(lambda buf: buf[indices],
                                            self._derived_buffer)
        return convert_device(result)

    def is_full(self):
        return (self.current_size == self._capacity).cpu().numpy()

    @property
    def current_size(self):
        return self._current_size[0]

    @property
    def current_pos(self):
        return self._current_pos[0]

    def get_all(self):
        return convert_device(
            alf.nest.map_structure(lambda buf: buf, self._derived_buffer))
