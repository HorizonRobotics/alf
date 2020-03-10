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

import torch
import torch.nn as nn

import alf
from alf.utils import common
from alf.nest import get_nest_batch_size
from alf.tensor_specs import TensorSpec
from alf.experience_replayers.replay_buffer import _convert_device


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
        return _convert_device(self._current_size)

    def add_batch(self, batch):
        """Add a batch of items to the buffer.

        Args:
            batch (Tensor): shape should be [batch_size] + tensor_space.shape
        """
        batch_size = alf.nest.get_nest_batch_size(batch)
        with alf.device(self._device):
            batch = _convert_device(batch)
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
        return _convert_device(result)

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
            indices = _convert_device(indices)
            indices.copy_(
                (indices +
                 (self._current_pos - self._current_size)) % self._capacity)
            result = alf.nest.map_structure(lambda buf: buf[indices],
                                            self._buffer)
        return _convert_device(result)

    def get_all(self):
        return _convert_device(self._buffer)

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
