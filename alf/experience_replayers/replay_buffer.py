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
import torch
import torch.nn as nn

import alf


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
                 device="cpu",
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
        super().__init__()
        self._name = name
        self._max_length = max_length
        self._num_envs = num_environments
        self._device = device

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
            self.register_buffer(
                "_current_pos", torch.zeros(
                    num_environments, dtype=torch.int64))
            self._buffer = alf.nest.map_structure(_create_buffer, data_spec)
            self._flattened_buffer = alf.nest.map_structure(
                lambda x: x.view(-1, *x.shape[2:]), self._buffer)

    def add_batch(self, batch, env_ids=None):
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
            env_ids = _convert_device(env_ids)
            batch = _convert_device(batch)
            if env_ids is None:
                env_ids = torch.arange(self._num_envs)
            else:
                env_ids = env_ids.to(torch.int64)

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
            print(self._device)
            print(self._current_size)
            min_size = self._current_size.min()
            assert min_size >= batch_length, (
                "Not all environments has enough data. The smallest data "
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

    def clear(self):
        """Clear the replay bufer."""
        self._current_size.fill_(0)
        self._current_pos.fill_(0)

    def gather_all(self):
        """Returns all the items in buffer.

        Returns:
            Returns all the items currently in the buffer. The shapes of the
            tensors are [B, T, ...] where B=num_environments, T=current_size
        Raises:
            AssertionError: if the current_size is not same for
                all the environments
        """
        size = self._current_size.min()
        max_size = self._current_size.max()
        assert size == max_size, (
            "Not all environment have the same size. min_size: %s "
            "max_size: %s" % (size, max_size))

        if size == self._max_length:
            result = self._buffer
        else:
            result = alf.nest.map_structure(lambda buf: buf[:, :size, ...],
                                            self._buffer)
        return _convert_device(result)

    @property
    def num_environments(self):
        return self._num_envs

    def total_size(self):
        """Total size from all environments."""
        return _convert_device(self._current_size.sum())
