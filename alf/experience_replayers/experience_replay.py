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
"""
NOTE: The APIs in this file are subject to changes when we implement generic replay
buffers for off-policy drivers in the future.
"""

import abc
import six
import torch
from torch import nn

import alf
from alf.experience_replayers.replay_buffer import ReplayBuffer

from alf.utils import common


@six.add_metaclass(abc.ABCMeta)
class ExperienceReplayer(nn.Module):
    """
    Base class for implementing experience storing and replay. A subclass should
    implement the abstract functions. This class object will be used by OffPolicyDrivers
    after training data are accumulated for a certain amount of time.
    """

    @abc.abstractmethod
    def observe(self, exp):
        """
        Observe a batch of `exp`, potentially storing it to replay buffers.

        Args:
            exp (Experience): each item has the shape of (`num_envs`, `env_batch_size`,
                `unroll_length`, ...), where `num_envs` is the number of tf_agents
                *batched* environments, each of which contains `env_batch_size`
                independent & parallel single environments.
        """

    @abc.abstractmethod
    def replay(self, sample_batch_size, mini_batch_length):
        """Replay experiences from buffers

        Args:
            sample_batch_size (int): A batch size to specify the number of items to
                return. A batch of `sample_batch_size` items is returned, where each
                tensor in items will have its first dimension equal to sample_batch_size
                and the rest of the dimensions match the corresponding data_spec.
            mini_batch_length (int): the temporal length of each sample

        Output:
            tuple:
                - nested Tensors: The samples. Its shapes are [batch_size, batch_length, ...]
                - BatchInfo: Information about the batch. Its shapes are [batch_size].
                    - env_ids: environment id for each sequence
                    - positions: starting position in the replay buffer for each sequence.
                    - importance_weights: importance weight divided by the average of
                        all non-zero importance weights in the buffer.
        """

    @abc.abstractmethod
    def replay_all(self):
        """Replay all experiences

        Output:
            exp (Experience): each item has the shape (`full_batch_size`,
                `buffer_length`, ...)
        """

    @abc.abstractmethod
    def clear(self):
        """Clear all buffers."""

    @abc.abstractmethod
    def batch_size(self):
        """
        Return the buffer's batch_size, assuming all buffers having the same
        batch_size
        """


class SyncExperienceReplayer(ExperienceReplayer):
    """
    For synchronous off-policy training.

    Example algorithms: DDPG, SAC
    """

    def __init__(self,
                 experience_spec,
                 batch_size,
                 max_length,
                 num_earliest_frames_ignored=0,
                 prioritized_sampling=False,
                 name="SyncExperienceReplayer"):
        """Create a ReplayBuffer.

        Args:
            data_experience_specspec (nested TensorSpec): spec describing a
                single item that can be stored in the replayer.
            batch_size (int): number of environments.
            max_length (int): The maximum number of items that can be stored
                for a single environment.
            num_earliest_frames_ignored (int): ignore the earlist so many frames
                when sample from the buffer. This is typically required when
                FrameStack is used.
            prioritized_sampling (bool): Use prioritized sampling if this is True.
        """
        super().__init__()
        self._experience_spec = experience_spec
        self._buffer = ReplayBuffer(
            experience_spec,
            batch_size,
            max_length=max_length,
            prioritized_sampling=prioritized_sampling,
            num_earliest_frames_ignored=num_earliest_frames_ignored,
            name=name)
        self._data_iter = None

    def observe(self, exp):
        """
        For the sync driver, `exp` has the shape (`env_batch_size`, ...)
        with `num_envs`==1 and `unroll_length`==1.
        """
        outer_rank = alf.nest.utils.get_outer_rank(exp, self._experience_spec)

        if outer_rank == 1:
            self._buffer.add_batch(exp, exp.env_id)
        elif outer_rank == 3:
            # The shape is [learn_queue_cap, unroll_length, env_batch_size, ...]
            for q in range(exp.step_type.shape[0]):
                for t in range(exp.step_type.shape[1]):
                    bat = alf.nest.map_structure(lambda x: x[q, t, ...], exp)
                    self._buffer.add_batch(bat, bat.env_id)
        else:
            raise ValueError("Unsupported outer rank %s of `exp`" % outer_rank)

    def replay(self, sample_batch_size, mini_batch_length):
        """Get a random batch.

        Args:
            sample_batch_size (int): number of sequences
            mini_batch_length (int): the length of each sequence
        Returns:
            tuple:
                - nested Tensors: The samples. Its shapes are [batch_size, batch_length, ...]
                - BatchInfo: Information about the batch. Its shapes are [batch_size].
                    - env_ids: environment id for each sequence
                    - positions: starting position in the replay buffer for each sequence.
                    - importance_weights: importance weight divided by the average of
                        all non-zero importance weights in the buffer.

        """
        return self._buffer.get_batch(sample_batch_size, mini_batch_length)

    def replay_all(self):
        return self._buffer.gather_all()

    def clear(self):
        self._buffer.clear()

    def update_priority(self, env_ids, positions, priorities):
        """Update the priorities for the given experiences.

        Args:
            env_ids (Tensor): 1-D int64 Tensor.
            positions (Tensor): 1-D int64 Tensor with same shape as ``env_ids``.
                This position should be obtained the BatchInfo returned by
                ``get_batch()``
        """
        self._buffer.update_priority(env_ids, positions, priorities)

    @property
    def batch_size(self):
        return self._buffer.num_environments

    @property
    def total_size(self):
        return self._buffer.total_size

    @property
    def replay_buffer(self):
        return self._buffer
