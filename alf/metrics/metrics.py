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
#
"""A set of metrics.
Converted to PyTorch from the TF version.
https://github.com/tensorflow/agents/blob/master/tf_agents/metrics/tf_metrics.py
"""

from alf.metrics import metric
import torch


class THDeque(torch.nn.Module):
    """Torch (TH) deque backed by torch.tensor storage."""

    def __init__(self, max_len, dtype):
        """Constructor.

        Args:
            max_len (int): maximum length of the deque
            dtype (torch.dtype): dtype of the content of the deque
        """
        super().__init__()
        shape = (max_len, )
        self.dtype = dtype
        self.register_buffer('_max_len',
                             torch.tensor(max_len, dtype=torch.int64))
        self.register_buffer('_buffer', torch.zeros(shape, dtype=dtype))
        self.register_buffer('_head', torch.zeros((), dtype=torch.int64))

    @property
    def data(self):
        return self._buffer[:self.length]

    def append(self, value):
        """Appends value to deque.
        Overwrites earliest value if deque exceeds max_len.

        Args:
            value (scalar or tensor of dtype): input value to collect

        Returns:
            Nothing
        """
        position = torch.remainder(self._head, self._max_len)
        self._buffer[position] = value
        self._head.add_(1)

    @property
    def length(self):
        return torch.min(self._head, self._max_len)

    def clear(self):
        self._head.fill_(0)
        self._buffer.fill_(0)

    def mean(self):
        if self._head == 0:
            return torch.zeros((), dtype=self.dtype)
        return torch.mean(self._buffer[:self.length])


class EnvironmentSteps(metric.StepMetric):
    """Counts the number of steps taken in the environment."""

    def __init__(self,
                 name='EnvironmentSteps',
                 prefix='Metrics',
                 dtype=torch.int64):
        super().__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.register_buffer('_environment_steps',
                             torch.zeros(1, dtype=self.dtype))

    def call(self, exp):
        """Increase the number of environment_steps according to exp.
        Step count is not increased on exp.is_first() since that step
        is not part of any episode.

        Args:
            exp (alf.data_structures.Experience): batched tensor
        Returns:
            The arguments, for easy chaining.
        """
        steps = (torch.logical_not(exp.is_first())).type(self.dtype)
        num_steps = torch.sum(steps)
        self._environment_steps.add_(num_steps)
        return exp

    def result(self):
        return self._environment_steps

    def reset(self):
        self._environment_steps.fill_(0)


class NumberOfEpisodes(metric.StepMetric):
    """Counts the number of episodes in the environment."""

    def __init__(self,
                 name='NumberOfEpisodes',
                 prefix='Metrics',
                 dtype=torch.int64):
        super(NumberOfEpisodes, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.register_buffer('_number_episodes',
                             torch.zeros(1, dtype=self.dtype))

    def call(self, exp):
        """Increase the number of number_episodes according to exp.
        It would increase for all exp.is_last().

        Args:
            exp (alf.data_structures.Experience): batched tensor
        Returns:
            The arguments, for easy chaining.
        """
        episodes = exp.is_last().type(self.dtype)
        num_episodes = torch.sum(episodes)
        self._number_episodes.add_(num_episodes)
        return exp

    def result(self):
        return self._number_episodes

    def reset(self):
        self._number_episodes.fill_(0)


class AverageReturnMetric(metric.StepMetric):
    """Metric to compute the average return."""

    def __init__(self,
                 name='AverageReturn',
                 prefix='Metrics',
                 dtype=torch.float32,
                 batch_size=1,
                 buffer_size=10):
        super(AverageReturnMetric, self).__init__(name=name, prefix=prefix)
        self._buffer = THDeque(max_len=buffer_size, dtype=dtype)
        self.dtype = dtype
        self.register_buffer('_return_accumulator',
                             torch.zeros(batch_size, dtype=dtype))

    def call(self, exp):
        """Accumulates returns from exp batched tensor.
        It would accumulate all exp.reward.

        Args:
            exp (alf.data_structures.Experience): batched tensor
        Returns:
            The arguments, for easy chaining.
        """
        # Zero out batch indices where a new episode is starting.
        self._return_accumulator[:] = torch.where(
            exp.is_first(), torch.zeros_like(self._return_accumulator),
            self._return_accumulator)

        # Update accumulator with received rewards.
        # Ignores first step whose reward comes from the boundary transition
        # of the last step from the previous episode.
        self._return_accumulator.add_(
            torch.where(exp.is_first(),
                        torch.zeros_like(self._return_accumulator),
                        exp.reward))

        # Add final returns to buffer.
        last_episode_indices = torch.squeeze(
            *torch.where(exp.is_last())).type(torch.int64)
        for indx in last_episode_indices:
            self._buffer.append(self._return_accumulator[indx])

        return exp

    def result(self):
        return self._buffer.mean()

    def reset(self):
        self._buffer.clear()
        self._return_accumulator.fill_(0)


class AverageEpisodeLengthMetric(metric.StepMetric):
    """Metric to compute the average episode length."""

    def __init__(self,
                 name='AverageEpisodeLength',
                 prefix='Metrics',
                 dtype=torch.float32,
                 batch_size=1,
                 buffer_size=10):
        super(AverageEpisodeLengthMetric, self).__init__(
            name=name, prefix=prefix)
        self._buffer = THDeque(max_len=buffer_size, dtype=dtype)
        self.dtype = dtype
        self.register_buffer('_length_accumulator',
                             torch.zeros(batch_size, dtype=dtype))

    def call(self, exp):
        """Accumulates the number of episode steps according to batched exp.
        It would increase for all non exp.is_first().  The first exp
        is the boundary step and needs to be ignored, different from tf_agents.

        Args:
            exp (alf.data_structures.Experience): batched tensor
        Returns:
            The arguments, for easy chaining.
        """
        # Each non-boundary exp (mid or last) represents a step.
        is_first = exp.is_first()
        non_boundary_indices = torch.squeeze(*torch.where(~is_first)).type(
            torch.int64)
        self._length_accumulator.scatter_add_(0, non_boundary_indices,
                                              torch.ones_like(exp.reward))

        # Add lengths to buffer when we hit end of episode
        is_last = exp.is_last()
        last_indices = torch.squeeze(*torch.where(is_last)).type(torch.int64)
        for indx in last_indices:
            self._buffer.append(self._length_accumulator[indx])

        # Clear length accumulator at the end of episodes.
        self._length_accumulator[last_indices] = torch.zeros_like(
            last_indices, dtype=self.dtype)

        return exp

    def result(self):
        return self._buffer.mean()

    def reset(self):
        self._buffer.clear()
        self._length_accumulator.fill_(0)
