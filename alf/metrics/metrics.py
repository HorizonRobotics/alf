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


class TCDeque(object):
    """Deque backed by torch.tensor storage."""

    def __init__(self, max_len, dtype):
        shape = (max_len, )
        self._dtype = dtype
        self.max_len = torch.tensor(max_len, dtype=torch.int32)
        self.buffer = torch.zeros(shape, dtype=dtype)

        self.head = torch.zeros((), dtype=torch.int32)

    @property
    def data(self):
        return self.buffer[:self.length]

    def append(self, value):
        self.add(value)

    def add(self, value):
        position = torch.remainder(self.head, self.max_len)
        self.buffer[position] = value
        self.head.add_(1)

    @property
    def length(self):
        return torch.min(self.head, self.max_len)

    def clear(self):
        self.head.fill_(0)
        self.buffer.fill_(0)

    def mean(self):
        if self.head == 0:
            return torch.zeros((), dtype=self._dtype)
        return torch.mean(self.buffer[:self.length])


class EnvironmentSteps(metric.StepMetric):
    """Counts the number of steps taken in the environment."""

    def __init__(self,
                 name='EnvironmentSteps',
                 prefix='Metrics',
                 dtype=torch.int64):
        super(EnvironmentSteps, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.environment_steps = torch.zeros(1, dtype=self.dtype)

    def call(self, time_step):
        """Increase the number of environment_steps according to time_step.
        Step count is not increased on time_step.is_first() since that step
        is not part of any episode.

        Args:
            time_step (alf.data_structures.TimeStep): batched tensor
        Returns:
            The arguments, for easy chaining.
        """
        steps = (torch.logical_not(time_step.is_first())).type(self.dtype)
        num_steps = torch.sum(steps)
        self.environment_steps.add_(num_steps)
        return time_step

    def result(self):
        return self.environment_steps

    def reset(self):
        self.environment_steps.fill_(0)


class NumberOfEpisodes(metric.StepMetric):
    """Counts the number of episodes in the environment."""

    def __init__(self,
                 name='NumberOfEpisodes',
                 prefix='Metrics',
                 dtype=torch.int64):
        super(NumberOfEpisodes, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.number_episodes = torch.zeros(1, dtype=self.dtype)

    def call(self, time_step):
        """Increase the number of number_episodes according to time_step.
        It would increase for all time_step.is_last().

        Args:
            time_step (alf.data_structures.TimeStep): batched tensor
        Returns:
            The arguments, for easy chaining.
        """
        episodes = time_step.is_last().type(self.dtype)
        num_episodes = torch.sum(episodes)
        self.number_episodes.add_(num_episodes)
        return time_step

    def result(self):
        return self.number_episodes

    def reset(self):
        self.number_episodes.fill_(0)


class AverageReturnMetric(metric.StepMetric):
    """Metric to compute the average return."""

    def __init__(self,
                 name='AverageReturn',
                 prefix='Metrics',
                 dtype=torch.float32,
                 batch_size=1,
                 buffer_size=10):
        super(AverageReturnMetric, self).__init__(name=name, prefix=prefix)
        # TODO: use tensor deque
        self.buffer = TCDeque(max_len=buffer_size, dtype=dtype)
        self._dtype = dtype
        self.return_accumulator = torch.zeros(batch_size, dtype=dtype)

    def call(self, time_step):
        """Accumulates returns from time_step batched tensor.
        It would accumulate all time_step.reward.

        Args:
            time_step (alf.data_structures.TimeStep): batched tensor
        Returns:
            The arguments, for easy chaining.
        """
        # Zero out batch indices where a new episode is starting.
        self.return_accumulator[:] = torch.where(
            time_step.is_first(), torch.zeros_like(self.return_accumulator),
            self.return_accumulator)

        # Update accumulator with received rewards.
        # Ignores first step whose reward comes from the boundary transition
        # of the last step from the previous episode.
        self.return_accumulator.add_(
            torch.where(time_step.is_first(),
                        torch.zeros_like(self.return_accumulator),
                        time_step.reward))

        # Add final returns to buffer.
        last_episode_indices = torch.squeeze(*torch.where(
            time_step.is_last())).type(torch.int64)
        for indx in last_episode_indices:
            self.buffer.append(self.return_accumulator[indx].clone().detach())

        return time_step

    def result(self):
        return self.buffer.mean()

    def reset(self):
        self.buffer.clear()
        self.return_accumulator.fill_(0)


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
        # TODO: use tensor deque
        self.buffer = TCDeque(max_len=buffer_size, dtype=dtype)
        self._dtype = dtype
        self.length_accumulator = torch.zeros(batch_size, dtype=dtype)

    def call(self, time_step):
        """Accumulates the number of episode steps according to batched time_step.
        It would increase for all non time_step.is_first().  The first time_step
        is the boundary step and needs to be ignored, different from tf_agents.

        Args:
            time_step (alf.data_structures.TimeStep): batched tensor
        Returns:
            The arguments, for easy chaining.
        """
        # Each non-boundary time_step (mid or last) represents a step.
        is_first = time_step.is_first()
        non_boundary = torch.where(
            torch.logical_not(is_first),
            torch.ones_like(self.length_accumulator),
            torch.zeros_like(self.length_accumulator))
        self.length_accumulator.add_(non_boundary)

        # Add lengths to buffer when we hit end of episode
        is_last = time_step.is_last()
        last_indices = torch.squeeze(*torch.where(is_last)).type(torch.int64)
        for indx in last_indices:
            self.buffer.append(self.length_accumulator[indx].clone().detach())

        # Clear length accumulator at the end of episodes.
        self.length_accumulator[:] = torch.where(
            is_last, torch.zeros_like(self.length_accumulator),
            self.length_accumulator)

        return time_step

    def result(self):
        return self.buffer.mean()

    def reset(self):
        self.buffer.clear()
        self.length_accumulator.fill_(0)
