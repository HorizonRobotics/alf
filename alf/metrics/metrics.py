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
# Code adapted from https://github.com/tensorflow/agents/blob/master/tf_agents/metrics/tf_metrics.py

import collections
from alf.metrics import metric
import torch


def mean(q, default=0):
    if len(q) > 0:
        return sum(q) / len(q)
    else:
        return default


class EnvironmentSteps(metric.StepMetric):
    """Counts the number of steps taken in the environment."""

    def __init__(self,
                 name='EnvironmentSteps',
                 prefix='Metrics',
                 dtype=torch.int64):
        super(EnvironmentSteps, self).__init__(name=name, prefix=prefix)
        self.dtype = dtype
        self.environment_steps = torch.zeros(0, self.dtype)

    def call(self, trajectory):
        """Increase the number of environment_steps according to trajectory.
    Step count is not increased on trajectory.boundary() since that step
    is not part of any episode.
    Args:
      trajectory: A tf_agents.trajectory.Trajectory
    Returns:
      The arguments, for easy chaining.
    """
        # The __call__ will execute this.
        steps = (~trajectory.is_boundary()).type(self.dtype)
        num_steps = torch.sum(steps)
        self.environment_steps.add_(num_steps)
        return trajectory

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
        self.number_episodes = torch.zeros(0, self.dtype)

    def call(self, trajectory):
        """Increase the number of number_episodes according to trajectory.
    It would increase for all trajectory.is_last().
    Args:
      trajectory: A tf_agents.trajectory.Trajectory
    Returns:
      The arguments, for easy chaining.
    """
        # The __call__ will execute this.
        episodes = trajectory.is_last().type(self.dtype)
        num_episodes = torch.sum(episodes)
        self.number_episodes.add_(num_episodes)
        return trajectory

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
        self._buffer = collections.deque(maxlen=buffer_size)
        self._dtype = dtype
        self._return_accumulator = torch.zeros(batch_size, dtype=dtype)

    def call(self, trajectory):
        # Zero out batch indices where a new episode is starting.
        self._return_accumulator[:] = torch.where(
            trajectory.is_first(), torch.zeros_like(self._return_accumulator),
            self._return_accumulator)

        # Update accumulator with received rewards.
        self._return_accumulator.add_(trajectory.reward)

        # Add final returns to buffer.
        last_episode_indices = torch.squeeze(*torch.where(
            trajectory.is_last())).type(torch.int64)
        for indx in last_episode_indices:
            self._buffer.append(self._return_accumulator[indx])

        return trajectory

    def result(self):
        return mean(self._buffer)

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
        self._buffer = collections.deque(max_len=buffer_size)
        self._dtype = dtype
        self._length_accumulator = torch.zeros(batch_size, dtype=dtype)

    def call(self, trajectory):
        # Each non-boundary trajectory (first, mid or last) represents a step.
        boundary = trajectory.is_boundary()
        non_boundary = torch.where(
            torch.logical_not(boundary), torch.ones_like(boundary),
            torch.zeros_like(trajectory))
        self._length_accumulator.add_(non_boundary)

        # Add lengths to buffer when we hit end of episode
        is_last = trajectory.is_last()
        last_indices = torch.squeeze(*torch.where(is_last)).type(torch.int64)
        for indx in last_indices:
            self._buffer.append(self._length_accumulator[indx])

        # Clear length accumulator at the end of episodes.
        self._length_accumulator[:] = torch.where(
            is_last,
            torch.zeros_like(self._length_accumulator, dtype=self._dtype),
            self._length_accumulator)

        return trajectory

    def result(self):
        return mean(self._buffer)

    def reset(self):
        self._buffer.clear()
        self._length_accumulator.fill_(0)
