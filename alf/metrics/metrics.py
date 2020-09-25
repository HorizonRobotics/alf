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
import torch

import alf
import alf.utils.data_buffer as db

from . import metric


class MetricBuffer(torch.nn.Module):
    """A metric buffer for computing average metric values. The buffer is assumed
    to store only scalar values."""

    def __init__(self, max_len, dtype):
        """
        Args:
            max_len (int): maximum length of the buffer
            dtype (torch.dtype): dtype of the content of the buffer
        """
        super().__init__()
        self._dtype = dtype
        self._buf = db.DataBuffer(
            data_spec=alf.tensor_specs.TensorSpec((), dtype=dtype),
            capacity=max_len,
            device='cpu')

    def append(self, value):
        """Append multiple values to the buffer.

        Args:
            value (Tensor): a batch of scalars with the shape :math:`[B]`.
        """
        self._buf.add_batch(value)

    def mean(self):
        if self._buf.current_size == 0:  # avoid nan
            return torch.tensor(0, dtype=self._dtype, device='cpu')
        return self._buf.get_all()[:self._buf.current_size].mean()

    def clear(self):
        return self._buf.clear()


class EnvironmentSteps(metric.StepMetric):
    """Counts the number of steps taken in the environment after FrameSkip.

    If Frames are skipped by any of the environment wrappers, a separate metric
    AverageEnvInfoMetric['num_env_frames'] will report the actual frame count including
    skipped ones.
    """

    def __init__(self,
                 name='EnvironmentSteps',
                 prefix='Metrics',
                 dtype=torch.int64):
        super().__init__(name=name, dtype=dtype, prefix=prefix)
        self.register_buffer('_environment_steps', torch.zeros((),
                                                               dtype=dtype))

    def call(self, time_step):
        """Increase the number of environment_steps according to ``time_step``.
        Step count is not increased on ``time_step.is_first()`` since that step
        is not part of any episode.

        Args:
            time_step (alf.data_structures.TimeStep): batched tensor
        Returns:
            The arguments, for easy chaining.
        """
        steps = (torch.logical_not(time_step.is_first())).type(self._dtype)
        num_steps = torch.sum(steps)
        self._environment_steps.add_(num_steps)
        return time_step

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
        super(NumberOfEpisodes, self).__init__(
            name=name, dtype=dtype, prefix=prefix)
        self.register_buffer('_number_episodes', torch.zeros((), dtype=dtype))

    def call(self, time_step):
        """Increase the number of number_episodes according to ``time_step``.
        It would increase for all ``time_step.is_last()``.

        Args:
            time_step (alf.data_structures.TimeStep): batched tensor
        Returns:
            The arguments, for easy chaining.
        """
        episodes = time_step.is_last().type(self._dtype)
        num_episodes = torch.sum(episodes)
        self._number_episodes.add_(num_episodes)
        return time_step

    def result(self):
        return self._number_episodes

    def reset(self):
        self._number_episodes.fill_(0)


class AverageEpisodicSumMetric(metric.StepMetric):
    """A base metric to sum up quantities over an episode. It supports accumulating
    a nest of scalar values.
    """

    def __init__(self,
                 name="AverageEpisodicSumMetric",
                 prefix='Metrics',
                 dtype=torch.float32,
                 batch_size=1,
                 buffer_size=10,
                 example_metric_value=None):
        """
        Args:
            name (str):
            prefix (str): a prefix indicating the category of the metric
            dtype (torch.dtype): dtype of metric values. Should be floating types
                in order to be averaged.
            batch_size (int): the number of metric values generated in parallel
            buffer_size (int): number of episodes the metric value will be averaged
                across
            example_metric_value (nest): an example of metric value to be summarized;
                if ``None``, a zero scalar is used.
        """
        super(AverageEpisodicSumMetric, self).__init__(
            name=name, dtype=dtype, prefix=prefix)
        if example_metric_value is None:
            example_metric_value = torch.zeros((), device='cpu')
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._initialize(example_metric_value)

    def _extract_metric_values(self, time_step):
        """Extract metrics from the time step. The return can be a nest."""
        raise NotImplementedError()

    def _initialize(self, example_metric_value):
        counter = [0]

        def _init_buf(val):
            return MetricBuffer(max_len=self._buffer_size, dtype=self._dtype)

        def _init_acc(val):
            accumulator = torch.zeros(
                self._batch_size, dtype=self._dtype, device='cpu')
            self.register_buffer('_accumulator%d' % counter[0], accumulator)
            counter[0] += 1
            return accumulator

        self._buffer = alf.nest.map_structure(_init_buf, example_metric_value)
        self._accumulator = alf.nest.map_structure(_init_acc,
                                                   example_metric_value)

    def call(self, time_step):
        """Accumulate values from the time step. The values are defined by
        subclasses' ``_extract_metric_values()``. It will ignore the values of
        first time steps.

        Args:
            time_step (alf.data_structures.TimeStep): batched tensor
        Returns:
            The arguments, for easy chaining.
        """

        values = self._extract_metric_values(time_step)

        assert all(
            alf.nest.flatten(
                alf.nest.map_structure(
                    lambda val: list(val.shape) == [self._batch_size],
                    values))), ("Value shape is not correct "
                                "(only scalar values are supported).")

        is_first = time_step.is_first()

        def _update_accumulator_(acc, val):
            """In-place update of the accumulators."""
            # Zero out batch indices where a new episode is starting.
            # Update with new values; Ignores first step whose reward comes from
            # the boundary transition of the last step from the previous episode.
            acc[:] = torch.where(is_first, torch.zeros_like(acc),
                                 acc + val.to(self._dtype))

        alf.nest.map_structure(_update_accumulator_, self._accumulator, values)

        # Add final accumulated value to buffer.
        last_episode_indices = torch.where(time_step.is_last())[0]

        if len(last_episode_indices) > 0:
            alf.nest.map_structure(
                lambda buf, acc: buf.append(acc[last_episode_indices]),
                self._buffer, self._accumulator)

        return time_step

    def result(self):
        return alf.nest.map_structure(lambda buf: buf.mean(), self._buffer)

    def reset(self):
        alf.nest.map_structure(lambda buf: buf.clear(), self._buffer)
        alf.nest.map_structure(lambda acc: acc.fill_(0), self._accumulator)


class AverageReturnMetric(AverageEpisodicSumMetric):
    """Metric for computing the average return."""

    def __init__(self,
                 name='AverageReturn',
                 prefix='Metrics',
                 reward_shape=(),
                 dtype=torch.float32,
                 batch_size=1,
                 buffer_size=10):
        if reward_shape == ():
            example_metric_value = torch.zeros((), device='cpu')
        else:
            example_metric_value = torch.zeros(
                reward_shape, device='cpu').reshape(-1)
            example_metric_value = list(example_metric_value)

        super(AverageReturnMetric, self).__init__(
            name=name,
            dtype=dtype,
            prefix=prefix,
            batch_size=batch_size,
            buffer_size=buffer_size,
            example_metric_value=example_metric_value)

    def _extract_metric_values(self, time_step):
        """Accumulate immediate rewards to get episodic return."""
        ndim = time_step.step_type.ndim
        if time_step.reward.ndim == ndim:
            return time_step.reward
        else:
            reward = time_step.reward.reshape(*time_step.step_type.shape, -1)
            return [reward[..., i] for i in range(reward.shape[-1])]


class AverageEpisodeLengthMetric(AverageEpisodicSumMetric):
    """Metric for computing the average episode length."""

    def __init__(self,
                 name='AverageEpisodeLength',
                 prefix='Metrics',
                 dtype=torch.float32,
                 batch_size=1,
                 buffer_size=10):
        super(AverageEpisodeLengthMetric, self).__init__(
            name=name,
            dtype=dtype,
            prefix=prefix,
            batch_size=batch_size,
            buffer_size=buffer_size)

    def _extract_metric_values(self, time_step):
        """Return a constant of 1 each time, except for ``time_step.is_first()``.
        The first time step is the boundary step and needs to be ignored, different
        from ``tf_agents``
        """
        return torch.where(time_step.is_first(),
                           torch.zeros_like(time_step.step_type),
                           torch.ones_like(time_step.step_type))


class AverageEnvInfoMetric(AverageEpisodicSumMetric):
    """Metric for computing average quantities contained in the environment info.
    An example of env info (which can be a nest) has to be provided when constructing
    an instance in order to initialize the accumulator and buffer with the same
    nested structure.
    """

    def __init__(self,
                 example_env_info,
                 name="AverageEnvInfoMetric",
                 prefix="Metrics",
                 dtype=torch.float32,
                 batch_size=1,
                 buffer_size=10):
        super(AverageEnvInfoMetric, self).__init__(
            name=name,
            dtype=dtype,
            prefix=prefix,
            batch_size=batch_size,
            buffer_size=buffer_size,
            example_metric_value=example_env_info)

    def _extract_metric_values(self, time_step):
        return time_step.env_info
