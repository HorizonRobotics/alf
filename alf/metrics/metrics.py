# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from typing import List
import numpy as np

import torch

import alf
import alf.utils.data_buffer as db
from alf.data_structures import TimeStep, StepType

from alf.utils import common
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
        self._max_len = max_len
        self.register_buffer(
            "_buf", torch.zeros((max_len, ), dtype=dtype, device='cpu'))
        self.register_buffer("_current_pos",
                             torch.zeros((), dtype=torch.int64, device='cpu'))

    def append(self, value):
        """Append multiple values to the buffer.

        Args:
            value (Tensor): a batch of scalars with the shape :math:`[B]`.
        """
        n = value.size(0)
        if n == 1:
            self._buf[self._current_pos % self._max_len] = value[0]
            self._current_pos += 1
        else:
            n = min(n, self._max_len)
            pos = (self._current_pos + torch.arange(
                n, device='cpu')) % self._max_len
            self._buf[pos] = value[:n]
            self._current_pos += n

    def mean(self):
        if self._current_pos == 0:  # avoid nan
            return torch.tensor(0, dtype=self._dtype)
        current_size = self._current_pos.clamp(max=self._max_len)
        return self._buf[:current_size].mean()

    def std(self):
        if self._current_pos == 0:  # avoid nan
            return torch.tensor(0, dtype=self._dtype)
        current_size = self._current_pos.clamp(max=self._max_len)
        return self._buf[:current_size].std()

    def latest(self):
        """Return the value added most recently.
        """
        assert self._current_pos > 0, "no valid latest value!"
        return self._buf[(self._current_pos - 1) % self._max_len]

    def clear(self):
        self._current_pos.zero_()


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

    def std(self):
        return torch.zeros_like(self._number_episodes)

    def reset(self):
        self._number_episodes.fill_(0)


class AverageEpisodicAggregationMetric(metric.StepMetric):
    """A base metric to aggregate quantities over an episode. It supports accumulating
    a nest of scalar values.

    NOTE: normally this class and its sub-classes report metrics by summing values
    over the whole episode. However, there are two special treatments:
    1. if ``_extract_metric_values()`` returns a nested structure in which a
       dictionary or namedtuple has a field with postfix "@step", the corresponding
       value will be averaged instead of summed over the whole episode length, so
       that a per-step average value is reported.
    2. If a field has a postfix "@max", then the aggregated value will be the
       maximum (instead of sum) of step values across the episode.

    This class supports partial aggregation, where if at any step the extracted
    metric value is not finite (inf or nan), then that step's value will be skipped
    for aggregation. If a field is skipped for an entire episode, its accumulated
    value won't be pushed into the metric buffer.
    """

    def __init__(self,
                 name="AverageEpisodicAggregationMetric",
                 prefix='Metrics',
                 dtype=torch.float32,
                 buffer_size=10,
                 example_time_step=None):
        """
        Args:
            name (str):
            prefix (str): a prefix indicating the category of the metric
            dtype (torch.dtype): dtype of metric values. Should be floating types
                in order to be averaged.
            buffer_size (int): number of episodes the metric value will be averaged
                across
            example_time_step (nest): an example of the time step where the metric
               values are extracted from. If ``None``, a zero scalar is used as the
               example metric value.
        """
        super(AverageEpisodicAggregationMetric, self).__init__(
            name=name, dtype=dtype, prefix=prefix)
        if example_time_step is None:
            example_metric_value = torch.zeros((), device='cpu')
        else:
            example_metric_value = self._extract_metric_values(
                example_time_step.cpu())
        self._batch_size = alf.nest.get_nest_batch_size(example_time_step)
        self._buffer_size = buffer_size
        self._initialize(example_metric_value)

        # ``self._current_step`` will be set to zero for the first step, and is
        # added by one otherwise. Therefore, at the episode end, its value
        # equals to episode length - 1.
        self._current_step = torch.zeros(self._batch_size, device='cpu')

    def _extract_metric_values(self, time_step):
        """Extract metrics from the time step. The return can be a nest."""
        raise NotImplementedError()

    def _initialize(self, example_metric_value):
        def _init_buf(val):
            return MetricBuffer(max_len=self._buffer_size, dtype=self._dtype)

        def _init_acc(val):
            accumulator = torch.zeros(
                self._batch_size, dtype=self._dtype, device='cpu')
            return accumulator

        def _init_mask(val):
            return torch.zeros(
                self._batch_size, dtype=torch.bool, device='cpu')

        def _init_step(val):
            return torch.zeros(
                self._batch_size, dtype=self._dtype, device='cpu')

        self._buffer = alf.nest.map_structure(_init_buf, example_metric_value)
        self._accumulator = alf.nest.map_structure(_init_acc,
                                                   example_metric_value)
        # which samples of a batch in ``self._accumulator`` are valid for being
        # put into ``self._buffer`` when step_type==LAST
        self._mask = alf.nest.map_structure(_init_mask, example_metric_value)
        self._steps = alf.nest.map_structure(_init_step, example_metric_value)

    def call(self, time_step):
        """Accumulate values from the time step. The values are defined by
        subclasses' ``_extract_metric_values()``. It will ignore the values of
        first time steps.

        Args:
            time_step (alf.data_structures.TimeStep): batched tensor
        Returns:
            The arguments, for easy chaining.
        """

        self._current_step = torch.where(time_step.is_first(),
                                         torch.zeros_like(self._current_step),
                                         self._current_step + 1)

        values = self._extract_metric_values(time_step)

        assert all(
            alf.nest.flatten(
                alf.nest.map_structure(
                    lambda val: list(val.shape) == [self._batch_size],
                    values))), ("Value shape is not correct "
                                "(only scalar values are supported).")

        is_first = time_step.is_first()

        def _update_accumulator_(path, mask, step, acc, val):
            """In-place update of the accumulators and mask."""
            val_valid = torch.isfinite(val)
            # If at any step the value is valid, then the acc value becomes valid
            mask[:] = torch.where(is_first, 0, mask | val_valid)
            # Only step+1 if the value is valid
            step[:] = torch.where(is_first, 0,
                                  step + val_valid.to(self._dtype))

            if path.endswith("@max"):
                # Don't max invalid values
                val = torch.where(val_valid, val, -float('inf'))
                acc[:] = torch.where(is_first, -float('inf'),
                                     torch.maximum(acc, val.to(self._dtype)))
            else:
                # Don't sum invalid values
                val = torch.where(val_valid, val, 0)
                # Zero out batch indices where a new episode is starting.
                # Update with new values; Ignores first step whose reward comes from
                # the boundary transition of the last step from the previous episode.
                acc[:] = torch.where(is_first, 0, acc + val.to(self._dtype))

        alf.nest.py_map_structure_with_path(_update_accumulator_, self._mask,
                                            self._steps, self._accumulator,
                                            values)

        def _episode_end_aggregate_(path, mask, step, buf, acc):
            value = self._extract_and_process_acc_value(
                acc, last_episode_indices)
            # If the metric's name ends with '@step', the value will
            # be further averaged over episode length so that the
            # result is per-step value.
            if path.endswith('@step'):
                value = value / step[last_episode_indices]
            mask = mask[last_episode_indices]
            value = value[mask]
            if value.numel() > 0:
                buf.append(value)

        # Extract the final accumulated value and do customizable processing
        # via ``_extract_and_process_acc_value``, and add the processed
        # result to buffer
        last_episode_indices = torch.where(time_step.is_last())[0]

        if len(last_episode_indices) > 0:
            alf.nest.py_map_structure_with_path(
                _episode_end_aggregate_, self._mask, self._steps, self._buffer,
                self._accumulator)

        return time_step

    def _extract_and_process_acc_value(self, acc, last_episode_indices):
        """Extract the final accumulated value and perform some optional
        customizable processing.
        Args:
            acc (Tensor): batched tensor representing an accumulator
            last_episode_indices (Tensor): indices representing the location
                of the position of the last step
        Returns:
            The value of the accumulator at the episode end.
        """
        return acc[last_episode_indices]

    def result(self):
        return alf.nest.map_structure(lambda buf: buf.mean(), self._buffer)

    def std(self):
        return alf.nest.map_structure(lambda buf: buf.std(), self._buffer)

    def latest(self):
        """Return the value added most recently.
        """
        return alf.nest.map_structure(lambda buf: buf.latest(), self._buffer)

    def reset(self):
        alf.nest.map_structure(lambda buf: buf.clear(), self._buffer)
        alf.nest.map_structure(lambda acc: acc.fill_(0), self._accumulator)


class AverageReturnMetric(AverageEpisodicAggregationMetric):
    """Metric for computing the average return."""

    def __init__(self,
                 example_time_step: TimeStep,
                 name='AverageReturn',
                 prefix='Metrics',
                 dtype=torch.float32,
                 buffer_size=10):
        super(AverageReturnMetric, self).__init__(
            name=name,
            dtype=dtype,
            prefix=prefix,
            buffer_size=buffer_size,
            example_time_step=example_time_step)

    def _extract_metric_values(self, time_step):
        """Accumulate immediate rewards to get episodic return."""
        ndim = time_step.step_type.ndim
        if time_step.reward.ndim == ndim:
            return time_step.reward
        else:
            reward = time_step.reward.reshape(*time_step.step_type.shape, -1)
            return [reward[..., i] for i in range(reward.shape[-1])]


@alf.configurable
class AverageDiscountedReturnMetric(AverageEpisodicAggregationMetric):
    r"""Metric for computing the average discounted episodic return.
    It is calculated according to the following formula:

    .. math::
        \begin{array}{ll}
            R &=\frac{1}{L} (r_1 + (1+\gamma) r_2 + (1+\gamma+\gamma^2) r_3 + \cdots) \\
            &= \frac{1}{L}\sum_{l=1}^L \sum_{k=0}^{l-1} \gamma^k r_l,
        \end{array}
    where :math:`\gamma` is the reward discount, and :math:`r_1` denotes the
    reward due to the first action, which is received at the second time step.
    :math:`L` equals to the episode length - 1.


    Note that if the last step is not due to time limit, the discounted return
    calculated from the formula above is unbiased. If the last step is due to
    time limit, it is a biased estimate and its expectation is lower than the
    ground-truth (when rewards are non-negative).
    """

    def __init__(self,
                 example_time_step: TimeStep,
                 name='AverageDiscountedReturn',
                 prefix='Metrics',
                 dtype=torch.float32,
                 discount=0.99,
                 reward_transformer=None,
                 buffer_size=10):
        """
        Args:
            discount (float): the discount factor for calculating the discounted
                return
            reward_transformer (Callable): if provided, will calculate the
                discounted return using the transformed reward. It will be called
                as ``transformed_reward = reward_transformer(original_reward)``.
        """
        torch.nn.Module.__init__(self)
        self._discount = discount
        batch_size = alf.nest.get_nest_batch_size(example_time_step)
        self._accumulated_discount = torch.zeros(batch_size, device='cpu')
        self._timestep_discount = torch.zeros(batch_size, device='cpu')
        self._reward_transformer = reward_transformer
        self._current_step = torch.zeros(batch_size, device='cpu')

        super().__init__(
            name=name,
            dtype=dtype,
            prefix=prefix,
            buffer_size=buffer_size,
            example_time_step=example_time_step)

    def _extract_metric_values(self, time_step):
        """Accumulate discounted immediate rewards to get discounted episodic
        return. It also updates the accumulated discount and step count.
        """
        self._update_discount(time_step)

        ndim = time_step.step_type.ndim
        reward = time_step.reward
        if self._reward_transformer is not None:
            # RewardNormalizer may change its statistics if the exe_mode is
            # ROLLOUT. We don't want that happen for metric calculation.
            old_mode = common.set_exe_mode(common.EXE_MODE_OTHER)
            reward = self._reward_transformer(reward)
            common.set_exe_mode(old_mode)
        if time_step.reward.ndim == ndim:
            discounted_reward = reward * self._accumulated_discount
        else:
            reward = reward.reshape(*time_step.step_type.shape, -1)
            discounted_reward = list(reward.T * self._accumulated_discount)

        return discounted_reward

    def _update_discount(self, time_step):
        """Set/Update the values of ``self._accumulated_discount``.

        If this is the first step, ``self._accumulated_discount`` will be set
        to zero. Otherwise, it is multiplied by ``discount`` and added by one.
        The updated accumulated discount will be used for computing the
        accumulated contribution of the the reward received at the current step
        to the discounted return.

        Args:
            time_step (alf.data_structures.TimeStep): batched tensor
        """
        is_first = time_step.is_first()

        # update discount for the next time step
        self._accumulated_discount = (self._discount * self._timestep_discount
                                      * self._accumulated_discount + 1)
        self._timestep_discount = time_step.discount
        self._accumulated_discount = torch.where(
            is_first, torch.zeros_like(self._accumulated_discount),
            self._accumulated_discount)

    def _extract_and_process_acc_value(self, acc, last_episode_indices):
        """Extract the final accumulated value and divide by the number of
        steps of an episode.
        Args:
            acc (Tensor): batched tensor representing an accumulator
            last_episode_indices (Tensor): indices representing the location
                of the position of the last step
        Returns:
            The value of the accumulator at the episode end.
        """
        return acc[last_episode_indices] / self._current_step[
            last_episode_indices]


@alf.configurable
class EpisodicStartAverageDiscountedReturnMetric(
        AverageDiscountedReturnMetric):
    r"""Metric for computing the discounted return from episode start states.
    It is calculated according to the following formula:

    .. math::
        \begin{array}{ll}
            R &=r_1 + \gamma r_2 + \gamma^2 r_3 + \cdots \\
            &= \sum_{l=1}^L \gamma^{l-1} r_l,
        \end{array}
    where :math:`\gamma` is the reward discount, and :math:`r_1` denotes the
    reward due to the first action, which is received at the second time step.
    :math:`L` equals to the episode length - 1.


    Note that if the last step is not due to time limit, the discounted return
    calculated from the formula above is unbiased. If the last step is due to
    time limit, it is a biased estimate and its expectation is lower than the
    ground-truth (when rewards are non-negative).
    """

    def __init__(self,
                 example_time_step: TimeStep,
                 name='EpisodicStartAverageDiscountedReturn',
                 prefix='Metrics',
                 buffer_size=10):
        super().__init__(
            name=name,
            prefix=prefix,
            buffer_size=buffer_size,
            example_time_step=example_time_step)

    def _extract_metric_values(self, time_step):
        """Accumulate discounted immediate rewards to get discounted episodic
        return. It also updates the accumulated discount and step count.
        """
        self._update_discount_and_step_count(time_step)

        ndim = time_step.step_type.ndim
        if time_step.reward.ndim == ndim:
            discounted_reward = time_step.reward * self._accumulated_discount
        else:
            reward = time_step.reward.reshape(*time_step.step_type.shape, -1)
            discounted_reward = list(
                (reward * self._accumulated_discount.unsqueeze(-1)).permute(
                    reward.ndim - 1, *torch.arange(reward.ndim - 1)))

        return discounted_reward

    def _update_discount_and_step_count(self, time_step):
        """Set/Update the values of ``self._accumulated_discount`` and
        the value of ``self._current_step``.
        If this is the first step, ``self._accumulated_discount`` will be set
        to zero. Otherwise, it is multiplied by ``discount`` and added by one.
        The updated accumulated discount will be used for computing the
        accumulated contribution of the the reward received at the current step
        to the discounted return.
        ``self._current_step`` will be set to zero for the first step, and
        is added by one otherwise. Therefore, at the episode end, its value
        equals to episode length - 1.

        Args:
            time_step (alf.data_structures.TimeStep): batched tensor
        """
        is_first = time_step.is_first()

        # update discount for the next time step
        self._accumulated_discount *= self._discount
        self._accumulated_discount.masked_fill_(
            self._accumulated_discount == 0, 1.)
        self._accumulated_discount.masked_fill_(is_first, 0.)

    def _extract_and_process_acc_value(self, acc, last_episode_indices):
        """Extract the final accumulated value.
        Args:
            acc (Tensor): batched tensor representing an accumulator
            last_episode_indices (Tensor): indices representing the location
                of the position of the last step
        Returns:
            The value of the accumulator at the episode end.
        """
        return acc[last_episode_indices]


@alf.configurable
class AverageRewardMetric(AverageDiscountedReturnMetric):
    """Metric for computing the average reward per time step for each episode.
    """

    def __init__(self,
                 example_time_step: TimeStep,
                 name='AverageReward',
                 prefix='Metrics',
                 buffer_size=10):
        super().__init__(
            example_time_step=example_time_step,
            name=name,
            prefix=prefix,
            buffer_size=buffer_size,
            discount=0)


class AverageEpisodeLengthMetric(AverageEpisodicAggregationMetric):
    """Metric for computing the average episode length."""

    def __init__(self,
                 example_time_step: TimeStep,
                 name='AverageEpisodeLength',
                 prefix='Metrics',
                 dtype=torch.float32,
                 buffer_size=10):
        super(AverageEpisodeLengthMetric, self).__init__(
            name=name,
            dtype=dtype,
            prefix=prefix,
            buffer_size=buffer_size,
            example_time_step=example_time_step)

    def _extract_metric_values(self, time_step):
        """Return a constant of 1 each time, except for ``time_step.is_first()``.
        The first time step is the boundary step and needs to be ignored, different
        from ``tf_agents``
        """
        return torch.where(time_step.is_first(),
                           torch.zeros_like(time_step.step_type),
                           torch.ones_like(time_step.step_type))


@alf.configurable(whitelist=['fields'])
class AverageEnvInfoMetric(AverageEpisodicAggregationMetric):
    """Metric for computing average quantities contained in the environment info.
    An example of env info (which can be a nest) has to be provided when constructing
    an instance in order to initialize the accumulator and buffer with the same
    nested structure.
    """

    def __init__(self,
                 example_time_step: TimeStep,
                 name="AverageEnvInfoMetric",
                 prefix="Metrics",
                 dtype=torch.float32,
                 fields: List[str] = None,
                 buffer_size=10):
        """
        Args:
            fields: a list of fields to include in the average env info metric.
                If None, all fields will be included.
        """
        self._fields = fields
        super(AverageEnvInfoMetric, self).__init__(
            name=name,
            dtype=dtype,
            prefix=prefix,
            buffer_size=buffer_size,
            example_time_step=example_time_step)

    def _extract_metric_values(self, time_step):
        if self._fields is None:
            return time_step.env_info
        else:
            return {f: time_step.env_info[f] for f in self._fields}
