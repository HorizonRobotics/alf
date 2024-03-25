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
"""Schedulers."""
from functools import partial
from numbers import Number
from typing import Callable

import alf

_progress = {
    "percent": 0.0,
    "iterations": 0.0,
    "env_steps": 0.0,
    "global_counter": 0.0
}


def update_progress(progress_type: str, value: Number):
    _progress[progress_type] = float(value)


def update_all_progresses(progresses):
    _progress.update(progresses)


def get_progress(progress_type: str) -> float:
    return _progress[progress_type]


def get_all_progresses():
    return _progress


_is_scheduler_allowed = True


def disallow_scheduler():
    """Disallow the use of scheduler.

    In some context, scheduler cannot be used due to the lack of the information
    about training progress. This function is called by the framework to prevent
    the use of scheduler in such context.
    """
    global _is_scheduler_allowed
    _is_scheduler_allowed = False


class Scheduler(object):
    """Base class of all schedulers.

    A scheduler is used to generate manually defined values based on the training
    progress.

    The subclass should call ``progress()`` to get the current training progress
    and use it to calculate the scheduled value. There are three types of training
    progresses:

    * "percent": percent of training completed.
    * "iterations": the number training iterations.
    * "env_steps": the number of environment steps
    * "global_counter": the value from ``alf.summary.get_global_counter()``

    """

    def __init__(self, progress_type):
        """
        Args:
            progress_type (str): one of "percent", "iterations", "env_steps"
        """
        assert progress_type in _progress
        self._progress_type = progress_type

    def progress(self):
        assert _is_scheduler_allowed, (
            "Scheduler is not allowed in Environment "
            "unless TrainerConfig.sync_progress_to_envs is set to True")
        return _progress[self._progress_type]

    def final_value(self):
        """The (limit of) final scheduled value.
        """
        raise NotImplementedError()


class ConstantScheduler(object):
    def __init__(self, value):
        self._value = value

    def __call__(self):
        return self._value

    def __repr__(self):
        return str(self._value)

    def final_value(self):
        return self._value


@alf.configurable
class StepScheduler(Scheduler):
    """There is one value for each defined region of training progress."""

    def __init__(self,
                 progress_type,
                 schedule,
                 warm_up_period: Number = 0,
                 start: Number = 0):
        """
        Args:
            progress_type (str): one of "percent", "iterations", "env_steps"
            schedule (list[tuple]): each tuple is a pair of ``(progress, value)``
                the scheduled result will be the ``value`` of the smallest
                ``progress`` such that it is greater than the current
                training progress.
            warm_up_period: linearly increasing the output value from 0 to the
                first value (i.e schedule[0][0]) for a duration of ``warm_up_period``
                starting from ``start``. The value before ``start`` will be 0.
            start: see ``warm_up_period``
        """
        super().__init__(progress_type)
        self._progresses, self._values = zip(*schedule)
        self._index = 0
        self._warm_up_period = warm_up_period
        self._start = start
        assert start + warm_up_period < self._progresses[0]

    def __call__(self):
        progress = self.progress()
        if progress < self._start + self._warm_up_period:
            return self._values[0] * max(progress - self._start,
                                         0) / self._warm_up_period
        index = self._index
        progresses = self._progresses
        while index < len(progresses) - 1 and progress >= progresses[index]:
            index += 1
        self._index = index
        return self._values[index]

    def __repr__(self):
        return "StepScheduler('%s', %s, warm_up_period=%s, start=%s)" % (
            self._progress_type, list(zip(self._progresses, self._values)),
            self._warm_up_period, self._start)

    def final_value(self):
        return self._values[-1]


@alf.configurable
class LinearScheduler(Scheduler):
    """The value is linearly changed in each defined region of progress."""

    def __init__(self, progress_type, schedule):
        """
        Args:
            progress_type (str): one of "percent", "iterations", "env_steps"
            schedule (list[tuple]): each tuple is a pair of (progress, value)
                which means that if the current progress between progress[i-1]
                and progress[i], a linear interpolation between value[i-1] and
                value[i] will be used. progress[0] must be 0. If the current
                progress is greater than progress[-1], value[-1] will be used.
        """
        super().__init__(progress_type)
        assert schedule[0][
            0] == 0, "The first progress for linear scheduler must be 0."
        assert len(
            schedule
        ) >= 2, "There should be at least two (progress, value) pairs"
        self._progresses, self._values = zip(*schedule)
        self._index = 1

    def __call__(self):
        progress = self.progress()
        index = self._index
        progresses = self._progresses
        while index < len(progresses) and progress >= progresses[index]:
            index += 1
        if index < len(progresses):
            w = (progress - progresses[index - 1]) / (
                progresses[index] - progresses[index - 1])
            value = (1 - w) * self._values[index - 1] + w * self._values[index]
        else:
            index -= 1
            value = self._values[index]
        self._index = index
        return value

    def __repr__(self):
        return "LinearScheduler('%s', %s)" % (
            self._progress_type, list(zip(self._progresses, self._values)))

    def final_value(self):
        return self._values[-1]


@alf.configurable
class ExponentialScheduler(Scheduler):
    """The value is exponentially decayed based on the progress."""

    def __init__(self, progress_type, initial_value, decay_rate, decay_time):
        """
        The value is calculated as ``initial_value * decay_rate**(progress/decay_time)``
        Args:
            progress_type (str): one of "percent", "iterations", "env_steps"
            initial_value (float): initial value
            decay_rate (float):
            decay_time (float):
        """
        super().__init__(progress_type)
        self._initial_value = initial_value
        self._decay_rate = decay_rate
        self._decay_time = decay_time

    def __call__(self):
        progress = self.progress()
        return self._initial_value * self._decay_rate**(
            progress / self._decay_time)

    def __repr__(self):
        return "ExponentialScheduler('%s', initial_value=%s, decay_rate=%s, decay_time=%s)" % (
            self._progress_type, self._initial_value, self._decay_rate,
            self._decay_time)

    def final_value(self):
        return 0.


@alf.configurable
class CyclicalScheduler(Scheduler):
    """The cyclical scheduler where the value changes cyclically between two bounds.
    Reference:
    ::
        Leslie N. Smith Cyclical Learning Rates for Training Neural Networks, 2017
        (https://arxiv.org/pdf/1506.01186.pdf)

    This implementation generalizes the original methods in two ways: 1) the
    initial value can start from either the lower-bound (as in the original method),
    or upper bound; 2) apart from the linear switching between the bounds, we
    also support step mode of switching.

    In terms of applications, beyond the standard case of using a cyclical
    learning rate to improve the learning behavior during NN training, this
    scheduler is also useful in other cases. One example is in reinforcement
    learning, sometimes we want to update the parameters of different modules at
    difference paces. For example, in TD3, we want to update the policy every other
    updates. In this case, we can use a ``CyclicalScheduler`` with ``step``
    switching mode to achieve this. Similar cases also appears in Dreamer.
    """

    def __init__(self,
                 progress_type,
                 base_lr,
                 bound_lr,
                 half_cycle_size,
                 switch_mode='step'):
        """
        Args:
            progress_type (str): one of "percent", "iterations", "env_steps"
            base_lr (float): the base learning rate, representing the starting
                value.
            bound_lr (float): the value of the learning rate on the other bound.
                The value of ``bound_lr`` could be either larger or smaller than
                the value of ``base_lr``.
            half_cycle_size (int|float): the length of half a cycle. Its actual
                length is based on the ``progress_type``. For example, if in
                "iterations" mode, it means the lr value will reach the opposite
                bound every ``half_cycle_size`` iterations.
            switch_mode (str): the way to switch from one bound to the other.
                Currently support the following modes:
                - step: directly jump from one mode to the other every half cycle
                - linear: linearly move from one mode to the other every half cycle
        """
        super().__init__(progress_type)
        self._base_lr = base_lr
        self._bound_lr = bound_lr
        self._half_cycle_size = half_cycle_size
        self._cycle_size = half_cycle_size * 2
        assert switch_mode in {
            "step", "linear"
        }, ("unsupportted switch mode {}".format(switch_mode))
        self._switch_mode = switch_mode
        self._current_value = base_lr
        # Apply rounding the the calculated progress in cycle and half-cycle
        # when progress_type is ``percent`` to avoid the issue in stage
        # transition due to numerical reasons.
        # For the other progress_types, no rounding is applied.
        self._rounding_func = partial(round, ndigits=10) \
                            if progress_type == "percent" else lambda x: x

    def __call__(self):
        progress = self.progress()

        progress_in_half_cycle = self._rounding_func(
            (progress % self._half_cycle_size / self._half_cycle_size)) % 1
        progress_in_cycle = self._rounding_func(
            (progress % self._cycle_size / self._cycle_size)) % 1

        if self._switch_mode == "step":
            # step mode changes value at half-cycle point
            if progress_in_cycle < 0.5:
                self._current_value = self._base_lr
            else:
                self._current_value = self._bound_lr

            return self._current_value

        elif self._switch_mode == "linear":
            if progress_in_cycle < 0.5:
                return (1 - progress_in_half_cycle) * self._base_lr + \
                    progress_in_half_cycle * self._bound_lr
            else:
                return progress_in_half_cycle * self._base_lr + \
                    (1 - progress_in_half_cycle) * self._bound_lr

    def __repr__(self):
        return ("CyclicalScheduler('%s', base_lr=%s, bound_lr=%s,"
                "half_cycle_size=%s, switch_mode=%s)") % (
                    self._progress_type, self._base_lr, self._bound_lr,
                    self._half_cycle_size, self._switch_mode)

    def final_value(self):
        raise RuntimeError(
            "This scheduler is cyclical and does not have a final value.")


def as_scheduler(value_or_scheduler):
    if isinstance(value_or_scheduler, Callable):
        return value_or_scheduler
    else:
        return ConstantScheduler(value_or_scheduler)
