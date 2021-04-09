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
import gin

import alf
from alf.trainers.policy_trainer import Trainer


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
        if progress_type == "percent":
            self._progress_func = Trainer.progress
        elif progress_type == "iterations":
            self._progress_func = Trainer.current_iterations
        elif progress_type == "env_steps":
            self._progress_func = Trainer.current_env_steps
        elif progress_type == "global_counter":
            self._progress_func = alf.summary.get_global_counter
        else:
            raise ValueError("Unknown progress_type: %s" % progress_type)

        self._progress_type = progress_type

    def progress(self):
        return float(self._progress_func())


class ConstantScheduler(object):
    def __init__(self, value):
        self._value = value

    def __call__(self):
        return self._value

    def __repr__(self):
        return 'ConstantScheduler(%s)' % self._value


@alf.configurable
class StepScheduler(Scheduler):
    """There is one value for each defined region of training progress."""

    def __init__(self, progress_type, schedule):
        """
        Args:
            progress_type (str): one of "percent", "iterations", "env_steps"
            schedule (list[tuple]): each tuple is a pair of (progress, value)
                which means that if current progress is less than progress,
                the current value will be value.
        """
        super().__init__(progress_type)
        self._progresses, self._values = zip(*schedule)
        self._index = 0

    def __call__(self):
        progress = self.progress()
        index = self._index
        progresses = self._progresses
        while index < len(progresses) - 1 and progress >= progresses[index]:
            index += 1
        self._index = index
        return self._values[index]

    def __repr__(self):
        return "StepScheduler('%s', %s)" % (
            self._progress_type, list(zip(self._progresses, self._values)))


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
