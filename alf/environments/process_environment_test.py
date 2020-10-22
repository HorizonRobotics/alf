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
"""Tests for the process_environment.
Adapted from TF-Agents' parallel_py_environment_test.py
"""

import functools
import torch

import alf
from alf.environments.process_environment import ProcessEnvironment
from alf.environments.random_alf_environment import RandomAlfEnvironment
import alf.tensor_specs as ts


class ProcessEnvironmentTest(alf.test.TestCase):
    def test_close_no_hang_after_init(self):
        constructor = functools.partial(
            RandomAlfEnvironment,
            ts.TensorSpec((3, 3), torch.float32),
            ts.BoundedTensorSpec([1], torch.float32, minimum=-1.0,
                                 maximum=1.0),
            episode_end_probability=0,
            min_duration=2,
            max_duration=2)
        env = ProcessEnvironment(constructor)
        env.start()
        env.close()

    def test_close_no_hang_after_step(self):
        constructor = functools.partial(
            RandomAlfEnvironment,
            ts.TensorSpec((3, 3), torch.float32),
            ts.BoundedTensorSpec([1], torch.float32, minimum=-1.0,
                                 maximum=1.0),
            episode_end_probability=0,
            min_duration=5,
            max_duration=5)
        env = ProcessEnvironment(constructor)
        env.start()
        action_spec = env.action_spec()
        env.reset()
        env.step(action_spec.sample())
        env.step(action_spec.sample())
        env.close()

    def test_reraise_exception_in_init(self):
        constructor = MockEnvironmentCrashInInit
        env = ProcessEnvironment(constructor)
        with self.assertRaises(Exception):
            env.start()

    def test_reraise_exception_in_reset(self):
        constructor = MockEnvironmentCrashInReset
        env = ProcessEnvironment(constructor)
        env.start()
        with self.assertRaises(Exception):
            env.reset()

    def test_reraise_exception_in_step(self):
        crash_at_step = 3
        constructor = functools.partial(MockEnvironmentCrashInStep,
                                        crash_at_step)
        env = ProcessEnvironment(constructor)
        env.start()
        env.reset()
        action_spec = env.action_spec()
        env.step(action_spec.sample())
        env.step(action_spec.sample())
        with self.assertRaises(Exception):
            env.step(action_spec.sample())


class MockEnvironmentCrashInInit(object):
    """Raise an error when instantiated."""

    def __init__(self, *unused_args, **unused_kwargs):
        raise RuntimeError()

    def action_spec(self):
        return []


class MockEnvironmentCrashInReset(object):
    """Raise an error when instantiated."""

    def __init__(self, *unused_args, **unused_kwargs):
        pass

    def action_spec(self):
        return []

    def _reset(self):
        raise RuntimeError()


class MockEnvironmentCrashInStep(RandomAlfEnvironment):
    """Raise an error after specified number of steps in an episode."""

    def __init__(self, crash_at_step, env_id=None):
        super(MockEnvironmentCrashInStep, self).__init__(
            observation_spec=ts.TensorSpec((3, 3), torch.float32),
            action_spec=ts.BoundedTensorSpec([1],
                                             torch.float32,
                                             minimum=-1.0,
                                             maximum=1.0),
            env_id=env_id,
            episode_end_probability=0,
            min_duration=crash_at_step + 1,
            max_duration=crash_at_step + 1)
        self._crash_at_step = crash_at_step
        self._steps = 0

    def _step(self, *args, **kwargs):
        transition = super(MockEnvironmentCrashInStep, self)._step(
            *args, **kwargs)
        self._steps += 1
        if self._steps == self._crash_at_step:
            raise RuntimeError()
        return transition


if __name__ == '__main__':
    alf.test.main()
