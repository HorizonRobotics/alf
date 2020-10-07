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
"""Tests for the thread_environment. """

import collections
import functools
import numpy as np
import time
import torch

import alf
import alf.data_structures as ds
from alf.environments.random_alf_environment import RandomAlfEnvironment
from alf.environments.thread_environment import ThreadEnvironment
import alf.tensor_specs as ts


class SlowStartingEnvironment(RandomAlfEnvironment):
    def __init__(self, *args, **kwargs):
        time_sleep = kwargs.pop('time_sleep', 1.0)
        time.sleep(time_sleep)
        super(SlowStartingEnvironment, self).__init__(*args, **kwargs)


class ThreadEnvironmentTest(alf.test.TestCase):
    def _set_default_specs(self):
        self.observation_spec = ts.TensorSpec((3, 3), torch.float32)
        self.action_spec = ts.BoundedTensorSpec([7],
                                                dtype=torch.float32,
                                                minimum=-1.0,
                                                maximum=1.0)
        self.time_step_spec = ds.time_step_spec(self.observation_spec,
                                                self.action_spec,
                                                ts.TensorSpec(()))

    def _make_thread_environment(self, constructor=None):
        self._set_default_specs()
        constructor = constructor or functools.partial(
            RandomAlfEnvironment, self.observation_spec, self.action_spec)
        return ThreadEnvironment(constructor)

    def test_close_no_hang_after_init(self):
        env = self._make_thread_environment()
        env.close()

    def test_get_specs(self):
        env = self._make_thread_environment()
        self.assertEqual(self.observation_spec, env.observation_spec())
        self.assertEqual(self.time_step_spec, env.time_step_spec())
        self.assertEqual(self.action_spec, env.action_spec())
        env.close()

    def test_step(self):
        env = self._make_thread_environment()
        action_spec = env.action_spec()
        observation_spec = env.observation_spec()
        action = action_spec.sample()
        env.reset()

        # Take one step and assert observation is batched the right way.
        time_step = env.step(action)
        self.assertEqual(observation_spec.shape,
                         time_step.observation.shape[1:])
        self.assertEqual(torch.Size(action_spec.shape), action.shape)

        # Take another step and assert that observations have the same shape.
        time_step2 = env.step(action)
        self.assertEqual(time_step.observation.shape,
                         time_step2.observation.shape)
        env.close()


if __name__ == '__main__':
    alf.test.main()
