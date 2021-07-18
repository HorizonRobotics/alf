# Copyright (c) 2021 Horizon Robotics. All Rights Reserved.
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

import functools
import torch
from bsuite import sweep

import alf
from alf.environments import suite_bsuite, alf_environment
from alf.environments import thread_environment, parallel_environment
import alf.nest as nest


class SuiteBSuiteTest(alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_bsuite.is_available():
            self.skipTest('suite_safety_gym is not available.')

    def test_reset(self):
        self._env = suite_bsuite.load(
            environment_name=sweep.CARTPOLE_SWINGUP[0])
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self.assertEqual(torch.float32, self._env.observation_spec().dtype)
        self.assertEqual(self._env.reset().observation.ndim, 1)

    def test_step(self):
        self._env = suite_bsuite.load(
            environment_name=sweep.CARTPOLE_SWINGUP[0])
        actions = self._env.action_spec().sample()
        for _ in range(10):
            time_step = self._env.step(actions.item())
        self.assertEqual(time_step.observation.ndim, 1)

    def test_parallel_env(self):
        env_num = 8

        def ctor(env_id=None):
            return suite_bsuite.load(
                environment_name=sweep.CARTPOLE_SWINGUP[0])

        constructor = functools.partial(ctor)

        self._env = parallel_environment.ParallelAlfEnvironment(
            [constructor] * env_num)
        self.assertTrue(self._env.batched)
        self.assertEqual(self._env.batch_size, env_num)
        self.assertEqual(torch.float32, self._env.observation_spec().dtype)

        actions = self._env.action_spec().sample(outer_dims=(env_num, ))
        for _ in range(10):
            time_step = self._env.step(actions)


if __name__ == '__main__':
    alf.test.main()
