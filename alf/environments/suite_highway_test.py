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

from absl.testing import parameterized
import functools
import torch

import alf
from alf.environments import suite_highway, alf_environment
from alf.environments import thread_environment, parallel_environment
import alf.nest as nest
from alf.tensor_specs import BoundedTensorSpec


class SuiteHighwayTest(parameterized.TestCase, alf.test.TestCase):
    def setUp(self):
        super().setUp()
        if not suite_highway.is_available():
            self.skipTest('suite_highway is not available.')

    def tearDown(self):
        super().tearDown()
        self._env.close()

    def test_unwrapped_env(self):
        self._env = suite_highway.load(environment_name='highway-v0')
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self.assertEqual(torch.float32, self._env.observation_spec().dtype)

        actions = self._env.action_spec().sample()
        for _ in range(10):
            # unwrapped env (not thread_env or parallel_env) needs to convert
            # from tensor to array
            time_step = self._env.step(actions.cpu().numpy())

    def test_thread_env(self):
        self._env = thread_environment.ThreadEnvironment(
            lambda: suite_highway.load(environment_name='highway-v0'))
        self.assertIsInstance(self._env, alf_environment.AlfEnvironment)
        self.assertEqual(torch.float32, self._env.observation_spec().dtype)

        actions = self._env.action_spec().sample()
        for _ in range(10):
            time_step = self._env.step(actions)

    def test_parallel_env(self):
        env_num = 5

        def ctor(env_id=None):
            return suite_highway.load(environment_name='highway-v0')

        constructor = functools.partial(ctor)

        self._env = parallel_environment.ParallelAlfEnvironment(
            [constructor] * env_num)

        self.assertTrue(self._env.batched)
        self.assertEqual(self._env.batch_size, env_num)
        self.assertEqual(torch.float32, self._env.observation_spec().dtype)

        actions = self._env.action_spec().sample(outer_dims=(env_num, ))
        for _ in range(10):
            time_step = self._env.step(actions)

    def test_env_info(self):
        self._env = suite_highway.load(environment_name="highway-v0")
        actions = self._env.action_spec().sample()
        time_step = self._env.step(actions.cpu().numpy())
        env_info = time_step.env_info
        for field in env_info:
            self.assertEqual(env_info[field].size, 1)
        self.assertFalse('action' in time_step.env_info.keys())

    def test_env_config(self):

        env = suite_highway.load(environment_name="highway-v0")
        self.assertEqual(env.observation_spec().shape, (35, ))
        self.assertTrue(env.action_spec().is_continuous)
        self.assertEqual(env.action_spec().shape, (2, ))
        env.close()

        # test env with specified config
        env_config = {
            "observation": {
                "type":
                    "Kinematics",
                "vehicles_count":
                    3,
                "features": [
                    "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"
                ],
            },
            "action": {
                "type": "DiscreteMetaAction"
            }
        }

        env = suite_highway.load(
            environment_name="highway-v0", env_config=env_config)
        self.assertEqual(env.observation_spec().shape, (21, ))
        self.assertTrue(env.action_spec().is_discrete)
        self.assertEqual(env.action_spec().numel, 1)

        actions = env.action_spec().sample().cpu().numpy()
        for _ in range(10):
            time_step = env.step(actions)

        self._env = env

    @parameterized.parameters(5, 50)
    def test_last_step(self, max_episode_steps):
        env_config = {
            "observation": {
                "type":
                    "Kinematics",
                "vehicles_count":
                    3,
                "features": [
                    "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"
                ],
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "duration": max_episode_steps
        }

        self._env = suite_highway.load(
            environment_name="highway-v0", env_config=env_config)

        for i in range(max_episode_steps):
            actions = self._env.action_spec().sample().cpu().numpy()
            time_step = self._env.step(actions)
            if time_step.step_type == 2:
                break

        if time_step.env_info['crashed'].item() is True:
            assert time_step.discount == 0.0

        if i == max_episode_steps - 1 and not time_step.env_info[
                'crashed'].item():
            assert time_step.discount == 1.0


if __name__ == '__main__':
    alf.test.main()
