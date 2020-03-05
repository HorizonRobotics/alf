# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

from absl import logging
from absl.testing import parameterized
import numpy as np
import torch

import alf
from alf.data_structures import TimeStep, StepType
from alf.environments.suite_unittest import ActionType
from alf.environments.suite_unittest import ValueUnittestEnv
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import RNNPolicyUnittestEnv


class SuiteUnittestEnvTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(ActionType.Discrete, ActionType.Continuous)
    def test_value_unittest_env(self, action_type):
        batch_size = 1
        steps_per_episode = 13

        env = ValueUnittestEnv(
            batch_size, steps_per_episode, action_type=action_type)

        time_step = env.reset()
        for _ in range(10):
            for s in range(steps_per_episode):
                if s == 0:
                    step_type = StepType.FIRST
                    discount = 1.0
                elif s == steps_per_episode - 1:
                    step_type = StepType.LAST
                    discount = 0.0
                else:
                    step_type = StepType.MID
                    discount = 1.0

                self.assertEqual(time_step.step_type,
                                 torch.full([batch_size], step_type))
                self.assertEqual(time_step.reward, torch.ones(batch_size))
                self.assertEqual(time_step.discount,
                                 torch.full([batch_size], discount))

                action = torch.randint(0, 2, (batch_size, 1))
                time_step = env.step(action)

    @parameterized.parameters(ActionType.Discrete, ActionType.Continuous)
    def test_policy_unittest_env(self, action_type):
        batch_size = 100
        steps_per_episode = 13

        env = PolicyUnittestEnv(
            batch_size, steps_per_episode, action_type=action_type)

        time_step = env.reset()
        for _ in range(10):
            for s in range(steps_per_episode):
                if s == 0:
                    step_type = StepType.FIRST
                    discount = 1.0
                elif s == steps_per_episode - 1:
                    step_type = StepType.LAST
                    discount = 0.0
                else:
                    step_type = StepType.MID
                    discount = 1.0

                if s == 0:
                    reward = torch.zeros(batch_size)
                else:
                    reward = (action == prev_observation.to(torch.int64)).to(
                        torch.float32)
                    reward = reward.reshape(batch_size)

                self.assertEqual(time_step.step_type,
                                 torch.full([batch_size], step_type))
                self.assertEqual(time_step.reward, reward)
                self.assertEqual(time_step.discount,
                                 torch.full([batch_size], discount))

                action = torch.randint(0, 2, (batch_size, 1))
                prev_observation = time_step.observation

                time_step = env.step(action)

    def test_rnn_policy_unittest_env(self):
        batch_size = 100
        steps_per_episode = 5
        gap = 3
        env = RNNPolicyUnittestEnv(batch_size, steps_per_episode, gap)

        time_step = env.reset()
        for _ in range(10):
            for s in range(steps_per_episode):
                if s == 0:
                    observation0 = time_step.observation

                if s == 0:
                    step_type = StepType.FIRST
                    discount = 1.0
                elif s == steps_per_episode - 1:
                    step_type = StepType.LAST
                    discount = 0.0
                else:
                    step_type = StepType.MID
                    discount = 1.0

                if s <= gap:
                    reward = torch.zeros(batch_size)
                else:
                    reward = (2 * action - 1 == observation0.to(
                        torch.int64)).to(torch.float32)
                    reward = reward.reshape(batch_size)

                self.assertEqual(time_step.step_type,
                                 torch.full([batch_size], step_type))
                self.assertEqual(time_step.reward, reward)
                self.assertEqual(time_step.discount,
                                 torch.full([batch_size], discount))

                action = torch.randint(0, 2, (batch_size, 1))
                time_step = env.step(action)


if __name__ == '__main__':
    alf.test.main()
