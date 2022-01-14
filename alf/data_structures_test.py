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
"""Test cases adapted from tf_agents policy_step_test.py and time_step_test.py."""

import unittest
import numpy as np
import torch

import alf
from alf.nest import flatten, map_structure
from alf.data_structures import AlgStep, StepType, TimeStep, Experience
from alf.experience_replayers.replay_buffer_test import get_exp_batch


class AlgStepTest(unittest.TestCase):
    def testCreate(self):
        action = torch.tensor(1)
        state = torch.tensor(2)
        info = torch.tensor(3)
        step = AlgStep(output=action, state=state, info=info)
        self.assertEqual(step.output, action)
        self.assertEqual(step.state, state)
        self.assertEqual(step.info, info)

    def testCreateWithAllDefaults(self):
        action = torch.tensor(1)
        state = ()
        info = ()
        step = AlgStep(output=action)
        self.assertEqual(step.output, action)
        self.assertEqual(step.state, state)
        self.assertEqual(step.info, info)

    def testCreateWithDefaultInfo(self):
        action = torch.tensor(1)
        state = torch.tensor(2)
        info = ()
        step = AlgStep(output=action, state=state)
        self.assertEqual(step.output, action)
        self.assertEqual(step.state, state)
        self.assertEqual(step.info, info)


class TimeStepTest(unittest.TestCase):
    def testCreate(self):
        step_type = torch.tensor(0, dtype=torch.int32)
        reward = torch.tensor(1, dtype=torch.int32)
        discount = 0.99
        observation = torch.tensor(-1)
        prev_action = torch.tensor(-1)
        env_id = torch.tensor(0, dtype=torch.int32)
        time_step = TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation,
            prev_action=prev_action,
            env_id=env_id)
        self.assertEqual(StepType.FIRST, time_step.step_type)
        self.assertEqual(reward, time_step.reward)
        self.assertEqual(discount, time_step.discount)
        self.assertEqual(observation, time_step.observation)
        self.assertEqual(prev_action, time_step.prev_action)
        self.assertEqual(env_id, time_step.env_id)


class ExperienceTest(alf.test.TestCase):
    def test_map_structure_on_experience(self):
        exp = get_exp_batch([0, 4, 7], 10, t=1, x=0.5)
        func = lambda x: x + 1
        res = map_structure(func, exp)

        map_structure(lambda x, y: self.assertTensorClose(func(x), y), exp,
                      res)

        flat_exp = flatten(exp)
        flat_res = map_structure(func, flat_exp)
        map_structure(lambda x, y: self.assertTensorClose(x, y), flatten(res),
                      flat_res)


if __name__ == '__main__':
    unittest.main()
