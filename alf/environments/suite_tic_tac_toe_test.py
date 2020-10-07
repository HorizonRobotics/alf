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

import torch

import alf
from alf.data_structures import StepType
from alf.environments.suite_tic_tac_toe import load


class TicTacToeTest(alf.test.TestCase):
    def test_tic_tac_toe(self):
        env = load(batch_size=2)
        time_step = env.reset()
        self.assertEqual(time_step.observation, torch.zeros(2, 3, 3))
        time_step = env.step(torch.tensor([4, 5]))
        self.assertEqual(time_step.reward, torch.zeros(2))
        self.assertEqual(time_step.prev_action, torch.tensor([4, 5]))
        self.assertEqual(
            time_step.observation,
            torch.tensor([
                [[0, 0, 0], [0, -1, 0], [0, 0, 0.]],
                [[0, 0, 0], [0, 0, -1], [0, 0, 0.]],
            ]))

        time_step = env.step(torch.tensor([3, 4]))
        self.assertEqual(time_step.reward, torch.zeros(2))
        self.assertEqual(
            time_step.observation,
            torch.tensor([
                [[0, 0, 0], [1, -1, 0], [0, 0, 0.]],
                [[0, 0, 0], [0, 1, -1], [0, 0, 0.]],
            ]))

        time_step = env.step(torch.tensor([0, 2]))
        self.assertEqual(time_step.step_type,
                         torch.tensor([StepType.MID, StepType.MID]))
        self.assertEqual(time_step.reward, torch.zeros(2))
        self.assertEqual(
            time_step.observation,
            torch.tensor([
                [[-1, 0, 0], [1, -1, 0], [0, 0, 0.]],
                [[0, 0, -1], [0, 1, -1], [0, 0, 0.]],
            ]))

        time_step = env.step(torch.tensor([8, 2]))
        self.assertEqual(time_step.discount, torch.tensor([1., 0.]))
        self.assertEqual(time_step.step_type,
                         torch.tensor([StepType.MID, StepType.LAST]))
        self.assertEqual(time_step.reward, torch.tensor([0., 1.]))
        self.assertEqual(
            time_step.observation,
            torch.tensor([
                [[-1, 0, 0], [1, -1, 0], [0, 0, 1.]],
                [[0, 0, -1], [0, 1, -1], [0, 0, 0.]],
            ]))

        time_step = env.step(torch.tensor([1, 3]))
        self.assertEqual(time_step.discount, torch.tensor([1., 1.]))
        self.assertEqual(time_step.prev_action, torch.tensor([1, 0]))
        self.assertEqual(time_step.step_type,
                         torch.tensor([StepType.MID, StepType.FIRST]))
        self.assertEqual(time_step.reward, torch.tensor([0., 0.]))
        self.assertEqual(
            time_step.observation,
            torch.tensor([
                [[-1, -1, 0], [1, -1, 0], [0, 0, 1.]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0.]],
            ]))

        time_step = env.step(torch.tensor([6, 3]))
        self.assertEqual(time_step.step_type,
                         torch.tensor([StepType.MID, StepType.MID]))
        self.assertEqual(time_step.reward, torch.tensor([0., 0.]))
        self.assertEqual(
            time_step.observation,
            torch.tensor([
                [[-1, -1, 0], [1, -1, 0], [1, 0, 1.]],
                [[0, 0, 0], [-1, 0, 0], [0, 0, 0.]],
            ]))

        time_step = env.step(torch.tensor([2, 4]))
        self.assertEqual(time_step.discount, torch.tensor([0., 1.]))
        self.assertEqual(time_step.step_type,
                         torch.tensor([StepType.LAST, StepType.MID]))
        self.assertEqual(time_step.reward, torch.tensor([1., 0.]))
        self.assertEqual(
            time_step.observation,
            torch.tensor([
                [[-1, -1, -1], [1, -1, 0], [1, 0, 1.]],
                [[0, 0, 0], [-1, 1, 0], [0, 0, 0.]],
            ]))


if __name__ == '__main__':
    alf.test.main()
