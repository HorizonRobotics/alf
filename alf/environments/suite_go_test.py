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
"""Tests for suite_go.py."""

import torch

import alf
from alf.data_structures import StepType
from alf.environments.suite_go import GoBoard, load


class BoardTest(alf.test.TestCase):
    def _get_test_cases(self):
        # yapf: disable
        sequence1 = [
            (0, 0, [[-1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0]]),
            (0, 1, [[-1,  1,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0]]),
            (1, 1, [[-1,  1,  0,  0],
                    [ 0, -1,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0]]),
            (1, 0, [[ 0,  1,  0,  0],
                    [ 1, -1,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0]]),
            (0, 2, [[ 0,  1, -1,  0],
                    [ 1, -1,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0]]),
            (2, 0, [[ 0,  1, -1,  0],
                    [ 1, -1,  0,  0],
                    [ 1,  0,  0,  0],
                    [ 0,  0,  0,  0]]),
            (0, 0, [[-1,  0, -1,  0],
                    [ 1, -1,  0,  0],
                    [ 1,  0,  0,  0],
                    [ 0,  0,  0,  0]]),
            (2, 1, [[-1,  0, -1,  0],
                    [ 1, -1,  0,  0],
                    [ 1,  1,  0,  0],
                    [ 0,  0,  0,  0]]),
            (2, 2, [[-1,  0, -1,  0],
                    [ 1, -1,  0,  0],
                    [ 1,  1, -1,  0],
                    [ 0,  0,  0,  0]]),
            (1, 2, [[-1,  0, -1,  0],
                    [ 1, -1,  1,  0],
                    [ 1,  1, -1,  0],
                    [ 0,  0,  0,  0]]),
            (3, 2, [[-1,  0, -1,  0],
                    [ 1, -1,  1,  0],
                    [ 1,  1, -1,  0],
                    [ 0,  0, -1,  0]]),
            (0, 1, [[ 0,  1, -1,  0],
                    [ 1,  0,  1,  0],
                    [ 1,  1, -1,  0],
                    [ 0,  0, -1,  0]]),
            (0, 0, [[ 0,  1, -1,  0],
                    [ 1,  0,  1,  0],
                    [ 1,  1, -1,  0],
                    [ 0,  0, -1,  0]]),
        ]
        area1 = (3, 7)
        invalid1 = True         # The last move is invalid

        sequence2 = [
            (3, 3, [[ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0, -1]]),
            (0, 0, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0, -1]]),
            (3, 1, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0, -1,  0, -1]]),
            (2, 1, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  1,  0,  0],
                    [ 0, -1,  0, -1]]),
            (3, 2, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  1,  0,  0],
                    [ 0, -1, -1, -1]]),
            (2, 2, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  1,  1,  0],
                    [ 0, -1, -1, -1]]),
            (3, 0, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  1,  1,  0],
                    [-1, -1, -1, -1]]),
            (2, 0, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 1,  1,  1,  0],
                    [-1, -1, -1, -1]]),
            (2, 3, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 1,  1,  1, -1],
                    [-1, -1, -1, -1]]),
            (1, 3, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  1],
                    [ 1,  1,  1,  0],
                    [ 0,  0,  0,  0]]),
            (1, 0, [[ 1,  0,  0,  0],
                    [-1,  0,  0,  1],
                    [ 1,  1,  1,  0],
                    [ 0,  0,  0,  0]]),
            (1, 1, [[ 1,  0,  0,  0],
                    [ 0,  1,  0,  1],
                    [ 1,  1,  1,  0],
                    [ 0,  0,  0,  0]]),
            (1, 0, [[ 1,  0,  0,  0],
                    [ 0,  1,  0,  1],
                    [ 1,  1,  1,  0],
                    [ 0,  0,  0,  0]]),
        ]
        area2 = (0, 7)
        invalid2 = True         # The last move is invalid

        sequence3 = [
            (3, 0, [[ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [-1,  0,  0,  0]]),
            (0, 0, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [-1,  0,  0,  0]]),
            (3, 1, [[ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [-1, -1,  0,  0]]),
            (1, 0, [[ 1,  0,  0,  0],
                    [ 1,  0,  0,  0],
                    [ 0,  0,  0,  0],
                    [-1, -1,  0,  0]]),
            (2, 0, [[ 1,  0,  0,  0],
                    [ 1,  0,  0,  0],
                    [-1,  0,  0,  0],
                    [-1, -1,  0,  0]]),
            (2, 1, [[ 1,  0,  0,  0],
                    [ 1,  0,  0,  0],
                    [-1,  1,  0,  0],
                    [-1, -1,  0,  0]]),
            (1, 1, [[ 1,  0,  0,  0],
                    [ 1, -1,  0,  0],
                    [-1,  1,  0,  0],
                    [-1, -1,  0,  0]]),
            (3, 2, [[ 1,  0,  0,  0],
                    [ 1, -1,  0,  0],
                    [ 0,  1,  0,  0],
                    [ 0,  0,  1,  0]]),
            (0, 1, [[ 1, -1,  0,  0],
                    [ 1, -1,  0,  0],
                    [ 0,  1,  0,  0],
                    [ 0,  0,  1,  0]]),
            (0, 2, [[ 1, -1,  1,  0],
                    [ 1, -1,  0,  0],
                    [ 0,  1,  0,  0],
                    [ 0,  0,  1,  0]]),
            (2, 0, [[ 0, -1,  1,  0],
                    [ 0, -1,  0,  0],
                    [-1,  1,  0,  0],
                    [ 0,  0,  1,  0]]),
            (1, 2, [[ 0, -1,  1,  0],
                    [ 0, -1,  1,  0],
                    [-1,  1,  0,  0],
                    [ 0,  0,  1,  0]]),
            (1, 0, [[ 0, -1,  1,  0],
                    [-1, -1,  1,  0],
                    [-1,  1,  0,  0],
                    [ 0,  0,  1,  0]]),
        ]
        area3 = (5, 4)
        invalid3 = False         # The last move is valid
        # yapf: enable

        sequences = [sequence1, sequence2, sequence3]
        areas = [area1, area2, area3]
        invalids = [invalid1, invalid2, invalid3]
        return sequences, areas, invalids

    def test_board(self):
        sequences, areas, invalids = self._get_test_cases()
        for sequence, expected_area, expected_invalid in zip(
                sequences, areas, invalids):
            board = GoBoard(1, 4, 4)
            player = torch.tensor(-1, dtype=torch.int8)
            for y, x, expected in sequence:
                board_indices = torch.tensor([0], dtype=torch.int64)
                y = torch.tensor([y])
                x = torch.tensor([x])
                expected = torch.tensor(expected, dtype=torch.int8)
                invalid = board.update(board_indices, y, x, player)
                self.assertEqual(board.get_board().squeeze(0), expected)
                player = -player

            self.assertEqual(invalid[0], expected_invalid)
            self.assertEqual(board.calc_area(), expected_area)

        board = GoBoard(len(areas), 4, 4)
        player = torch.tensor([-1] * len(areas), dtype=torch.int8)
        for case in zip(*sequences):
            y, x, expected = list(zip(*case))
            board_indices = torch.arange(len(y))
            y = torch.tensor(y)
            x = torch.tensor(x)
            expected = torch.tensor(expected, dtype=torch.int8)
            invalid = board.update(board_indices, y, x, player)
            self.assertEqual(board.get_board(), expected)
            player = -player

        self.assertEqual(invalid, torch.tensor(invalids))
        area = list(zip(*areas))
        self.assertEqual(board.calc_area()[0], torch.tensor(area[0]))
        self.assertEqual(board.calc_area()[1], torch.tensor(area[1]))

    def test_environment(self):
        sequences, areas, invalids = self._get_test_cases()
        batch_size = len(sequences)
        env = load(batch_size=batch_size, height=4, width=4, winning_thresh=0)
        time_step = env.reset()
        to_play = torch.tensor([0] * batch_size, dtype=torch.int8)
        self.assertEqual(time_step.observation['board'],
                         torch.zeros(batch_size, 1, 4, 4))
        self.assertEqual(time_step.observation['to_play'],
                         torch.zeros(batch_size))
        self.assertEqual(time_step.step_type,
                         torch.tensor([StepType.FIRST] * batch_size))
        for case in zip(*sequences):
            y, x, expected = list(zip(*case))
            y = torch.tensor(y, dtype=torch.int64)
            x = torch.tensor(x, dtype=torch.int64)
            expected = torch.tensor(expected, dtype=torch.int8).unsqueeze(1)
            to_play = 1 - to_play
            time_step = env.step(action=4 * y + x)
            self.assertEqual(time_step.observation['board'], expected)
            self.assertEqual(time_step.observation['to_play'], to_play)

        self.assertEqual(
            time_step.step_type,
            torch.tensor([StepType.LAST, StepType.LAST, StepType.MID]))
        self.assertEqual(time_step.reward, torch.tensor([-1., -1., 0.]))
        self.assertEqual(time_step.discount, torch.tensor([0., 0., 1.]))

        # Test game 0 and 1 are restarted
        time_step = env.step(
            action=torch.tensor([3, 4, 16], dtype=torch.int64))
        self.assertEqual(
            time_step.step_type,
            torch.tensor([StepType.FIRST, StepType.FIRST, StepType.MID]))
        self.assertEqual(time_step.reward, torch.tensor([0., 0., 0.]))
        self.assertEqual(time_step.discount, torch.tensor([1., 1., 1.]))
        self.assertEqual(time_step.observation['to_play'],
                         torch.tensor([0, 0, 0], dtype=torch.int8))
        self.assertEqual(time_step.observation['board'][0:2],
                         torch.zeros(2, 1, 4, 4))
        # board does not change after pass
        self.assertEqual(time_step.observation['board'][2],
                         torch.tensor(sequences[2][-1][2]).unsqueeze(0))

        # Both pass
        time_step = env.step(
            action=torch.tensor([4, 5, 16], dtype=torch.int64))
        self.assertEqual(
            time_step.step_type,
            torch.tensor([StepType.MID, StepType.MID, StepType.LAST]))
        # player 0 wins
        self.assertEqual(time_step.reward, torch.tensor([0., 0., 1.]))
        # board does not change after pass
        self.assertEqual(time_step.observation['board'][2],
                         torch.tensor(sequences[2][-1][2]).unsqueeze(0))


if __name__ == '__main__':
    alf.test.main()
