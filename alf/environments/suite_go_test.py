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
"""Tests for suite_go.py."""

import torch

import alf
from alf.data_structures import StepType
from alf.environments.suite_go import GoBoard, GoEnvironment


class BoardTest(alf.test.TestCase):
    def _get_test_cases(self):
        # yapf: disable
        sequence1 = [
            dict(y=0, x=0,
                board='x   '
                      '    '
                      '    '
                      '    '),
            dict(y=0, x=1,
                board='x*  '
                      '    '
                      '    '
                      '    '),
            dict(y=1, x=1,
                board='x*  '
                      ' x  '
                      '    '
                      '    '),
            dict(y=1, x=0,
                 board=' *  '
                       '*x  '
                       '    '
                       '    '),
            dict(y=0, x=2,
                 board=' *x '
                       '*x  '
                       '    '
                       '    ',
                 suicidal=[[1, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]],
                 repeated=[[1, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]]),
            dict(y=2, x=0,
                 board=' *x '
                       '*x  '
                       '*   '
                       '    '),
            dict(y=0, x=0,
                 board='x x '
                       '*x  '
                       '*   '
                       '    '),
            dict(y=2, x=1,
                 board='x x '
                       '*x  '
                       '**  '
                       '    ',
                 repeated=[[0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]]),
            dict(y=2, x=2,
                 board='x x '
                       '*x  '
                       '**x '
                       '    '),
            dict(y=1, x=2,
                 board='x x '
                       '*x* '
                       '**x '
                       '    '),
            dict(y=3, x=2,
                 board='x x '
                       '*x* '
                       '**x '
                       '  x '),
            dict(y=0, x=1,
                 board=' *x '
                       '* * '
                       '**x '
                       '  x '),
            dict(y=0, x=0,
                 board=' *x '
                       '* * '
                       '**x '
                       '  x ',
                 suicidal=[[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]],
                 repeated=[[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]]),
        ]
        area1 = (3, 7)
        invalid1 = True         # The last move is invalid

        sequence2 = [
            dict(y=3, x=3,
                 board='    '
                       '    '
                       '    '
                       '   x'),
            dict(y=0, x=0,
                 board='*   '
                       '    '
                       '    '
                       '   x'),
            dict(y=3, x=1,
                 board='*   '
                       '    '
                       '    '
                       ' x x'),
            dict(y=2, x=1,
                 board='*   '
                       '    '
                       ' *  '
                       ' x x'),
            dict(y=3, x=2,
                 board='*   '
                       '    '
                       ' *  '
                       ' xxx'),
            dict(y=2, x=2,
                 board='*   '
                       '    '
                       ' ** '
                       ' xxx'),
            dict(y=3, x=0,
                 board='*   '
                       '    '
                       ' ** '
                       'xxxx'),
            dict(y=2, x=0,
                 board='*   '
                       '    '
                       '*** '
                       'xxxx'),
            dict(y=1, x=0,
                 board='*   '
                       'x   '
                       '*** '
                       'xxxx'),
            dict(y=1, x=3,
                 board='*   '
                       'x  *'
                       '*** '
                       'xxxx'),
            dict(y=0, x=1,
                 board=' x  '
                       'x  *'
                       '*** '
                       'xxxx',
                 suicidal=[[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]]),
            dict(y=1, x=1,
                 board=' x  '
                       'x* *'
                       '*** '
                       'xxxx',
                 suicidal=[[1, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]],
                 repeated=[[1, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]]),
            dict(y=2, x=3,
                 board=' x  '
                       'x* *'
                       '*** '
                       '    ',
                 suicidal=[[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]]),
        ]
        area2 = (3, 10)
        invalid2 = True         # The last move is invalid

        sequence3 = [
            dict(y=3, x=0,
                 board='    '
                       '    '
                       '    '
                       'x   '),
            dict(y=0, x=0,
                 board='*   '
                       '    '
                       '    '
                       'x   '),
            dict(y=3, x=1,
                 board='*   '
                       '    '
                       '    '
                       'xx  '),
            dict(y=1, x=0,
                 board='*   '
                       '*   '
                       '    '
                       'xx  '),
            dict(y=2, x=0,
                 board='*   '
                       '*   '
                       'x   '
                       'xx  '),
            dict(y=2, x=1,
                 board='*   '
                       '*   '
                       'x*  '
                       'xx  '),
            dict(y=1, x=1,
                 board='*   '
                       '*x  '
                       'x*  '
                       'xx  '),
            dict(y=3, x=2,
                 board='*   '
                       '*x  '
                       ' *  '
                       '  * '),
            dict(y=0, x=1,
                 board='*x  '
                       '*x  '
                       ' *  '
                       '  * '),
            dict(y=0, x=2,
                 board='*x* '
                       '*x  '
                       ' *  '
                       '  * '),
            dict(y=2, x=0,
                 board=' x* '
                       ' x  '
                       'x*  '
                       '  * '),
            dict(y=1, x=2,
                 board=' x* '
                       ' x* '
                       'x*  '
                       '  * '),
            dict(y=1, x=0,
                 board=' x* '
                       'xx* '
                       'x*  '
                       '  * '),
        ]
        area3 = (5, 9)
        invalid3 = False         # The last move is valid
        # yapf: enable

        sequences = [sequence1, sequence2, sequence3]
        areas = [area1, area2, area3]
        invalids = [invalid1, invalid2, invalid3]
        for seq in sequences:
            for step in seq:
                board = list(step['board'])
                board = [
                    0 if c == ' ' else (-1 if c == 'x' else 1) for c in board
                ]
                step['y'] = torch.tensor(step['y'], dtype=torch.int64)
                step['x'] = torch.tensor(step['x'], dtype=torch.int64)
                step['board'] = torch.tensor(
                    board, dtype=torch.int8).reshape(4, 4)
                if 'suicidal' in step:
                    suicidal = torch.tensor(step['suicidal'], dtype=torch.bool)
                else:
                    suicidal = torch.zeros((4, 4), dtype=torch.bool)
                step['suicidal'] = suicidal
                if 'repeated' in step:
                    repeated = torch.tensor(step['repeated'], dtype=torch.bool)
                else:
                    repeated = torch.zeros((4, 4), dtype=torch.bool)
                step['repeated'] = repeated

        return sequences, areas, invalids

    def test_board(self):
        sequences, areas, invalids = self._get_test_cases()
        for sequence, expected_area, expected_invalid in zip(
                sequences, areas, invalids):
            board = GoBoard(1, 4, 4, 32)
            player = torch.tensor([-1], dtype=torch.int8)
            for step in sequence:
                y, x, expected = step['y'], step['x'], step['board']
                board_indices = torch.tensor([0], dtype=torch.int64)
                print(expected)
                occupied, suicidal, repeated = board.classify_all_moves(player)
                self.assertEqual(occupied, board.get_board() != 0)
                self.assertEqual(suicidal[0], step['suicidal'])
                self.assertEqual(repeated[0], step['repeated'])
                invalid = board.update(board_indices, y, x, player)
                self.assertEqual(board.get_board()[0], expected)
                player = -player

            self.assertEqual(invalid[0], expected_invalid)
            self.assertEqual(board.calc_area(), expected_area)

        board = GoBoard(len(areas), 4, 4, 32)
        player = torch.tensor([-1] * len(areas), dtype=torch.int8)

        length = len(sequences[0])
        for i in range(length):
            step = alf.nest.map_structure(lambda *x: torch.stack(x),
                                          *[seq[i] for seq in sequences])
            board_indices = torch.arange(len(sequences))
            occupied, suicidal, repeated = board.classify_all_moves(player)
            self.assertEqual(occupied, board.get_board() != 0)
            self.assertEqual(suicidal, step['suicidal'])
            self.assertEqual(repeated, step['repeated'])

            invalid = board.update(board_indices, step['y'], step['x'], player)
            self.assertEqual(board.get_board(), step['board'])
            player = -player

        self.assertEqual(invalid, torch.tensor(invalids))
        area = list(zip(*areas))
        self.assertEqual(board.calc_area()[0], torch.tensor(area[0]))
        self.assertEqual(board.calc_area()[1], torch.tensor(area[1]))

    def test_environment(self):
        sequences, areas, invalids = self._get_test_cases()
        batch_size = len(sequences)
        env = GoEnvironment(
            batch_size=batch_size,
            height=4,
            width=4,
            winning_thresh=0,
            allow_suicidal_move=False)
        time_step = env.reset()
        to_play = torch.tensor([0] * batch_size, dtype=torch.int8)
        self.assertEqual(time_step.observation['board'],
                         torch.zeros(batch_size, 1, 4, 4))
        self.assertEqual(time_step.observation['to_play'],
                         torch.zeros(batch_size))
        self.assertEqual(time_step.step_type,
                         torch.tensor([StepType.FIRST] * batch_size))
        length = len(sequences[0])
        for i in range(length):
            step = alf.nest.map_structure(lambda *x: torch.stack(x),
                                          *[seq[i] for seq in sequences])
            y, x, expected = step['y'], step['x'], step['board']
            expected = expected.unsqueeze(1)
            to_play = 1 - to_play
            time_step = env.step(action=4 * y + x)
            self.assertEqual(time_step.observation['prev_action'], 4 * y + x)
            self.assertEqual(time_step.prev_action, 4 * y + x)
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
        # board[2] does not change after pass
        self.assertEqual(time_step.observation['board'][2][0],
                         sequences[2][-1]['board'])

        # board[2] both pass
        time_step = env.step(
            action=torch.tensor([4, 5, 16], dtype=torch.int64))
        self.assertEqual(
            time_step.step_type,
            torch.tensor([StepType.MID, StepType.MID, StepType.LAST]))
        # player 1 wins
        self.assertEqual(time_step.reward, torch.tensor([0., 0., -1.]))
        # board does not change after pass
        self.assertEqual(time_step.observation['board'][2][0],
                         sequences[2][-1]['board'])

    def test_environment2(self):
        """Test whether too_long is correctly handled."""
        height = 4
        width = 4
        env = GoEnvironment(
            batch_size=1,
            height=height,
            width=width,
            winning_thresh=0,
            allow_suicidal_move=False)
        env.reset()
        pass_action = height * width
        for i in range(height * width * 2 - 3):
            if i % 2 == 0:
                action = i // 2
            else:
                action = pass_action
            time_step = env.step(
                action=torch.tensor([action], dtype=torch.int64))
            self.assertEqual(time_step.reward[0], 0.)
            self.assertEqual(time_step.step_type[0], StepType.MID)
        time_step = env.step(
            action=torch.tensor([height * width - 1], dtype=torch.int64))
        time_step = env.step(
            action=torch.tensor([pass_action], dtype=torch.int64))
        time_step = env.step(
            action=torch.tensor([height * width - 2], dtype=torch.int64))
        # test game over because of game is too long
        self.assertEqual(time_step.step_type[0], StepType.LAST)
        self.assertEqual(time_step.reward[0], -1.)

        time_step = env.step(action=torch.tensor([0], dtype=torch.int64))
        self.assertEqual(time_step.step_type[0], StepType.FIRST)
        self.assertEqual(time_step.reward[0], 0.)
        time_step = env.step(action=torch.tensor([0], dtype=torch.int64))
        self.assertEqual(time_step.step_type[0], StepType.MID)
        self.assertEqual(time_step.reward[0], 0.)
        time_step = env.step(
            action=torch.tensor([pass_action], dtype=torch.int64))
        self.assertEqual(time_step.step_type[0], StepType.MID)
        self.assertEqual(time_step.reward[0], 0.)
        time_step = env.step(
            action=torch.tensor([pass_action], dtype=torch.int64))
        self.assertEqual(time_step.step_type[0], StepType.LAST)
        self.assertEqual(time_step.reward[0], 1.)

        time_step = env.step(action=torch.tensor([0], dtype=torch.int64))
        self.assertEqual(time_step.step_type[0], StepType.FIRST)
        self.assertEqual(time_step.reward[0], 0.)
        time_step = env.step(action=torch.tensor([0], dtype=torch.int64))
        self.assertEqual(time_step.step_type[0], StepType.MID)
        self.assertEqual(time_step.reward[0], 0.)
        time_step = env.step(
            action=torch.tensor([pass_action], dtype=torch.int64))
        self.assertEqual(time_step.step_type[0], StepType.MID)
        self.assertEqual(time_step.reward[0], 0.)
        time_step = env.step(
            action=torch.tensor([pass_action], dtype=torch.int64))
        self.assertEqual(time_step.step_type[0], StepType.LAST)
        self.assertEqual(time_step.reward[0], 1.)


if __name__ == '__main__':
    alf.test.main()
