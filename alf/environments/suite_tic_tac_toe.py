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
from alf.data_structures import TimeStep, StepType

from .alf_environment import AlfEnvironment


class TicTacToeEnvironment(AlfEnvironment):
    """A Simple 3x3 board game.

    For two players, X and O, who take turns marking the spaces in a 3×3 grid.
    The player who succeeds in placing three of their marks in a horizontal,
    vertical, or diagonal line is the winner.

    The reward is +1 if player 0 win, -1 if player 1 win and 0 for draw.
    An invalid move will give the reward for the opponent.
    """

    def __init__(self, batch_size):
        self._batch_size = batch_size
        self._observation_spec = alf.TensorSpec((3, 3))
        self._action_spec = alf.BoundedTensorSpec((),
                                                  minimum=0,
                                                  maximum=8,
                                                  dtype=torch.int64)
        self._line_x = torch.tensor(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],
             [0, 1, 2], [0, 1, 2]]).unsqueeze(0)
        self._line_y = torch.tensor(
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 0, 0], [1, 1, 1], [2, 2, 2],
             [0, 1, 2], [2, 1, 0]]).unsqueeze(0)
        self._B = torch.arange(self._batch_size)
        self._empty_board = self._observation_spec.zeros()
        self._boards = self._observation_spec.zeros((self._batch_size, ))
        self._env_ids = torch.arange(batch_size)
        self._player_0 = torch.tensor(-1.)
        self._player_1 = torch.tensor(1.)

    @property
    def is_tensor_based(self):
        return True

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._batch_size

    def env_info_spec(self):
        return {
            "play0_win": alf.TensorSpec(()),
            "play1_win": alf.TensorSpec(()),
            "draw": alf.TensorSpec(()),
            "invalid_move": alf.TensorSpec(()),
        }

    def observation_spec(self):
        return self._observation_spec

    def observation_desc(self):
        return ""

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        self._boards = self._observation_spec.zeros((self._batch_size, ))
        self._game_over = torch.zeros((self._batch_size, ), dtype=torch.bool)
        self._prev_action = self._action_spec.zeros((self._batch_size, ))
        return TimeStep(
            observation=self._boards.clone().detach(),
            step_type=torch.full((self._batch_size, ), StepType.FIRST),
            reward=torch.zeros((self._batch_size, )),
            discount=torch.ones((self._batch_size, )),
            prev_action=self._action_spec.zeros((self._batch_size, )),
            env_id=self._env_ids,
            env_info={
                "play0_win": torch.zeros(self._batch_size),
                "play1_win": torch.zeros(self._batch_size),
                "draw": torch.zeros(self._batch_size),
                "invalid_move": torch.zeros(self._batch_size),
            })

    def _step(self, action):
        prev_game_over = self._game_over
        prev_action = action.clone()
        prev_action[prev_game_over] = 0
        self._boards[prev_game_over] = self._empty_board
        step_type = torch.full((self._batch_size, ), int(StepType.MID))
        player = self._get_current_player().to(torch.float32)
        x = action % 3
        y = action // 3
        valid = self._boards[self._B, y, x] == 0
        self._boards[self._B[valid], y[valid], x[valid]] = player[valid]
        won = self._check_player_win(player)
        reward = torch.where(won, -player, torch.tensor(0.))
        reward = torch.where(valid, reward, player)
        game_over = self._check_game_over()
        game_over = torch.max(game_over, ~valid)
        step_type[game_over] = int(StepType.LAST)
        step_type[prev_game_over] = int(StepType.FIRST)
        discount = torch.ones(self._batch_size)
        discount[game_over] = 0.
        self._boards[prev_game_over] = self._empty_board
        self._game_over = game_over
        self._prev_action = action
        player0_win = self._check_player_win(self._player_0)
        player1_win = self._check_player_win(self._player_1)
        draw = torch.min(game_over, reward == 0)

        return TimeStep(
            observation=self._boards.clone().detach(),
            reward=reward.detach(),
            step_type=step_type.detach(),
            discount=discount.detach(),
            prev_action=prev_action.detach(),
            env_id=self._env_ids,
            env_info={
                "play0_win": player0_win.to(torch.float32),
                "play1_win": player1_win.to(torch.float32),
                "draw": draw.to(torch.float32),
                "invalid_move": (~valid).to(torch.float32),
            })

    def _check_player_win(self, player):
        B = self._B.unsqueeze(-1).unsqueeze(-1)
        player = player.unsqueeze(-1).unsqueeze(-1)
        lines = self._boards[B, self._line_y, self._line_x]
        return ((lines == player).sum(dim=2) == 3).any(dim=1)

    def _check_game_over(self):
        board_full = (self._boards == 0).sum(dim=(1, 2)) == 0
        B = self._B.unsqueeze(-1).unsqueeze(-1)
        lines = self._boards[B, self._line_y, self._line_x]
        player0_won = ((lines == self._player_0).sum(dim=2) == 3).any(dim=1)
        player1_won = ((lines == self._player_1).sum(dim=2) == 3).any(dim=1)
        return torch.max(board_full, torch.max(player0_won, player1_won))

    def _get_current_player(self):
        return ((self._boards != 0).sum(dim=(1, 2)) % 2) * 2 - 1

    def render(self, mode):
        if mode == 'human':
            action = self._prev_action[0].cpu().numpy()
            ay = action // 3
            ax = action % 3
            board = self._boards[0].cpu().numpy()
            img = '-----\n'
            for y in range(3):
                img += '|'
                for x in range(3):
                    if board[y, x] == 0:
                        img += ' '
                    elif board[y, x] == -1:
                        img += 'x'
                    elif board[y, x] == 1:
                        img += 'o'
                    if x == ax and y == ay:
                        img = img[:-1] + img[-1].upper()
                img += '|\n'
            img += '-----\n'
            print(img)
        else:
            raise ValueError("Unsupported render mode %s" % mode)


@alf.configurable(whitelist=[])
def load(name='', batch_size=1):
    """Load TicTacToeEnvironment

    Args:
        name (str): not used
        batch_size (int): the number of games in the simulation.
    """
    return TicTacToeEnvironment(batch_size)


# environments.utils.create_environment() check this flag to see if load()
# has direct support for batched environment or not.
load.batched = True
