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
"""GoEnvironment."""

from collections import OrderedDict
import gin
import torch

import alf
from alf.data_structures import TimeStep, StepType

from .alf_environment import AlfEnvironment

logical_or = torch.max
logical_and = torch.min


class GoBoard(object):
    """This implements Go board.

    This class only takes care how the board changes when a valid move is given.
    Other go rules are handled by GoEnvironment

    We maintain the following data and incrementally update them:

    1. _board: the current board of shape [B, H, W]. At each position, 0 means
        it is empty, -1 means a stone of player 0, 1 means a stone of player 1.
        The board is padded with 2 on four sides to make the handling of boundary
        simpler.
    2. _cc_id: the connected component (CC) which each position belongs to. The
        shape is [B, H, W].
    3. _cc_qi: the qi (liberty) of each CC. The shape is [B, max_num_ccs].
    4. _num_ccs: the number of CCs.

    Note that the qi is different from the common definition of qi. For example.
    in the following board, the qi of the connected component "o" is 4 in our data
    structure because position (1, 1) is counted adjacent to (1, 0) and (0, 1) and
    is counted twice towards the qi of "o". While using the common definition of
    qi, the liberty of "o" is 3. We use this different way of calculating qi
    so that code can be simplified.

    .. code-block:: text

          0123
         ------
        0|oo  |
        1|o   |
        2|    |
        3|    |
         ------

    """

    def __init__(self, batch_size, height, width):
        """
        Args:
            batch_size (int): the number of parallel boards
            height (int): height of each board
            width (int): width of each board
        """
        self._B = torch.arange(batch_size)
        self._board = torch.full((batch_size, height + 2, width + 2),
                                 2,
                                 dtype=torch.int8)
        self._y = torch.arange(1, height + 1).unsqueeze(0).unsqueeze(-1)
        self._x = torch.arange(1, width + 1).unsqueeze(0).unsqueeze(0)
        self._board[self._B.unsqueeze(-1).unsqueeze(-1), self._y, self._x] = 0

        # connected component which this location belongs to, 0 for empty
        self._cc_id = torch.zeros((batch_size, height + 2, width + 2),
                                  dtype=torch.int64)
        self._max_num_ccs = 2 * height * width
        # the qi (liberty) of each connected component.
        self._cc_qi = torch.zeros((batch_size, self._max_num_ccs),
                                  dtype=torch.int32).contiguous()

        # The number of connected components.
        # CC 0 is reserved for the CC of empty space. So num_ccs start from 1.
        self._num_ccs = torch.ones((batch_size, ), dtype=torch.int64)

        # The coordinate delta of neighbors.
        # It can be interesting to have a different dydxs. For example:
        # [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]
        self._dydxs = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    def update(self, board_indices, y, x, player):
        """Update the board for given move at (y, x).

        It assumes the move is at an empty location.

        Args:
            board_indices (Tensor): int64 Tensor to indicate which boards to update.
            y (Tensor): int64 Tensor of the same shape as ``board_indices``
                to indicate the y coordinate of the move
            x (Tensor): int64 Tensor of the same shape as ``board_indices``
                to indicate the x coordinate of the move
            player (Tensor): int8 Tensor of the same shape as ``board_indices``
                to indicate which player make the move
        Returns:
            Tensor: bool Tensor with the same size as ``board_indices``. It indicates
                whether the move for each board is invalid. Invalid move is caused
                by suicidal move (i.e., making the qi of the player 0). Note that
                invalid move may change the board because all the stones of the player
                which are connected to the invalid move will be removed.
        """
        B = board_indices
        y = y + 1
        x = x + 1
        opponent = -player

        # 1. update the qi of opponent, remove if dead
        neighbors = [(y + dy, x + dx) for dy, dx in self._dydxs]
        for ny, nx in neighbors:
            cc_id = self._cc_id[B, ny, nx]
            is_opponent = self._board[B, ny, nx] == opponent
            opponent_B = B[is_opponent]
            opponent_cc_id = cc_id[is_opponent]
            self._cc_qi[opponent_B, opponent_cc_id] -= 1
            self._remove_cc_if_dead(opponent_B, opponent_cc_id)

        # 2. update the qi of self and merge all the neighboring CC into a new CC

        # This is going to be the qi of the newly merged CC
        qi = torch.zeros_like(board_indices, dtype=torch.int32)
        new_cc_id = self._get_new_cc_id(board_indices)

        # Merge each of the 4 neighbors one by one
        for ny, nx in neighbors:
            cc_id = self._cc_id[B, ny, nx]
            # Only merge it if it is the same player
            same_player = self._board[B, ny, nx] == player
            not_merged = logical_and(same_player, cc_id != new_cc_id)
            # Only add the qi of the neighbor CC if it is not merged yet
            qi = torch.where(not_merged, qi + self._cc_qi[B, cc_id], qi)
            # The new stone at (y, x) will decrease the qi by 1
            qi[same_player] -= 1
            # merge it with new_cc if it is not merged yet
            self._change_cc_id(B[not_merged], cc_id[not_merged],
                               new_cc_id[not_merged])
            empty_neighbor = self._board[B, ny, nx] == 0
            # if the neighbor is empty, we increase qi by 1.
            qi[empty_neighbor] += 1

        self._cc_qi[B, new_cc_id] = qi
        self._board[B, y, x] = player
        self._cc_id[B, y, x] = new_cc_id

        invalid = qi == 0
        if invalid.any():
            # restore the board for invalid moves
            self._remove_cc_if_dead(B[invalid], new_cc_id[invalid])

        return invalid

    def _remove_cc_if_dead(self, board_indices, cc_id):
        B = board_indices
        dead = torch.where(self._cc_qi[B, cc_id] == 0)[0]
        if dead.numel() == 0:
            return
        self._remove_cc(B[dead], cc_id[dead])

    def _remove_cc(self, board_indices, cc_id):
        B = board_indices
        to_be_removed = self._cc_id[B] == cc_id.unsqueeze(-1).unsqueeze(-1)
        to_be_removed = torch.where(to_be_removed)
        B = B[to_be_removed[0]]
        y, x = to_be_removed[1:]

        # set to_be_removed to empty
        self._board[B, y, x] = 0
        self._cc_id[B, y, x] = 0

        # update the qi of the neighbours of to_be_removed
        for dy, dx in self._dydxs:
            cc_id = self._cc_id[B, y + dy, x + dx]
            # "self._cc_qi[B, cc_id] += 1" cannot handle duplicated cc_id,
            # so we have to use scatter_add_
            self._cc_qi.view(-1).scatter_add_(
                dim=0,
                index=B * self._max_num_ccs + cc_id,
                src=torch.ones(1, dtype=torch.int32).expand_as(cc_id))

    def _change_cc_id(self, board_indices, cc_id, new_cc_id):
        B = board_indices
        if B.numel() == 0:
            return
        to_be_changed = self._cc_id[B] == cc_id.unsqueeze(-1).unsqueeze(-1)
        to_be_changed = torch.where(to_be_changed)
        B = B[to_be_changed[0]]
        y, x = to_be_changed[1:]
        self._cc_id[B, y, x] = new_cc_id[to_be_changed[0]]

    def _get_new_cc_id(self, board_indices):
        B = board_indices
        new_cc_id = self._num_ccs[B]
        self._num_ccs[B] += 1
        return new_cc_id

    def calc_area(self, board_indices=None):
        """Calculate the area of each player.

        In order for a position to be considered to be owned by a player, it has
        to be either the player's stone or fully surrounded by the player's stone.
        With this definition of area, players have to play until the board is full
        except the eyes of only one position. This shouldn't change how the game
        is played.

        Args:
            board_indices (Tensor): int64 Tensor to indicate the boards
        Returns:
            tuple (Tensor, Tensor): area for player 0 and player 1
        """
        B = self._B if board_indices is None else board_indices
        y = self._y
        x = self._x
        B = B.unsqueeze(-1).unsqueeze(-1)

        area = []
        for player in [-1, 1]:
            occupied = self._board[B, y, x] == player
            surrounded = self._board[B, y, x] == 0
            for dy, dx in self._dydxs:
                neighbor = self._board[B, y + dy, x + dx]
                surrounded = logical_and(surrounded, neighbor != -player)
                surrounded = logical_and(surrounded, neighbor != 0)
            area.append(logical_or(occupied, surrounded).sum(dim=(1, 2)))

        return area[0], area[1]

    def get_board(self, board_indices=None):
        """Get the current board.

        Args:
            board_indices (Tensor): int64 Tensor to indicate the boards
        Returns:
            Tensor: int8 Tensor of the shape [B, height, width].
        """
        B = self._B if board_indices is None else board_indices
        return self._board[B.unsqueeze(-1).unsqueeze(-1), self._y, self._x]

    def reset_board(self, board_indices=None):
        """Reset the board to initial condition.

        Args:
            board_indices (Tensor): int64 Tensor to indicate the boards
        """
        B = self._B if board_indices is None else board_indices
        self._board[B.unsqueeze(-1).unsqueeze(-1), self._y, self._x] = 0
        self._cc_id[B, ...] = 0
        self._cc_qi[B, ...] = 0
        self._num_ccs[B] = 1


class GoEnvironment(AlfEnvironment):
    """Go environment.

    The game plays until one of the following events happen:

    1. Both player pass. In this case, the area of each player will be calculated
       and the reward is 1 if player 0 win, -1 if player 1 win. When calculating
       the area, in order for a position to be considered to be owned by a player,
       it has to be either the player's stone or fully surrounded by the player's
       stone. With this definition of area, players have to play until all the
       dead stones of the opponent have been taken away and the board is full
       except the eyes of only one position. This shouldn't change how the game
       is played.

    2. An invalid move. The opponent will get reward, which means that if player
       0 make an invalid move, the reward is -1. If player 1 make an invalid move,
       the reward is 1. There are two types of invalid moves:
       a. a move to position which is already occupied.
       b. a move which leads to a board exactly same as the previous board.
    3. The total number of moves exceeds ``max_num_moves``. This is considered a
       draw and reward 0 is given. ``max_num_moves`` is set to ``2 * height * width``.

    The observation is an ``OrderedDict`` containing two fields:

    1. board: a [batch_size, 1, height, width] int8 Tensor, with 0 indicating empty
       location, -1 indicating a stone of player 0 and 1 indicating a stone of
       player 1
    2. to_play: a [batch_size] int8 Tensor indicating who is going to make the
       next move. Its value is either 0 or 1

    The action is an int64 scalar. If it is smaller than ``height*width``, it means
    to play the stone at (action // width, action % width). If it is equal to
    ``height * width``, it means to pass for this round.
    """

    def __init__(self, batch_size, height, width, winning_thresh):
        """
        Args:
            batch_size (int): the number of parallel boards
            height (int): height of each board
            width (int): width of each board
            winning_thresh (float): player 0 wins if its area is so much more
                than the area of player 1.
        """
        self._batch_size = batch_size
        self._width = width
        self._height = height
        self._observation_spec = OrderedDict(
            board=alf.TensorSpec((1, height, width), torch.int8),
            to_play=alf.TensorSpec((), torch.int8))
        self._max_num_moves = 2 * height * width
        self._winning_thresh = winning_thresh

        # width*height for pass
        # otherwise it is a move at (y=action // width, x=action % width)
        self._action_spec = alf.BoundedTensorSpec((),
                                                  minimum=0,
                                                  maximum=width * height,
                                                  dtype=torch.int64)
        self._B = torch.arange(self._batch_size)
        self._env_ids = torch.arange(batch_size)
        self._pass_action = width * height
        self._board = GoBoard(batch_size, height, width)
        self._previous_board = self._board.get_board()
        self._num_moves = torch.zeros((batch_size, ), dtype=torch.int32)
        self._game_over = torch.zeros((batch_size, ), dtype=torch.bool)
        self._prev_action = torch.full((batch_size, ),
                                       self._pass_action,
                                       dtype=torch.int64)

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
        self._num_moves.fill_(0)
        self._board.reset_board(self._B)
        self._previous_board = self._board.get_board()
        self._game_over.fill_(False)
        self._prev_action.fill_(self._pass_action)

        return TimeStep(
            observation=OrderedDict(
                board=self._board.get_board().detach().unsqueeze(1),
                to_play=torch.zeros((self._batch_size), dtype=torch.int8)),
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
        current_board = self._board.get_board()
        player = ((self._num_moves % 2) * 2 - 1).to(torch.int8)
        prev_game_over = self._game_over
        step_type = torch.full((self._batch_size, ), int(StepType.MID))
        x = action % self._width
        y = (action // self._width).clamp(max=self._height - 1)

        is_pass = action == self._pass_action
        valid = logical_or(current_board[self._B, y, x] == 0, is_pass)
        placing = logical_and(valid, ~is_pass)

        invalid = self._board.update(self._B[placing], y[placing], x[placing],
                                     player[placing])
        valid[self._B[placing][invalid]] = False

        both_pass = logical_and(self._prev_action == self._pass_action,
                                action == self._pass_action)
        duplicated = (self._board.get_board() == self._previous_board).reshape(
            self._batch_size, -1).all(dim=1)
        illegal_duplicated = logical_and(duplicated, ~both_pass)
        valid = logical_and(valid, ~illegal_duplicated)
        too_long = self._num_moves > self._max_num_moves
        game_over = logical_or(logical_or(both_pass, too_long), ~valid)
        game_over[prev_game_over] = False

        # calc reward
        reward = torch.zeros(self._batch_size)
        areas = self._board.calc_area(self._B[both_pass])
        win = areas[0] - areas[1] > self._winning_thresh
        reward[both_pass] = win.to(torch.float32) * 2 - 1.
        reward = torch.where(valid, reward, player.to(torch.float32))
        reward[prev_game_over] = 0.

        self._num_moves += 1
        self._board.reset_board(self._B[prev_game_over])
        self._num_moves[prev_game_over] = 0

        step_type[game_over] = int(StepType.LAST)
        step_type[prev_game_over] = int(StepType.FIRST)
        discount = torch.ones(self._batch_size)
        discount[game_over] = 0.
        self._game_over = game_over
        self._prev_action = action.detach().clone()
        self._prev_action[prev_game_over] = 0
        player0_win = reward == 1.0
        player1_win = reward == -1.0
        draw = logical_and(game_over, reward == 0)
        self._previous_board = current_board

        return TimeStep(
            observation=OrderedDict(
                board=self._board.get_board().detach().unsqueeze(1),
                to_play=(self._num_moves % 2).to(torch.int8)),
            reward=reward.detach(),
            step_type=step_type.detach(),
            discount=discount.detach(),
            prev_action=self._prev_action,
            env_id=self._env_ids,
            env_info={
                "play0_win": player0_win.to(torch.float32),
                "play1_win": player1_win.to(torch.float32),
                "draw": draw.to(torch.float32),
                "invalid_move": (~valid).to(torch.float32),
            })

    def render(self, mode):
        """Render the game.

        Args:
            mode (str): only 'human' is supported.
        """
        if mode == 'human':
            action = self._prev_action[0].cpu().numpy()
            ay = action // self._width
            ax = action % self._height
            board = self._board.get_board()[0].cpu().numpy()
            img = '-' * (self._width + 2)
            img += '\n'
            for y in range(self._height):
                img += '|'
                for x in range(self._width):
                    if board[y, x] == 0:
                        img += ' '
                    elif board[y, x] == -1:
                        img += 'x'
                    elif board[y, x] == 1:
                        img += 'o'
                    if x == ax and y == ay:
                        img = img[:-1] + img[-1].upper()
                img += '|\n'
            img += '-' * (self._width + 2)
            img += '\n'
            print(img)
        else:
            raise ValueError("Unsupported render mode %s" % mode)


@gin.configurable(whitelist=['height', 'width', 'winning_thresh'])
def load(name='', batch_size=1, height=19, width=19, winning_thresh=7.5):
    """Load GoEnvironment.

    Args:
        name (str): not used
        Args:
            batch_size (int): the number of parallel boards
            height (int): height of each board
            width (int): width of each board
            winning_thresh (float): player 0 wins if its area is so much more
                than the area of player 1.
    Returns:
        GoEnvironment
    """
    return GoEnvironment(
        batch_size, height, width, winning_thresh=winning_thresh)


# environments.utils.create_environment() check this flag to see if load()
# has direct support for batched environment or not.
load.batched = True
