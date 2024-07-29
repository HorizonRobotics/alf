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
"""GoEnvironment."""

from absl import logging
from collections import OrderedDict
import numpy as np
import sys
import time
import torch

import alf
from alf.data_structures import TimeStep, StepType
from alf.nest.utils import convert_device

from .alf_environment import AlfEnvironment


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

    def __init__(self,
                 batch_size,
                 height,
                 width,
                 max_num_moves,
                 num_previous_boards=10):
        """
        Args:
            batch_size (int): the number of parallel boards
            height (int): height of each board
            width (int): width of each board
            max_num_moves (int): maximum number of moves allowed
            num_previous_boards (int): previous so many board situation will be
                stored. They will be used by ``classify_all_moves()`` to check
                whether a move will lead to board situation same as one of these
                previous board situations.
        """
        self._width = width
        self._height = height
        self._num_previous_boards = num_previous_boards

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
        # The additional 2 are for cc_id 0 and 1 more move from GoEnvironment
        self._max_num_ccs = max_num_moves + 2
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

        self._prev_boards = torch.full(
            (batch_size, num_previous_boards, height + 2, width + 2),
            2,
            dtype=torch.int8)
        self._prev_boards[self._B.reshape(-1, 1, 1, 1),
                          torch.arange(num_previous_boards).
                          reshape(1, -1, 1, 1),
                          self._y.unsqueeze(0),
                          self._x.unsqueeze(0)] = 0
        self._num_moves = torch.zeros(batch_size, dtype=torch.int64)

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
                whether the move for each board is suicidal (i.e., making the qi
                of the player 0). Note that suicidal move may change the board
                because all the stones of the player which are connected to the
                suicidal move will be removed.
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
        qi = torch.zeros_like(B, dtype=torch.int32)
        new_cc_id = self._get_new_cc_id(B)

        # Merge each of the 4 neighbors one by one
        for ny, nx in neighbors:
            cc_id = self._cc_id[B, ny, nx]
            # Only merge it if it is the same player
            same_player = self._board[B, ny, nx] == player
            not_merged = same_player & (cc_id != new_cc_id)
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

        suicidal = qi == 0
        if suicidal.any():
            # remove the connected stones for suicidal move
            self._remove_cc(B[suicidal], new_cc_id[suicidal])

        prev_idx = self._num_moves[B].reshape(1,
                                              -1) % self._num_previous_boards
        self._prev_boards[B, prev_idx] = self._board[B]
        self._num_moves[B] += 1

        return suicidal

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

    def calc_area_simple(self, board_indices=None):
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
                surrounded = surrounded & (neighbor != -player)
                surrounded = surrounded & (neighbor != 0)
            area.append((occupied | surrounded).sum(dim=(1, 2)))

        return area[0], area[1]

    def calc_area(self, board_indices=None):
        """Calculate the area of each player.

        In order for a position to be considered to be owned by a player, it has
        to be either the player's stone or cannot be reached by the opponent's stones.
        With this definition of area, players have to play until all the dead stones
        have been taken out. This shouldn't change how the game is played. This
        is the so called `Tromp-Taylor rules <https://en.wikibooks.org/wiki/Computer_Go/Tromp-Taylor_Rules>`_

        Args:
            board_indices (Tensor): int64 Tensor to indicate the boards
        Returns:
            tuple (Tensor, Tensor): area for player 0 and player 1
        """
        B = self._B if board_indices is None else board_indices
        y = self._y
        x = self._x

        BB = torch.arange(B.shape[0]).unsqueeze(-1).unsqueeze(-1)
        expanded_boards = []
        nynxs = [(y + dy, x + dx) for dy, dx in self._dydxs]
        for player in [-1, 1]:
            # After the loop finish, expanded[b, x, y] == player iff (x, y)
            # can be reached by one of stones of player in board b. This is achieved
            # step-by-step by checking whether the neighbor of a location can
            # be reached. The loop ends until no more new location can be connected.
            expanded = self._board[B]
            area = (expanded == player).sum(dim=(1, 2))
            while True:
                connected = None
                for ny, nx in nynxs:
                    neighbor = expanded[BB, ny, nx]
                    if connected is None:
                        connected = neighbor == player
                    else:
                        connected = connected | (neighbor == player)
                connected = connected & (expanded[BB, y, x] == 0)
                new_area = area + connected.sum(dim=(1, 2))
                if (area == new_area).all():
                    break
                cb, cy, cx = torch.where(connected)
                expanded[cb, cy + 1, cx + 1] = player
                area = new_area

            expanded_boards.append(expanded)

        # area includes positions that can be reached by self, but not by opponent
        area0 = (expanded_boards[0] == -1) & (expanded_boards[1] != 1)
        area1 = (expanded_boards[0] != -1) & (expanded_boards[1] == 1)

        return area0.sum(dim=(1, 2)), area1.sum(dim=(1, 2))

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
        self._board[B.reshape(-1, 1, 1), self._y, self._x] = 0
        self._cc_id[B, ...] = 0
        self._cc_qi[B, :] = 0
        self._num_ccs[B] = 1
        self._num_moves[B] = 0
        self._prev_boards[B.reshape(-1, 1, 1, 1),
                          torch.arange(self._num_previous_boards).
                          reshape(1, -1, 1, 1),
                          self._y.unsqueeze(0),
                          self._x.unsqueeze(0)] = 0

    def classify_all_moves(self, player, board_indices=None):
        """Classify all the moves on the board.

        This function will examine all possible moves except PASS and annotate
        them using 3 boolean attributes: occupied, suicidal, and repeated.

        Args:
            player (Tensor): int8 Tensor to indicate which player to consider.
            board_indices (Tensor): int64 Tensor to indicate the boards
        Returns:
            tuple: each one is a bool Tensor of shape [B, height, width].
            - occupied: occupied[b, y, x] means whether a move at (y, x) overlapped
                with existing stone on the board[b]
            - suicidal: suicidal[b, y, x] means whether a move at (y, x) is a
                suicidal move for player[b] on board[b]
            - repeated: repeated[b, y, x] means whether a move at (y, x) by player[b]
                will result in a board same as one of the previous boards of board[b].
        """
        if board_indices is None:
            board_indices = self._B
        prev_boards = self._prev_boards[board_indices]

        # We replicate each board to ``height*width`` boards so that we can try
        # each of ``height*width`` moves independently without interference. The
        # process is very similar to ``update()``.

        # [B * H * W]
        x = torch.arange(1, self._width + 1).repeat(
            board_indices.shape[0] * self._height)
        # [B * H * W]
        y = torch.arange(1, self._height + 1).repeat_interleave(
            self._width).repeat(board_indices.shape[0])
        board_indices = board_indices.repeat_interleave(
            self._height * self._width)
        player = player.repeat_interleave(self._height * self._width)
        opponent = -player

        # [B * H * W, height + 2, width + 2]
        boards = self._board[board_indices]
        # [B * H * W, height + 2, width + 2]
        cc_ids = self._cc_id[board_indices]
        # [B * H * W, max_num_ccs]
        cc_qis = self._cc_qi[board_indices]

        B = torch.arange(board_indices.shape[0])

        empty = boards[B, y, x] == 0

        def _remove_cc(board_indices, cc_id):
            """remove stones belongs to cc_id."""
            B = board_indices
            to_be_removed = cc_ids[B] == cc_id.unsqueeze(-1).unsqueeze(-1)
            to_be_removed = torch.where(to_be_removed)
            B = B[to_be_removed[0]]
            y, x = to_be_removed[1:]
            boards[B, y, x] = 0

        def _change_cc_id(board_indices, cc_id, new_cc_id):
            """Change cc_id to new_cc_id."""
            B = board_indices
            to_be_changed = cc_ids[B] == cc_id.unsqueeze(-1).unsqueeze(-1)
            to_be_changed = torch.where(to_be_changed)
            B = B[to_be_changed[0]]
            y, x = to_be_changed[1:]
            cc_ids[B, y, x] = new_cc_id[to_be_changed[0]]

        # 1. update the qi of opponent, remove if dead
        neighbors = [(y + dy, x + dx) for dy, dx in self._dydxs]
        for ny, nx in neighbors:
            cc_id = cc_ids[B, ny, nx]
            is_opponent = (boards[B, ny, nx] == opponent) & empty
            cc_qis[B[is_opponent], cc_id[is_opponent]] -= 1
            dead = (cc_qis[B, cc_id] == 0) & is_opponent
            _remove_cc(B[dead], cc_id[dead])

        # 2. update the qi of self and merge all the neighboring CC into a new CC

        # This is going to be the qi of the newly merged CC
        qi = torch.zeros_like(B, dtype=torch.int32)
        new_cc_id = self._num_ccs[board_indices]

        # Merge each of the 4 neighbors one by one
        for ny, nx in neighbors:
            cc_id = cc_ids[B, ny, nx]
            # Only merge it if it is the same player
            same_player = (boards[B, ny, nx] == player) & empty
            not_merged = same_player & (cc_id != new_cc_id)
            # Only add the qi of the neighbor CC if it is not merged yet
            qi = torch.where(not_merged, qi + cc_qis[B, cc_id], qi)
            # The new stone at (y, x) will decrease the qi by 1
            qi[same_player] -= 1

            # merge it with new_cc if it is not merged yet
            _change_cc_id(B[not_merged], cc_id[not_merged],
                          new_cc_id[not_merged])

            empty_neighbor = boards[B, ny, nx] == 0
            # if the neighbor is empty, we increase qi by 1.
            qi[empty_neighbor] += 1

        empty_move = B[empty], y[empty], x[empty]
        boards[empty_move] = player[empty]
        cc_ids[empty_move] = new_cc_id[empty]

        suicidal = (qi == 0) & empty
        # remove the connected stones for suicidal move
        _remove_cc(B[suicidal], new_cc_id[suicidal])

        boards = boards.reshape(-1, 1, self._height, self._width,
                                self._height + 2, self._width + 2)
        prev_boards = prev_boards.reshape(-1, self._num_previous_boards, 1, 1,
                                          self._height + 2, self._width + 2)
        repeated = boards == prev_boards
        repeated = repeated.reshape(*repeated.shape[:-2],
                                    -1).all(dim=-1).any(dim=1)
        suicidal = suicidal.reshape(-1, self._height, self._width)
        occupied = ~empty.reshape(-1, self._height, self._width)
        repeated = repeated & ~occupied

        return occupied, suicidal, repeated


@alf.configurable(blacklist=['batch_size'])
class GoEnvironment(AlfEnvironment):
    """Go environment.

    The game plays until one of the following events happen:

    1. Both player pass. In this case, the area of each player will be calculated
      and the reward is 1 if player 0 win, -1 if player 1 win. When calculating the
      area, in order for a position to be considered to be owned by a player, it has
      to be either the player's stone or cannot be reached by the opponent's stones.
      With this definition of area, players have to play until all the dead stones
      have been taken out. This shouldn't change how the game is played. This
      is the so called `Tromp-Taylor rules <https://en.wikibooks.org/wiki/Computer_Go/Tromp-Taylor_Rules>`_
    2. An invalid move. The opponent will get reward, which means that if player
       0 make an invalid move, the reward is -1. If player 1 make an invalid move,
       the reward is 1. There are two types of invalid moves:
       a. a move to position which is already occupied.
       b. a move which leads to a board exactly same as the previous board.
    3. The total number of moves exceeds ``max_num_moves``. This is considered as
       both passing. ``max_num_moves`` is set to ``2 * height * width``.

    The observation is an ``OrderedDict`` containing three fields:

    1. board: a [batch_size, 1, height, width] int8 Tensor, with 0 indicating empty
       location, -1 indicating a stone of player 0 and 1 indicating a stone of
       player 1
    2. to_play: a [batch_size] int8 Tensor indicating who is going to make the
       next move. Its value is either 0 or 1
    3. prev_action: a [batch_size] int64 Tensor indicating the action taken by
       the previous player. This is pass action for the first step.

    The action is an int64 scalar. If it is smaller than ``height*width``, it means
    to play the stone at (action // width, action % width). If it is equal to
    ``height * width``, it means to pass for this round.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self,
                 batch_size,
                 height=19,
                 width=19,
                 winning_thresh=7.5,
                 allow_suicidal_move=False,
                 reward_shaping=False,
                 human_player=None):
        """
        Args:
            batch_size (int): the number of parallel boards
            height (int): height of each board
            width (int): width of each board
            winning_thresh (float): player 0 wins if area0 - area1 > winning_thresh,
                lose if area0 - area1 < winning_thresh, otherwise draw.
            allow_suicidal_move (bool): whether suicidal move is allowed.
            reward_shaping (bool): if True, instead of using +1,-1 as reward,
                use ``alf.math.softsign(area0 - area1 - winning_thresh)`` as reward
                to encourage capture more area.
            human_player (int|None): 0, 1 or None
        """
        self._batch_size = batch_size
        self._width = width
        self._height = height
        self._max_num_moves = 2 * height * width
        self._winning_thresh = float(winning_thresh)
        self._allow_suicical_move = allow_suicidal_move
        self._reward_shaping = reward_shaping
        self._human_player = human_player

        # width*height for pass
        # otherwise it is a move at (y=action // width, x=action % width)
        self._action_spec = alf.BoundedTensorSpec((),
                                                  minimum=0,
                                                  maximum=width * height,
                                                  dtype=torch.int64)
        self._observation_spec = OrderedDict(
            board=alf.TensorSpec((1, height, width), torch.int8),
            prev_action=self._action_spec,
            valid_action_mask=alf.TensorSpec([width * height + 1], torch.bool),
            steps=alf.TensorSpec((), torch.int32),
            to_play=alf.TensorSpec((), torch.int8))

        self._B = torch.arange(self._batch_size)
        self._env_ids = torch.arange(batch_size)
        self._pass_action = width * height
        self._board = GoBoard(batch_size, height, width, self._max_num_moves)
        self._previous_board = self._board.get_board()
        self._num_moves = torch.zeros((batch_size, ), dtype=torch.int32)
        self._game_over = torch.zeros((batch_size, ), dtype=torch.bool)
        self._prev_action = torch.full((batch_size, ),
                                       self._pass_action,
                                       dtype=torch.int64)
        self._surface = None
        if human_player is not None:
            logging.info("Use mouse click to place a stone")
            logging.info("Kayboard control:")
            logging.info("P     : pass")
            logging.info("SPACE : refresh display")

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
            "player0_win": alf.TensorSpec(()),
            "player1_win": alf.TensorSpec(()),
            "player0_pass": alf.TensorSpec(()),
            "player1_pass": alf.TensorSpec(()),
            "draw": alf.TensorSpec(()),
            "invalid_move": alf.TensorSpec(()),
            "too_long": alf.TensorSpec(()),
            "bad_move": alf.TensorSpec(()),
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
                prev_action=self._prev_action,
                valid_action_mask=self._get_valid_action_mask(),
                steps=self._num_moves,
                to_play=torch.zeros((self._batch_size), dtype=torch.int8)),
            step_type=torch.full((self._batch_size, ),
                                 StepType.FIRST,
                                 dtype=torch.int32),
            reward=torch.zeros((self._batch_size, )),
            discount=torch.ones((self._batch_size, )),
            prev_action=self._prev_action,
            env_id=self._env_ids,
            env_info={
                "player0_win": torch.zeros(self._batch_size),
                "player1_win": torch.zeros(self._batch_size),
                "player0_pass": torch.zeros(self._batch_size),
                "player1_pass": torch.zeros(self._batch_size),
                "draw": torch.zeros(self._batch_size),
                "invalid_move": torch.zeros(self._batch_size),
                "too_long": torch.zeros(self._batch_size),
                "bad_move": torch.zeros(self._batch_size),
            })

    def _get_valid_action_mask(self):
        player = ((self._num_moves % 2) * 2 - 1).to(torch.int8)
        occupied, suicidal, repeated = self._board.classify_all_moves(player)
        invalid = occupied | repeated
        if not self._allow_suicical_move:
            invalid = invalid | suicidal

        valid = (~invalid).reshape(self._batch_size, -1)
        valid = torch.cat(
            [valid, torch.ones(self._batch_size, 1, dtype=torch.bool)], dim=1)
        return valid

    def _step1(self, action):
        """``_step1()`` is used for actually update the board and calculate the
        reward given the action.
        """
        prev_game_over = self._game_over
        self._board.reset_board(self._B[prev_game_over])
        current_board = self._board.get_board()
        player = ((self._num_moves % 2) * 2 - 1).to(torch.int8)
        step_type = torch.full((self._batch_size, ),
                               StepType.MID,
                               dtype=torch.int32)
        height = self._height
        width = self._width

        x = action % width
        y = (action // width).clamp(max=height - 1)

        is_pass = action == self._pass_action
        valid = (current_board[self._B, y, x] == 0) | is_pass
        placing = valid & ~is_pass & ~prev_game_over
        suicidal = self._board.update(self._B[placing], y[placing], x[placing],
                                      player[placing])
        if not self._allow_suicical_move:
            valid[self._B[placing][suicidal]] = False

        both_pass = ((self._prev_action == self._pass_action) &
                     (action == self._pass_action))
        duplicated = (self._board.get_board() == self._previous_board).reshape(
            self._batch_size, -1).all(dim=1) & ~prev_game_over
        illegal_duplicated = duplicated & ~is_pass & ~prev_game_over
        valid = valid & ~illegal_duplicated

        self._num_moves += 1
        too_long = self._num_moves >= self._max_num_moves
        game_over = both_pass | too_long | ~valid
        game_over[prev_game_over] = False

        # calc reward
        reward = torch.zeros(self._batch_size)
        scoring = both_pass | too_long
        areas = self._board.calc_area(self._B[scoring])
        if self._batch_size == 1:
            if scoring:
                print(areas)
        score = areas[0] - areas[1] - self._winning_thresh
        if self._reward_shaping:
            reward[scoring] = alf.math.softsign(score)
        else:
            reward[scoring] = score.sign()
        reward = torch.where(valid, reward, player.to(torch.float32))
        reward[prev_game_over] = 0.

        self._num_moves[prev_game_over] = 0

        # early moves on side are bad
        bad_move = (0 < self._num_moves) & (self._num_moves <= width) & (
            (x == 0) | (x == width - 1) | (y == 0) | (y == height - 1))

        step_type[game_over] = int(StepType.LAST)
        step_type[prev_game_over] = int(StepType.FIRST)
        discount = torch.ones(self._batch_size)
        discount[game_over] = 0.
        self._game_over = game_over
        self._prev_action = action.detach().clone()
        self._prev_action[prev_game_over] = self._pass_action
        player0_win = reward > 0
        player1_win = reward < 0
        draw = game_over & (reward == 0)
        self._previous_board = current_board

        return TimeStep(
            observation=OrderedDict(
                board=self._board.get_board().detach().unsqueeze(1),
                prev_action=self._prev_action,
                valid_action_mask=self._get_valid_action_mask(),
                steps=self._num_moves,
                to_play=(self._num_moves % 2).to(torch.int8)),
            reward=reward.detach(),
            step_type=step_type.detach(),
            discount=discount.detach(),
            prev_action=self._prev_action,
            env_id=self._env_ids,
            env_info={
                "player0_win": player0_win.to(torch.float32),
                "player1_win": player1_win.to(torch.float32),
                "player0_pass": is_pass & (player == -1),
                "player1_pass": is_pass & (player == 1),
                "draw": draw.to(torch.float32),
                "invalid_move": (~valid).to(torch.float32),
                "too_long": too_long.to(torch.float32),
                "bad_move": bad_move.to(torch.float32),
            })

    def _step(self, action):
        """When there is a human player, the human player is part of the environment
        for the computer algorithm. ``_step()`` is the interface for the computer
        to interact with the environment. So ``_step()`` needs to get the action
        from the human player and call the underlying ``_step1()`` to update the
        board and calculate reward.
        """
        if self._human_player is None:
            return self._step1(action)
        if self._num_moves == 0 and self._human_player == 0:
            self.render('human')
            human_action = self._get_human_action()
            time_step = self._step1(human_action)
            self.render('human')
            return time_step
        time_step = self._step1(action)
        self.render('human')
        if time_step.step_type[0] == StepType.LAST:
            return time_step
        human_action = self._get_human_action()
        time_step = self._step1(human_action)
        self.render('human')
        return time_step

    def _get_human_action(self):
        import pygame
        import pygame.locals as K

        valid_action_mask = self._get_valid_action_mask()[0].cpu()
        grid_size = self._grid_size
        offset = self._offset
        stone_radius = self._stone_radius
        action = None
        while action is None:
            time.sleep(0.01)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONUP:
                    mx, my = event.pos
                    x = int((mx - offset) / grid_size + 0.5)
                    y = int((my - offset) / grid_size + 0.5)
                    gx = offset + x * grid_size
                    gy = offset + y * grid_size
                    if ((mx - gx) * (mx - gx) +
                        (my - gy) * (my - gy) <= stone_radius * stone_radius
                            and 0 <= x < self._width and 0 <= y < self._height
                            and valid_action_mask[y * self._width + x]):
                        action = y * self._width + x
                        break
                if event.type == pygame.KEYUP:
                    if event.key == K.K_p:
                        action = self._pass_action
                        break
                    if event.key == K.K_SPACE:
                        # For some unknown reason, the display may not update occasionally.
                        # Press SPACE to update the display.
                        self.render('human')
        return torch.tensor([action], dtype=torch.int64)

    def render(self, mode):
        import pygame
        self._grid_size = 40
        self._offset = self._grid_size // 2
        self._stone_radius = int(0.4 * self._grid_size)
        grid_size = self._grid_size
        offset = self._offset
        stone_radius = self._stone_radius
        if self._surface is None:
            pygame.init()
            if mode == 'human':
                self._surface = pygame.display.set_mode(
                    (self._width * grid_size, self._height * grid_size),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
            else:
                self._surface = pygame.Surface((self._width * grid_size,
                                                self._height * grid_size))

        self._surface.fill((128, 128, 64))
        for y in range(self._height):
            pygame.draw.line(self._surface, (0, 0, 0),
                             (offset, offset + y * grid_size),
                             (offset + (self._width - 1) * grid_size,
                              offset + y * grid_size))

        for x in range(self._width):
            pygame.draw.line(self._surface, (0, 0, 0),
                             (offset + x * grid_size, offset),
                             (offset + x * grid_size,
                              offset + (self._height - 1) * grid_size))

        action = self._prev_action[0].cpu().numpy()
        ay = action // self._width
        ax = action % self._height
        board = self._board.get_board()[0].cpu().numpy()
        for y in range(self._height):
            for x in range(self._width):
                if board[y, x] != 0:
                    c = 128 + 127 * board[y, x]
                    pygame.draw.circle(
                        self._surface, (c, c, c),
                        (offset + x * grid_size, offset + y * grid_size),
                        stone_radius, 0)

        if action != self._pass_action:
            pygame.draw.circle(
                self._surface, (255, 0, 0),
                (offset + ax * grid_size, offset + ay * grid_size), 2, 0)

        if mode == 'human':
            pygame.display.flip()
            time.sleep(0.1)
        elif mode == 'rgb_array':
            # (x, y, c) => (y, x, c)
            return np.transpose(
                pygame.surfarray.array3d(self._surface), (1, 0, 2))
        else:
            raise ValueError("Unsupported render mode: %s" % mode)


@alf.configurable(whitelist=[])
def load(name='', batch_size=1):
    """Load GoEnvironment.

    Args:
        name (str): not used
        Args:
            batch_size (int): the number of parallel boards
    Returns:
        GoEnvironment
    """
    return GoEnvironment(batch_size)


# environments.utils.create_environment() check this flag to see if load()
# has direct support for batched environment or not.
load.batched = True
