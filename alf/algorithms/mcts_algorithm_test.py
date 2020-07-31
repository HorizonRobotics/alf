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
import torch

import alf
from alf.algorithms.mcts_algorithm import MCTSModel, MCTSState, ModelOutput, create_board_game_mcts
from alf.data_structures import StepType, TimeStep


class TicTacToeModel(MCTSModel):
    """A Simple 3x3 board game.

    For two players, X and O, who take turns marking the spaces in a 3×3 grid.
    The player who succeeds in placing three of their marks in a horizontal,
    vertical, or diagonal row is the winner.
    """

    def __init__(self):
        super().__init__()
        self._line_x = torch.tensor(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],
             [0, 1, 2], [0, 1, 2]]).unsqueeze(0)
        self._line_y = torch.tensor(
            [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 0, 0], [1, 1, 1], [2, 2, 2],
             [0, 1, 2], [2, 1, 0]]).unsqueeze(0)
        self._actions = torch.arange(9, dtype=torch.int64).unsqueeze(0)

    def initial_inference(self, observation):
        batch_size = observation.shape[0]
        board = observation
        player = -self._get_current_player(board)
        won = self._check_player_win(board, player)
        reward = torch.where(won, -player.to(torch.float32), torch.tensor(0.))
        game_over = self._check_game_over(board)
        prob = self._get_action_probs(board)
        value = torch.zeros((batch_size, ))

        return ModelOutput(
            value=value,
            reward=reward,
            state=observation,
            actions=self._actions.expand(batch_size, -1),
            action_probs=prob,
            game_over=game_over)

    def recurrent_inference(self, state, action):
        batch_size = state.shape[0]
        B = torch.arange(batch_size)
        x = action % 3
        y = action // 3
        board = state.clone()
        game_over = self._check_game_over(board)
        assert not torch.any(game_over)
        player = self._get_current_player(board).to(torch.float32)
        valid = board[B, y, x] == 0
        board[B[valid], y[valid], x[valid]] = player[valid]
        won = self._check_player_win(board, player)
        reward = torch.where(won, -player, torch.tensor(0.))
        reward = torch.where(valid, reward, player)
        game_over = self._check_game_over(board)
        game_over = torch.max(game_over, ~valid)
        prob = self._get_action_probs(board)
        value = torch.zeros((batch_size, ))
        return ModelOutput(
            value=value,
            reward=reward,
            state=board,
            actions=self._actions.expand(batch_size, -1),
            action_probs=prob,
            game_over=game_over)

    def _check_player_win(self, board, player):
        B = torch.arange(board.shape[0]).unsqueeze(-1).unsqueeze(-1)
        player = player.unsqueeze(-1).unsqueeze(-1)
        lines = board[B, self._line_y, self._line_x]
        return ((lines == player).sum(dim=2) == 3).any(dim=1)

    def _check_game_over(self, board):
        board_full = (board == 0).sum(dim=(1, 2)) == 0
        B = torch.arange(board.shape[0]).unsqueeze(-1).unsqueeze(-1)
        lines = board[B, self._line_y, self._line_x]
        player1_won = ((lines == -1).sum(dim=2) == 3).any(dim=1)
        player2_won = ((lines == 1).sum(dim=2) == 3).any(dim=1)
        return torch.max(board_full, torch.max(player1_won, player2_won))

    def _get_current_player(self, board):
        return ((board != 0).sum(dim=(1, 2)) % 2) * 2 - 1

    def _get_action_probs(self, board):
        p = (board.reshape(board.shape[0], -1) == 0).to(torch.float)
        return p / (p.sum(dim=1, keepdims=True) + 1e-30)


class TicTacToeModelTest(alf.test.TestCase):
    def test_tic_tac_toe(self):
        model = TicTacToeModel()
        observation = torch.tensor([[[ 1,  0, -1.],
                                     [-1,  1,  1],
                                     [-1, -1,  1.]]]) # yapf: disable
        model_output = model.initial_inference(observation)
        self.assertEqual(model_output.reward, torch.tensor([[-1.]]))
        self.assertEqual(model_output.game_over, torch.tensor([[True]]))

        observation = torch.tensor([[[-1,  1,  1.],
                                     [-1, -1,  1],
                                     [-1, -1,  1.]]]) # yapf: disable
        model_output = model.initial_inference(observation)
        self.assertEqual(model_output.game_over, torch.tensor([[True]]))

        self.assertEqual(model_output.reward, torch.tensor([[1.]]))
        observation = torch.tensor([[[0.,  0.,  1.],
                                     [0.,  1., -1.],
                                     [1., -1., -1.]]]) # yapf: disable
        model_output = model.initial_inference(observation)
        self.assertEqual(model_output.reward, torch.tensor([[-1.]]))
        self.assertEqual(model_output.game_over, torch.tensor([[True]]))
        # calling recurrent_inference on ended game causes exception
        self.assertRaises(AssertionError, model.recurrent_inference,
                          observation, torch.tensor([3]))

        observation = torch.tensor([[[0.,  0.,  1.],
                                     [0.,  1., -1.],
                                     [0., -1., -1.]]]) # yapf: disable
        model_output = model.recurrent_inference(observation,
                                                 torch.tensor([6]))
        self.assertEqual(model_output.reward, torch.tensor([[-1.]]))
        self.assertEqual(model_output.game_over, torch.tensor([[True]]))

        # not a valid move for player -1
        model_output = model.recurrent_inference(observation,
                                                 torch.tensor([5]))
        self.assertEqual(model_output.reward, torch.tensor([[1.]]))
        self.assertEqual(model_output.game_over, torch.tensor([[True]]))

        model_output = model.recurrent_inference(observation,
                                                 torch.tensor([3]))
        self.assertEqual(model_output.reward, torch.tensor([[0.]]))
        self.assertEqual(model_output.game_over, torch.tensor([[False]]))


class MCTSAlgorithmTest(alf.test.TestCase):
    def test_mcts_algorithm(self):
        observation_spec = alf.TensorSpec((3, 3))
        action_spec = alf.BoundedTensorSpec((),
                                            dtype=torch.int64,
                                            minimum=0,
                                            maximum=8)
        mcts = create_board_game_mcts(
            observation_spec,
            action_spec,
            dirichlet_alpha=100.,
            num_simulations=2000)
        model = TicTacToeModel()
        time_step = TimeStep(step_type=torch.tensor([StepType.MID]))

        # board situations and expected actions
        # yapf: disable
        cases = [
            ([[1, -1,  1],
              [1, -1, -1],
              [0,  0,  1]], 6),
            ([[0,  0,  0],
              [0, -1, -1],
              [0,  1,  0]], 3),
            ([[ 1, -1, -1],
              [-1, -1,  0],
              [ 0,  1,  1]], 6),
            ([[-1,  0,  1],
              [ 0, -1, -1],
              [ 0,  0,  1]], 3),
            ([[0, 0,  0],
              [0, 0,  0],
              [0, 0, -1]], 4),
            ([[0,  0, 0],
              [0, -1, 0],
              [0,  0, 0]], (0, 2, 6, 8)),
            ([[0,  0,  0],
              [0,  1, -1],
              [1, -1, -1]], 2),
        ]
        # yapf: enable

        # test case serially
        for observation, action in cases:
            observation = torch.tensor([observation], dtype=torch.float32)
            state = MCTSState(steps=(observation != 0).sum(dim=(1, 2)))
            # We use varing num_simulations instead of a fixed large number such
            # as 2000 to make the test faster.
            num_simulations = int((observation == 0).sum().cpu()) * 200
            mcts = create_board_game_mcts(
                observation_spec,
                action_spec,
                dirichlet_alpha=100.,
                num_simulations=num_simulations)
            mcts.set_model(model)
            alg_step = mcts.predict_step(
                time_step._replace(observation=observation), state)
            print(observation, alg_step.output, alg_step.info)
            if type(action) == tuple:
                self.assertTrue(alg_step.output[0] in action)
            else:
                self.assertEqual(alg_step.output[0], action)

        # test batch predict
        observation = torch.tensor([case[0] for case in cases],
                                   dtype=torch.float32)
        state = MCTSState(steps=(observation != 0).sum(dim=(1, 2)))
        mcts = create_board_game_mcts(
            observation_spec,
            action_spec,
            dirichlet_alpha=100.,
            num_simulations=2000)
        mcts.set_model(model)
        alg_step = mcts.predict_step(
            time_step._replace(
                step_type=torch.tensor([StepType.MID] * len(cases)),
                observation=observation), state)
        for i, (observation, action) in enumerate(cases):
            if type(action) == tuple:
                self.assertTrue(alg_step.output[i] in action)
            else:
                self.assertEqual(alg_step.output[i], action)


if __name__ == '__main__':
    alf.test.main()
