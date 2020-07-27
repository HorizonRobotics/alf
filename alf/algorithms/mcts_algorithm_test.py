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
import time
import torch
from torch.distributions import Categorical

import alf
from alf.algorithms.mcts_algorithm import MCTSModel, MCTSState, ModelOutput, MCTSAlgorithm, create_board_game_mcts
from alf.data_structures import StepType, TimeStep


class TicTacToeModel(MCTSModel):
    """A Simple 3x3 board game.

    For two players, X and O, who take turns marking the spaces in a 3×3 grid.
    The player who succeeds in placing three of their marks in a horizontal,
    vertical, or diagonal row is the winner.
    """

    def __init__(self):
        self._line_x = torch.tensor([
            0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
            1, 2
        ])
        self._line_y = torch.tensor([
            0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2,
            1, 0
        ])
        self._actions = torch.arange(9, dtype=torch.int32).unsqueeze(0)

    def initial_inference(self, observation):
        batch_size = observation.shape[0]
        reward = torch.zeros((batch_size, ))
        prob = torch.zeros((batch_size, 9))
        game_over = torch.zeros((batch_size, ), dtype=torch.bool)
        for i in range(batch_size):
            board = observation[i]
            player = -self._get_current_player(board)
            reward[i] = -player * self._check_player_win(board, player).to(
                torch.float)
            prob[i] = self._get_action_probs(board)
            game_over[i] = self._check_game_over(board)

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
        x = action % 3
        y = action // 3
        new_state = state.clone()
        reward = torch.zeros((batch_size, ))
        prob = torch.zeros((batch_size, 9))
        game_over = torch.zeros((batch_size, ), dtype=torch.bool)
        for i in range(batch_size):
            board = new_state[i]
            player = self._get_current_player(board)
            if not self._check_game_over(board):
                assert board[y[i], x[i]] == 0
                board[y[i], x[i]] = player
                reward[i] = -player * self._check_player_win(board, player).to(
                    torch.float)
            prob[i] = self._get_action_probs(board)
            game_over[i] = self._check_game_over(board)

        value = torch.zeros((batch_size, ))
        return ModelOutput(
            value=value,
            reward=reward,
            state=new_state,
            actions=self._actions.expand(batch_size, -1),
            action_probs=prob,
            game_over=game_over)

    def _check_player_win(self, board, player):
        lines = board[(self._line_y, self._line_x)].reshape(-1, 3)
        return ((lines == player).sum(dim=1) == 3).any()

    def _check_game_over(self, board):
        return (board == 0).sum() == 0 or self._check_player_win(
            board, -1) or self._check_player_win(board, 1)

    def _get_current_player(self, board):
        return ((board != 0).sum() % 2) * 2 - 1

    def _get_action_probs(self, board):
        p = (board.reshape(-1) == 0).to(torch.float)
        if p.sum() != 0:
            return p / p.sum()
        else:
            return torch.ones((9, )) / 9


class TicTacToeModelTest(alf.test.TestCase):
    def test_tic_tac_toe(self):
        model = TicTacToeModel()
        observation = torch.tensor([[[1, 0, -1.], [-1, 1, 1], [-1, -1, 1.]]])

        model_output = model.initial_inference(observation)
        self.assertEqual(model_output.reward, torch.tensor([[-1.]]))
        self.assertEqual(model_output.game_over, torch.tensor([[True]]))

        observation = torch.tensor([[[-1, 1, 1.], [-1, -1, 1], [-1, -1, 1.]]])
        model_output = model.initial_inference(observation)
        self.assertEqual(model_output.game_over, torch.tensor([[True]]))

        self.assertEqual(model_output.reward, torch.tensor([[1.]]))
        observation = torch.tensor([[[0., 0., 1.], [0., 1., -1.],
                                     [1., -1., -1.]]])
        model_output = model.initial_inference(observation)
        self.assertEqual(model_output.reward, torch.tensor([[-1.]]))
        self.assertEqual(model_output.game_over, torch.tensor([[True]]))


class MCTSAlgorithmTest(alf.test.TestCase):
    def test_mcts_algorithm(self):
        observation_spec = alf.TensorSpec((3, 3))
        action_spec = alf.BoundedTensorSpec((),
                                            dtype=torch.int32,
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
        cases = [
            ([[1, -1, 1], [1, -1, -1], [0, 0, 1]], 6),
            ([[0, 0, 0], [0, -1, -1], [0, 1, 0]], 3),
            ([[1, -1, -1], [-1, -1, 0], [0, 1, 1]], 6),
            ([[-1, 0, 1], [0, -1, -1], [0, 0, 1]], 3),
            ([[0, 0, 0], [0, 0, 0], [0, 0, -1]], 4),
            ([[0, 0, 0], [0, -1, 0], [0, 0, 0]], (0, 2, 6, 8)),
            ([[0, 0, 0], [0, 1, -1], [1, -1, -1]], 2),
        ]

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
