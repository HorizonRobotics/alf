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
"""Monte-Carlo Tree Search."""

import gin
import torch
import torch.distributions as td
import torch.nn.functional as F
from typing import Callable

import alf
from alf import TensorSpec
from alf.data_structures import AlgStep, LossInfo, namedtuple, TimeStep
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.utils import dist_utils
from alf import nest
from alf.nest.utils import convert_device
from alf.trainers.policy_trainer import Trainer
from alf.utils import common, tensor_utils
from .mcts_models import MCTSModel, ModelOutput

MAXIMUM_FLOAT_VALUE = float('inf')


class _MCTSTree(object):
    def __init__(self, num_expansions, model_output, known_value_bounds):
        batch_size, branch_factor = model_output.action_probs.shape
        action_spec = dist_utils.extract_spec(model_output.actions, from_dim=2)
        state_spec = dist_utils.extract_spec(model_output.state, from_dim=1)
        if known_value_bounds:
            self.fixed_bounds = True
            self.minimum, self.maximum = known_value_bounds
        else:
            self.fixed_bounds = False
            self.minimum, self.maximum = MAXIMUM_FLOAT_VALUE, -MAXIMUM_FLOAT_VALUE
        self.minimum = torch.full((batch_size, ),
                                  self.minimum,
                                  dtype=torch.float32)
        self.maximum = torch.full((batch_size, ),
                                  self.maximum,
                                  dtype=torch.float32)
        if known_value_bounds:
            self.normalize_scale = 1 / (self.maximum - self.minimum + 1e-30)
            self.normalize_base = self.minimum
        else:
            self.normalize_scale = torch.ones((batch_size, ))
            self.normalize_base = torch.zeros((batch_size, ))

        self.B = torch.arange(batch_size)
        self.root_indices = torch.zeros((batch_size, ), dtype=torch.int64)
        self.branch_factor = branch_factor

        shape = (batch_size, (num_expansions + 1) * branch_factor)

        self.visit_count = torch.zeros(shape, dtype=torch.int32)

        # the player who will take action from the current state
        self.to_play = torch.zeros(shape, dtype=torch.int64)
        self.prior = torch.zeros(shape)

        # value for player 0
        self.value_sum = torch.zeros(shape)

        # 0 for not expanded
        self.first_child_index = torch.zeros(shape, dtype=torch.int64)
        self.model_state = common.zero_tensor_from_nested_spec(
            state_spec, shape)

        # reward for player 0
        self.reward = torch.zeros(shape)

        # action leading to this state
        self.action = torch.zeros(
            shape + action_spec.shape, dtype=action_spec.dtype)

        # in cpu
        self.game_over = torch.zeros(shape, dtype=torch.bool, device='cpu')

        # in cpu
        self.best_child_index = torch.zeros(
            shape, dtype=torch.int64, device='cpu')
        self.ucb_score = torch.zeros(shape)

    def update_value_stats(self, nodes, valid):
        """Update min max stats for each tree.

        Only valid values are used to update the stats.
        We maintain separate stats for each tree.
        The shapes of ``nodes`` and ``valid`` are [T, B].
        The invalid entries in ``values`` will be modified.
        """
        if self.fixed_bounds:
            return
        values = self.calc_value(nodes)
        invalid = ~valid
        values[invalid] = -MAXIMUM_FLOAT_VALUE
        self.maximum = torch.max(self.maximum, values.max(dim=0)[0])
        values[invalid] = MAXIMUM_FLOAT_VALUE
        self.minimum = torch.min(self.minimum, values.min(dim=0)[0])

        # We normalize only when we have set the maximum and minimum values.
        normalize = self.maximum > self.minimum
        self.normalize_scale = torch.where(
            normalize, 1 / (self.maximum - self.minimum + 1e-30),
            self.normalize_scale)
        self.normalize_base = torch.where(normalize, self.minimum,
                                          self.normalize_base)

    def normalize_value(self, value, batch_index):
        return self.normalize_scale[batch_index] * (
            value - self.normalize_base[batch_index])

    def calc_value(self, nodes):
        return self.value_sum[nodes] / (self.visit_count[nodes] + 1e-30)


def _nest_slice(nested, i):
    return nest.map_structure(lambda x: x[i], nested)


MCTSState = namedtuple("MCTSState", ["steps"])
MCTSInfo = namedtuple(
    "MCTSInfo",
    ["candidate_actions", "value", "candidate_action_visit_probabilities"])


@gin.configurable
class MCTSAlgorithm(OffPolicyAlgorithm):
    r"""Monte-Carlo Tree Search algorithm.

    The code largely follows the pseudocode of
    `Schrittwieser et. al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model <https://arxiv.org/abs/1911.08265>`_.
    The pseudocode can be downloaded from `<https://arxiv.org/src/1911.08265v2/anc/pseudocode.py>`_

    There are several differences:

    1. In this implementation, all values and rewards are for player 0. It seems
       that the values and rewards in the pseudocode can be either for player 0 or
       player 1 depending on who is on current turn. It makes reasoning the logic
       of the code more difficult and error prone. And it indeed seems there is
       a bug in the pseudocode related to this. More concretely, in the pseudocode,
       line 524 suggests that the value_sum is relative to a changing player;
       line 528 suggests that all the rewards along a path are relative to a same
       player; while line 499 combines the reward and value without considering the player.
    2. When calculating UCB score, the pseudocode normalizes value before adding
       with reward. We normalize after summing reward and value.
    3. When calculating UCB score, if the visit count of the node is 0, the value
       component of the score is 0 in the pseudocode. We use 0.5 instead so that
       it is not always the lowest score (or highest for player 1) no matter what
       the outcome of its siblings are.
    4. The pseudocode initializes the visit count of root to 0. We initialize it
       to 1 instead so that prior is not neglected in the first select_child().
       This is consistent with how the visit_count of other nodes are initialized.
       When other nodes are expanded, the immediately subsequenct backup() will
       make their initial visit_count to be 1.
    5. We add a game_over field to ModelOutput to indicate the game is over so
       that we won't keep expanding over that branch.
    """

    def __init__(
            self,
            observation_spec,
            action_spec: alf.BoundedTensorSpec,
            num_simulations,
            root_dirichlet_alpha,
            root_exploration_fraction: float,
            pb_c_init: float,  # c1 in Appendix B (2)
            pb_c_base: float,  # c2 in Appendix B (2)
            discount: float,
            is_two_player_game: bool,
            visit_softmax_temperature_fn: Callable,
            known_value_bounds=None,
            debug_summaries=False,
    ):
        r"""
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            num_simulations (int): the number of simulations per search (calls to model)
            root_dirichlet_alpha (float): alpha of dirichlet prior for exploration
            root_exploration_fraction (float): noise generated by the dirichlet distribution
                is combined with the action distribution from the model to be used as
                the action prior for the children of the root.
            pb_c_init (float): c1 of the pUCT rule in Appendix B, equation (2)
            pb_c_base (flaot): c2 of the pUCT rule in Appendix B, equation (2)
            discount (float): reward discount factor
            is_two_player_game (bool): whether this is a two player (zero-sum) game
            visit_softmax_temperature_fn (Callable): function for calculating the
                softmax temperature for sampling action based on the visit_counts
                of the children of the root. :math:`P(a) \propto \exp(visit\_count/t)`.
                This function is called as ``visit_softmax_temperature_fn(steps)``,
                where ``steps`` is a vector representing the number of steps in
                the games. And it is expected to return a float vector of the same
                shape as ``steps``.
            known_value_bounds (tuple|None): known bound of the values.
        """
        assert not nest.is_nested(
            action_spec), "nested action_spec is not supported"
        self._num_simulations = num_simulations
        self._known_value_bounds = known_value_bounds
        self._model = None
        self._pb_c_base = pb_c_base  # c2 in Appendix B (2)
        self._pb_c_init = pb_c_init  # c1 in Appendix B (2)
        self._discount = discount
        self._is_two_player_game = int(is_two_player_game)
        self._root_dirichlet_alpha = root_dirichlet_alpha
        self._root_exploration_fraction = root_exploration_fraction
        self._visit_softmax_temperature_fn = visit_softmax_temperature_fn
        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=MCTSState(
                steps=alf.TensorSpec((), dtype=torch.int64)),
            debug_summaries=debug_summaries)

    def set_model(self, model: MCTSModel):
        """Set the model used by the algorithm."""
        self._model = model

    @property
    def discount(self):
        return self._discount

    @torch.no_grad()
    def predict_step(self, time_step: TimeStep, state: MCTSState):
        """Predict the action.

        Args:
            time_step (TimeStep): the time step structure
            state (MCTSState): state of the algorithm
        Returns:
            AlgStep:
            - output (Tensor): the selected action
            - state (MCTSState): the state of the algorithm
            - info (MCTSInfo): info from the search. It might be used for
                training the model.
        """
        assert self._model is not None, "Need to call `set_model` before `predict_step`"
        model_output = self._model.initial_inference(time_step.observation)
        trees = _MCTSTree(self._num_simulations + 1, model_output,
                          self._known_value_bounds)
        to_plays = ()
        if self._is_two_player_game:
            # We may need the environment to pass to_play and pass to_play to
            # model because players may not always alternate in some game.
            # And for many games, it is not always obvious to determine which
            # player is in the current turn by looking at the board.
            to_plays = state.steps % 2
        roots = (trees.B, trees.root_indices)
        self._expand_node(
            trees,
            0,
            trees.root_indices,
            to_plays=to_plays,
            model_output=model_output,
            dirichlet_alpha=self._root_dirichlet_alpha,
            exploration_fraction=self._root_exploration_fraction)
        # paper pseudocode starts visit_count from 0
        # we start the root visit_count from 1 so that the first update_best_child
        # will be based on none-zero ucb_scores
        trees.visit_count[roots] = 1
        self._update_best_child(trees, roots)

        if self._is_two_player_game:
            to_plays = to_plays.cpu()
        for sim in range(1, self._num_simulations + 1):
            search_paths, path_lengths, last_to_plays = self._search(
                trees, to_plays)
            nodes_to_expand = search_paths[path_lengths - 1, trees.B]
            prev_nodes = search_paths[path_lengths - 2, trees.B]
            model_state = trees.model_state[trees.B, prev_nodes]
            action = trees.action[trees.B, nodes_to_expand]
            model_output = self._model.recurrent_inference(model_state, action)

            self._expand_node(
                trees,
                sim,
                nodes_to_expand,
                to_plays=last_to_plays,
                model_output=model_output)

            self._backup(
                trees,
                search_paths=search_paths,
                path_lengths=path_lengths,
                values=model_output.value,
                to_plays=last_to_plays)

        action, info = self._select_action(trees, state.steps)
        return AlgStep(
            output=action, state=MCTSState(steps=state.steps + 1), info=info)

    def _print_tree(self, trees: _MCTSTree, b):
        """Helper function to visualize the b-th search tree."""
        nodes = [(b, 0)]
        while len(nodes) > 0:
            node = nodes.pop(0)
            if trees.best_child_index[node] != 0:
                print(trees.model_state[node], trees.calc_value(node),
                      trees.prior[node], trees.ucb_score[node],
                      trees.visit_count[node])
                children = torch.arange(
                    trees.first_child_index[node],
                    trees.first_child_index[node] + trees.branch_factor)
                nodes.extend([(b, int(c)) for c in list(children)])

    def _search(self, trees, to_plays):
        """
        Returns:
            tuple:
            - search_paths: [T, B] int64 matrix, where T is the max length of the
                search paths. For paths whose length is shorter than T, search_path
                is padded with the last node index of that path.
            - path_lengths: [B] vector, length of each search path
            - last_to_plays: to_play for the last node of each path
        """
        best_child_index = trees.best_child_index
        game_over = trees.game_over
        search_paths = []
        B = trees.B
        nodes = trees.root_indices.cpu()
        path_lengths = torch.ones_like(trees.B, device='cpu')
        search_paths = [nodes]
        while True:
            done = torch.max(best_child_index[B, nodes] == 0,
                             game_over[B, nodes])
            if torch.all(done):
                break
            nodes = torch.where(done, nodes, best_child_index[B, nodes])
            path_lengths[~done] += 1
            search_paths.append(nodes)
            if self._is_two_player_game:
                to_plays = torch.where(done, to_plays,
                                       self._is_two_player_game - to_plays)

        search_paths = torch.stack(search_paths)
        return convert_device((search_paths, path_lengths, to_plays))

    def _select_action(self, trees, steps):
        roots = (trees.B, trees.root_indices)
        children = torch.arange(trees.branch_factor, 2 * trees.branch_factor)
        children = (trees.B.unsqueeze(1), children.unsqueeze(0))
        visit_counts = trees.visit_count[children]
        t = self._visit_softmax_temperature_fn(steps).unsqueeze(-1)
        action_id = torch.multinomial(
            F.softmax(visit_counts / t, dim=1), num_samples=1).squeeze(1)
        candidate_actions = trees.action[children]
        action = candidate_actions[trees.B, action_id]
        children_visit_count = trees.visit_count[children]
        parent_visit_count = (trees.visit_count[roots] - 1.0).unsqueeze(-1)
        info = MCTSInfo(
            candidate_actions=trees.action[children],
            value=trees.calc_value(roots),
            candidate_action_visit_probabilities=children_visit_count /
            parent_visit_count,
        )
        return action, info

    def _update_best_child(self, trees: _MCTSTree, parents):
        children = trees.first_child_index[parents].unsqueeze(-1)
        children = children + torch.arange(trees.branch_factor).unsqueeze(0)
        children = (parents[0].unsqueeze(-1), children)
        ucb_scores = self._ucb_score(trees, parents, children)
        invalid = trees.prior[children] == 0
        ucb_scores[invalid] = -MAXIMUM_FLOAT_VALUE
        trees.ucb_score[children] = ucb_scores

        best_child_index = trees.first_child_index[
            parents] + ucb_scores.argmax(dim=1)
        trees.best_child_index[(parents[0].cpu(),
                                parents[1].cpu())] = best_child_index.cpu()

    def _ucb_score(self, trees, parents, children):
        parent_visit_count = trees.visit_count[parents].unsqueeze(-1)
        child_visit_count = trees.visit_count[children]
        pb_c = torch.log((parent_visit_count + self._pb_c_base + 1.) /
                         self._pb_c_base) + self._pb_c_init
        pb_c = pb_c * torch.sqrt(parent_visit_count.to(
            torch.float32)) / (child_visit_count + 1.)

        prior_score = pb_c * trees.prior[children]
        value_score = trees.reward[
            children] + self._discount * trees.calc_value(children)
        value_score = trees.normalize_value(value_score, children[0])
        value_score[child_visit_count == 0] = 0.5
        if self._is_two_player_game:
            value_score = value_score * (
                (trees.to_play[parents] == 0) * 2 - 1).unsqueeze(-1)

        return prior_score + value_score

    def _backup(self, trees: _MCTSTree, search_paths, path_lengths, values,
                to_plays):
        B = trees.B.unsqueeze(0)
        T = search_paths.shape[0]
        depth = torch.arange(T).unsqueeze(-1)

        reward = trees.reward[B, search_paths]
        reward[depth > path_lengths] = 0.
        # [T+1, batch_size]
        reward = tensor_utils.tensor_extend_zero(reward)
        reward[path_lengths, B] = values
        discounts = (self._discount
                     **torch.arange(T + 1, dtype=torch.float32)).unsqueeze(-1)

        # discounted_return[t] = discount^t * reward[t]
        discounted_return = reward * discounts
        # discounted_return[t] = \sum_{s=t}^T discount^s * reward[s]
        discounted_return = reward.flip(0).cumsum(dim=0).flip(0)

        # discounted_return[t] = \sum_{s=t}^T discount^(s-t) * reward[s]
        discounted_return = discounted_return / discounts
        discounted_return = discounted_return[1:]

        value_sum = trees.value_sum[B, search_paths] + discounted_return

        valid = depth < path_lengths
        nodes = (B.expand(T, -1)[valid], search_paths[valid])
        trees.visit_count[nodes] += 1
        trees.value_sum[nodes] = value_sum[valid]
        trees.update_value_stats((B, search_paths), valid)
        self._update_best_child(trees, nodes)

    def _expand_node(
            self,
            trees: _MCTSTree,
            n,  # n-th expansion, zero-based
            node_indices,
            to_plays,
            model_output: ModelOutput,
            dirichlet_alpha=None,
            exploration_fraction=None):
        batch_size = model_output.action_probs.shape[0]
        indices = (trees.B, node_indices)
        if self._is_two_player_game:
            trees.to_play[indices] = to_plays
        first_child_index = (n + 1) * trees.branch_factor
        trees.first_child_index[indices] = first_child_index
        trees.game_over[(indices[0].cpu(),
                         indices[1].cpu())] = model_output.game_over.cpu()
        nest.map_structure(lambda ns, s: ns.__setitem__(indices, s),
                           trees.model_state, model_output.state)
        trees.reward[indices] = model_output.reward

        children = torch.arange(first_child_index,
                                first_child_index + trees.branch_factor)
        children = (trees.B.unsqueeze(1), children.unsqueeze(0))

        trees.action[children] = model_output.actions
        prior = model_output.action_probs

        if dirichlet_alpha is not None:
            noise_dist = td.Dirichlet(
                dirichlet_alpha * torch.ones(trees.branch_factor))
            noise = noise_dist.sample((batch_size, ))
            noise = noise * (prior != 0)
            noise = noise / noise.sum(dim=1, keepdims=True)
            prior = exploration_fraction * noise + (
                1 - exploration_fraction) * prior

        trees.prior[children] = prior


@gin.configurable
def create_atari_mcts(observation_spec, action_spec):
    """Helper function for creating MCTSAlgorithm for atari games."""

    def visit_softmax_temperature(num_moves):
        progress = Trainer.progress()
        if progress < 0.5:
            t = 1.0
        elif progress < 0.75:
            t = 0.5
        else:
            t = 0.25
        return t * torch.ones_like(num_moves, dtype=torch.float32)

    return MCTSAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        discount=0.997,
        num_simulations=50,
        root_dirichlet_alpha=0.25,
        root_exploration_fraction=0.25,
        pb_c_init=1.25,
        pb_c_base=19652,
        is_two_player_game=False,
        visit_softmax_temperature_fn=visit_softmax_temperature)


@gin.configurable
class VisitSoftmaxTemperatureByMoves(object):
    def __init__(self, move_temperature_pairs=[(29, 1.0), (10000, 0.0001)]):
        """Scheduling the temperature by move.

        Args:
            move_temperature_pairs (list[tuple]): each (moves, temperature) pair
                indicates using this temperature until so many moves have been
                played in the current game. The moves should be in ascending order.
                Note that ``num_moves`` used to calculate the temperature starts
                from 0.
        """
        self._move_temperature_pairs = list(
            reversed(move_temperature_pairs[:-1]))
        self._last_temperature = move_temperature_pairs[-1][1]

    def __call__(self, num_moves):
        t = torch.full_like(
            num_moves, self._last_temperature, dtype=torch.float32)
        for move, temp in self._move_temperature_pairs:
            t[num_moves <= move] = temp
        return t


@gin.configurable
def create_board_game_mcts(observation_spec,
                           action_spec,
                           dirichlet_alpha: float,
                           pb_c_init=1.25,
                           num_simulations=800,
                           debug_summaries=False):
    """Helper function for creating MCTSAlgorithm for board games."""

    def visit_softmax_temperature(num_moves):
        t = torch.ones_like(num_moves, dtype=torch.float32)
        # paper pseudocode uses 0.0
        # Current code does not support 0.0, so use a small value, which should
        # not make any difference since a difference of 1 for visit_count translates
        # to exp(1/0.0001) probability ratio.
        t[num_moves >= 30] = 0.0001
        return t

    return MCTSAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        discount=1.0,
        root_dirichlet_alpha=dirichlet_alpha,
        root_exploration_fraction=0.25,
        num_simulations=num_simulations,
        pb_c_init=pb_c_init,
        pb_c_base=19652,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        known_value_bounds=(-1, 1),
        is_two_player_game=True)


@gin.configurable
def create_go_mcts(observation_spec, action_spec, debug_summaries):
    return create_board_game_mcts(
        observation_spec,
        action_spec,
        dirichlet_alpha=0.03,
        debug_summaries=debug_summaries)


@gin.configurable
def create_chess_mcts(observation_spec, action_spec, debug_summaries):
    return create_board_game_mcts(
        observation_spec,
        action_spec,
        dirichlet_alpha=0.3,
        debug_summaries=debug_summaries)


@gin.configurable
def create_shogi_mcts(observation_spec, action_spec, debug_summaries):
    return create_board_game_mcts(
        observation_spec,
        action_spec,
        dirichlet_alpha=0.15,
        debug_summaries=debug_summaries)


@gin.configurable
def create_control_mcts(observation_spec,
                        action_spec,
                        num_simulations=50,
                        debug_summaries=False):
    """Helper function for creating MCTSAlgorithm for control tasks."""

    def visit_softmax_temperature(num_moves):
        progress = Trainer.progress()
        if progress < 0.5:
            t = 1.0
        elif progress < 0.75:
            t = 0.5
        else:
            t = 0.25
        return t * torch.ones_like(num_moves, dtype=torch.float32)

    return MCTSAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        discount=0.997,
        num_simulations=num_simulations,
        root_dirichlet_alpha=0.25,
        root_exploration_fraction=0.25,
        pb_c_init=1.25,
        pb_c_base=19652,
        is_two_player_game=False,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        debug_summaries=debug_summaries)


@gin.configurable
class VisitSoftmaxTemperatureByProgress(object):
    def __init__(self,
                 progress_temperature_pairs=[(0.5, 1.0), (0.75, 0.5), (1,
                                                                       0.25)]):
        """Scheduling the temperature by training progress.

        Args:
            progress_temperature_pairs (list[tuple]): each (progress, temperature)
                pair indicates using this temperature until this training
                progress. Note that progress should be in ascending order.
        """
        self._progress_temperature_pairs = progress_temperature_pairs

    def __call__(self, num_moves):
        progress = Trainer.progress()
        for p, t in self._progress_temperature_pairs:
            if progress < p:
                break
        return torch.full_like(num_moves, t, dtype=torch.float32)
