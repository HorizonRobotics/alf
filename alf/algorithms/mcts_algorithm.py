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

        parent_shape = (batch_size, num_expansions)
        children_shape = (batch_size, num_expansions, branch_factor)

        self.visit_count = torch.zeros(parent_shape, dtype=torch.int32)

        # the player who will take action from the current state
        self.to_play = torch.zeros(parent_shape, dtype=torch.int64)
        self.prior = torch.zeros(children_shape)

        # value for player 0
        self.value_sum = torch.zeros(parent_shape)

        # 0 for not expanded, value in range [0, num_expansions)
        self.children_index = torch.zeros(children_shape, dtype=torch.int64)
        self.model_state = common.zero_tensor_from_nested_spec(
            state_spec, parent_shape)

        # reward for player 0
        self.reward = None
        if isinstance(model_output.reward, torch.Tensor):
            self.reward = torch.zeros(parent_shape)

        self.action = None
        if isinstance(model_output.actions, torch.Tensor):
            # candidate actions for this state
            self.action = torch.zeros(
                children_shape + action_spec.shape, dtype=action_spec.dtype)

        self.game_over = None
        if isinstance(model_output.game_over, torch.Tensor):
            self.game_over = torch.zeros(parent_shape, dtype=torch.bool)

        # value in range [0, branch_factor)
        self.best_child_index = torch.zeros(parent_shape, dtype=torch.int64)
        self.ucb_score = torch.zeros(children_shape)

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
        return self.value_sum[nodes] / self.visit_count[nodes]


def _nest_slice(nested, i):
    return nest.map_structure(lambda x: x[i], nested)


MCTSState = namedtuple("MCTSState", ["steps"])
MCTSInfo = namedtuple(
    "MCTSInfo", ["candidate_actions", "value", "candidate_action_policy"])


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
    6. We add support for using a stochastic policy instead of using UCB to
       do the search/learn/act. This can be enabled by setting  ``act_with_exploratin_policy``
       ``search_with_exploratin_policy``, ``learn_with_exploratin_policy`` to True.
       See `Grill et. al. Monte-Carlo tree search as regularized policy optimization
       <https://arxiv.org/abs/2007.12509>`_for reference.

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
            unexpanded_value_score=0.5,
            act_with_exploratin_policy=False,
            search_with_exploratin_policy=False,
            learn_with_exploratin_policy=False,
            debug_summaries=False,
    ):
        r"""
        Args:
            observation_spec (nested TensorSpec): if the observation is a dictionary,
                ``MCTSAlgorithm`` will use the following three fields if they are
                contained in the dictionary:
                1. valid_action_mask: a bool Tensor to indicate which actions are
                    allowed. It will be used to mask out invalid actions. If not
                    provided, all possible actions are considered.
                2. steps: int32 Tensor to indicate the number of steps since the
                    beginning of the game. If not provided, an internal counter
                    will be used. However, this internal count will not be
                    correct if the algorithm is used to play against human because
                    it is not used to generate all the moves of both players.
                3. to_play: int8 Tensor whose elements are 0 or 1 to indicate who
                    is the player to take the action. If not provided, stesp % 2
                    wil be used as to_play.
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
            unexpanded_value_score (float|str): The value score for an unexpanded
                child. If 'max'/'min'/'mean', will use the maximum/minimum/mean
                of the value scores of the expanded siblings. If 'none', when
                exploration policy is used, will keep the policy for the unexpanded
                children same as prior; when exporation is not used, 'none' behaves
                same as 'min'.
            act_with_exploratin_policy (bool):
            search_with_exploratin_policy (bool):
            learn_with_exploratin_policy (bool):
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
        if isinstance(unexpanded_value_score, str):
            assert unexpanded_value_score in ('max', 'min', 'mean', 'none'), (
                "Unsupported unexpanded_value_score=%s" %
                unexpanded_value_score)
        self._unexpanded_value_score = unexpanded_value_score
        self._act_with_exploratin_policy = act_with_exploratin_policy
        self._search_with_exploratin_policy = search_with_exploratin_policy
        self._learn_with_exploratin_policy = learn_with_exploratin_policy

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
        if isinstance(time_step.observation, dict):
            valid_action_mask = time_step.observation.get(
                'valid_action_mask', None)
            to_plays = time_step.observation.get('to_play', None)
            steps = time_step.observation.get('steps', state.steps)
        else:
            valid_action_mask = None
            to_plays = None
            steps = state.steps

        model_output = self._model.initial_inference(time_step.observation)

        if valid_action_mask is not None:
            # mask out invalid actions
            assert model_output.actions == ()
            model_output = model_output._replace(
                action_probs=model_output.action_probs *
                valid_action_mask.to(torch.float32))

        trees = _MCTSTree(self._num_simulations + 1, model_output,
                          self._known_value_bounds)
        if self._is_two_player_game and to_plays is None:
            # We may need the environment to pass to_play and pass to_play to
            # model because players may not always alternate in some game.
            # And for many games, it is not always obvious to determine which
            # player is in the current turn by looking at the board.
            to_plays = steps % 2
        roots = (trees.B, trees.root_indices)
        self._expand_node(
            trees,
            0,
            to_plays=to_plays,
            model_output=model_output,
            dirichlet_alpha=self._root_dirichlet_alpha,
            exploration_fraction=self._root_exploration_fraction)
        # paper pseudocode starts visit_count from 0
        # we start the root visit_count from 1 so that the first update_best_child
        # will be based on none-zero ucb_scores
        trees.visit_count[roots] = 1
        trees.value_sum[roots] = model_output.value
        self._update_best_child(trees, roots)

        for sim in range(1, self._num_simulations + 1):
            search_paths, path_lengths, last_to_plays = self._search(
                trees, to_plays)
            prev_nodes = search_paths[path_lengths - 2, trees.B]
            model_state = trees.model_state[trees.B, prev_nodes]
            best_child_index = trees.best_child_index[trees.B, prev_nodes]
            if trees.action is None:
                action = best_child_index
            else:
                action = trees.action[trees.B, prev_nodes, best_child_index]
            model_output = self._model.recurrent_inference(model_state, action)
            self._expand_node(
                trees, sim, to_plays=last_to_plays, model_output=model_output)
            # If a child is expanded before, use its existing node
            child_index = trees.children_index[trees.
                                               B, prev_nodes, best_child_index]
            child_index[(child_index == 0) & (path_lengths > 1)] = sim
            trees.children_index[trees.
                                 B, prev_nodes, best_child_index] = child_index
            search_paths[path_lengths - 1, trees.B] = child_index
            self._backup(
                trees,
                search_paths=search_paths,
                path_lengths=path_lengths,
                values=model_output.value)

        action, info = self._select_action(trees, steps)
        return AlgStep(
            output=action, state=MCTSState(steps=state.steps + 1), info=info)

    def _print_tree(self, trees: _MCTSTree, b):
        """Helper function to visualize the b-th search tree."""
        nodes = [(b, 0)]
        while len(nodes) > 0:
            node = nodes.pop(0)
            print(trees.model_state[node], trees.calc_value(node),
                  trees.prior[node], trees.ucb_score[node],
                  trees.visit_count[node])
            children = trees.children_index[node]
            nodes.extend([(b, int(c)) for c in list(children) if c != 0])

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
        children_index = trees.children_index
        best_child_index = trees.best_child_index
        game_over = trees.game_over
        search_paths = []
        B = trees.B
        nodes = trees.root_indices
        path_lengths = torch.ones_like(B)
        search_paths = [nodes]
        if game_over is not None:
            done = game_over[B, nodes]
        else:
            done = torch.zeros_like(B, dtype=torch.bool)
        while not torch.all(done):
            nodes = torch.where(
                done, nodes,
                children_index[B, nodes, best_child_index[B, nodes]])
            path_lengths[~done] += 1
            search_paths.append(nodes)
            if self._is_two_player_game:
                to_plays = torch.where(done, to_plays,
                                       self._is_two_player_game - to_plays)
            # nodes == 0 means unexpanded child
            done = nodes == 0
            if game_over is not None:
                done = game_over[B, nodes] | done

        search_paths = torch.stack(search_paths)
        return search_paths, path_lengths, to_plays

    def _select_action(self, trees, steps):
        roots = (trees.B, trees.root_indices)

        if self._act_with_exploratin_policy or self._learn_with_exploratin_policy:
            policy = self._calculate_policy(trees, roots)

        if not self._act_with_exploratin_policy or not self._learn_with_exploratin_policy:
            children = trees.children_index[roots]
            children = (trees.B.unsqueeze(1), children)
            visit_counts = trees.visit_count[children]
            visit_counts[children[1] == 0] = 0

        if self._act_with_exploratin_policy:
            probs = policy
        else:
            probs = visit_counts
        t = self._visit_softmax_temperature_fn(steps).unsqueeze(-1)
        probs = F.softmax((probs + 1e-30).log() / t, dim=1)
        probs = probs * (trees.prior[roots] > 0).to(torch.float32)
        action_id = torch.multinomial(probs, num_samples=1).squeeze(1)

        if trees.action is not None:
            candidate_actions = trees.action[roots]
            action = candidate_actions[trees.B, action_id]
        else:
            candidate_actions = ()
            action = action_id

        if not self._learn_with_exploratin_policy:
            parent_visit_count = (trees.visit_count[roots] - 1.0).unsqueeze(-1)
            policy = visit_counts / parent_visit_count
        info = MCTSInfo(
            candidate_actions=candidate_actions,
            value=trees.calc_value(roots),
            candidate_action_policy=policy,
        )
        return action, info

    def _update_best_child(self, trees: _MCTSTree, parents):
        if self._search_with_exploratin_policy:
            self._sample_child(trees, parents)
        else:
            self._ucb_child(trees, parents)

    def _ucb_child(self, trees: _MCTSTree, parents):
        """Get child using UCB score."""
        ucb_scores = self._ucb_score(trees, parents)
        trees.ucb_score[parents] = ucb_scores

        best_child_index = ucb_scores.argmax(dim=1)
        trees.best_child_index[parents] = best_child_index

    def _ucb_score(self, trees, parents):
        prior = trees.prior[parents]
        value_score, child_visit_count = self._value_score(trees, parents)
        value_score[prior == 0] = -MAXIMUM_FLOAT_VALUE
        parent_visit_count = trees.visit_count[parents].unsqueeze(-1)
        pb_c = torch.log((parent_visit_count + self._pb_c_base + 1.) /
                         self._pb_c_base) + self._pb_c_init
        pb_c = pb_c * torch.sqrt(parent_visit_count.to(
            torch.float32)) / (child_visit_count + 1.)
        prior_score = pb_c * prior
        return prior_score + value_score

    def _value_score(self, trees: _MCTSTree, parents):
        children = trees.children_index[parents]
        children = (parents[0].unsqueeze(-1), children)
        child_visit_count = trees.visit_count[children]
        unexpanded = children[1] == 0
        child_visit_count[unexpanded] = 0
        value_score = self._discount * trees.calc_value(children)
        if trees.reward is not None:
            value_score = value_score + trees.reward[children]
        value_score = trees.normalize_value(value_score, children[0])
        if not isinstance(self._unexpanded_value_score, str):
            value_score[unexpanded] = self._unexpanded_value_score
        if self._is_two_player_game:
            value_score = value_score * (
                (trees.to_play[parents] == 0) * 2 - 1).unsqueeze(-1)

        if self._unexpanded_value_score == 'max':
            # Set the value of unexpanded children to maximum so that the probability
            # of it being selected is proportional to its prior.
            value_score[unexpanded] = -MAXIMUM_FLOAT_VALUE
            max_value_score = value_score.max(dim=1, keepdim=True)[0]
            # max_value_score is not defined if none of the children is expanded.
            # set it to 0
            max_value_score[max_value_score == -MAXIMUM_FLOAT_VALUE] = 0.
            value_score = torch.where(unexpanded, max_value_score, value_score)
        elif self._unexpanded_value_score in ('min', 'none'):
            value_score[unexpanded] = MAXIMUM_FLOAT_VALUE
            min_value_score = value_score.min(dim=1, keepdim=True)[0]
            # min_value_score is not defined if none of the children is expanded.
            # set it to 0
            min_value_score[min_value_score == MAXIMUM_FLOAT_VALUE] = 0.
            value_score = torch.where(unexpanded, min_value_score, value_score)
        elif self._unexpanded_value_score == 'mean':
            value_score[unexpanded] = 0.
            n = (~unexpanded).sum(dim=1, keepdim=True) + 1e-30
            mean_value_score = value_score.sum(dim=1, keepdim=True) / n
            value_score = torch.where(unexpanded, mean_value_score,
                                      value_score)

        return value_score, child_visit_count

    def _calculate_policy(self, trees: _MCTSTree, parents):
        parent_visit_count = trees.visit_count[parents].unsqueeze(-1)
        c = torch.log((parent_visit_count + self._pb_c_base + 1.) /
                      self._pb_c_base) + self._pb_c_init
        c = c / parent_visit_count.to(torch.float32).sqrt()
        prior = trees.prior[parents]
        value_score, child_visit_count = self._value_score(trees, parents)
        if self._unexpanded_value_score != 'none':
            return calculate_exploration_policy(value_score, prior, c)[0]
        else:
            # For 'none', we keep the policy for the unexpanded children same
            # as prior.
            expanded = child_visit_count != 0
            expanded_prior = prior * (expanded.to(torch.float32) + 1e-30)
            sum_expanded = expanded_prior.sum(dim=1, keepdim=True)
            expanded_prior = expanded_prior / sum_expanded
            policy = calculate_exploration_policy(value_score, expanded_prior,
                                                  c)[0]
            return torch.where(expanded, policy * sum_expanded, prior)

    def _sample_child(self, trees: _MCTSTree, parents):
        """Get child by sampling from exploration policy."""
        policy = self._calculate_policy(trees, parents)
        action = torch.multinomial(policy, 1).squeeze(1)
        trees.best_child_index[parents] = action

    def _backup(self, trees: _MCTSTree, search_paths, path_lengths, values):
        B = trees.B.unsqueeze(0)
        T = search_paths.shape[0]
        depth = torch.arange(T).unsqueeze(-1)

        if trees.reward is not None:
            reward = trees.reward[B, search_paths]
            reward[depth > path_lengths] = 0.
            # [T+1, batch_size]
            reward = tensor_utils.tensor_extend_zero(reward)
            reward[path_lengths, B] = values
            discounts = (self._discount**torch.arange(
                T + 1, dtype=torch.float32)).unsqueeze(-1)

            # discounted_return[t] = discount^t * reward[t]
            discounted_return = reward * discounts
            # discounted_return[t] = \sum_{s=t}^T discount^s * reward[s]
            discounted_return = reward.flip(0).cumsum(dim=0).flip(0)

            # discounted_return[t] = \sum_{s=t}^T discount^(s-t) * reward[s]
            discounted_return = discounted_return / discounts
            discounted_return = discounted_return[1:]
        else:
            # [T, 1]
            steps = torch.arange(1, T + 1, dtype=torch.float32).unsqueeze(-1)
            # [T, B]
            discounts = self._discount**(path_lengths.unsqueeze(0) - steps)
            discounted_return = values.unsqueeze(0) * discounts

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
            to_plays,
            model_output: ModelOutput,
            dirichlet_alpha=None,
            exploration_fraction=0.):
        if self._is_two_player_game:
            trees.to_play[:, n] = to_plays
        if trees.game_over is not None:
            trees.game_over[:, n] = model_output.game_over

        def _set_tree_state(ts, s):
            ts[:, n] = s

        nest.map_structure(_set_tree_state, trees.model_state,
                           model_output.state)
        if trees.reward is not None:
            trees.reward[:, n] = model_output.reward
        if trees.action is not None:
            trees.action[:, n] = model_output.actions
        prior = model_output.action_probs

        if exploration_fraction > 0.:
            batch_size = model_output.action_probs.shape[0]
            noise_dist = td.Dirichlet(
                dirichlet_alpha * torch.ones(trees.branch_factor))
            noise = noise_dist.sample((batch_size, ))
            noise = noise * (prior != 0)
            noise = noise / noise.sum(dim=1, keepdim=True)
            prior = exploration_fraction * noise + (
                1 - exploration_fraction) * prior

        trees.prior[:, n] = prior


def calculate_exploration_policy(value, prior, c, tol=1e-6):
    r"""Calculate exploration policy.

    The policy is based on
    `Grill et. al. Monte-Carlo tree search as regularized policy optimization
    <https://arxiv.org/abs/2007.12509>`_

    Notation:

        q: prior policy
        p: sampling probability
        v: value

    The exploration policy is found by minimizing the following:

    .. math::

        p = -\arg\min_p E_p(v) + c KL(q||p)

    which leads to the following solution:

    .. math::

        p_i = c\frac{q_i}{\alpha - v_i}

    where :math:`\alpha \ge \max_i(v_i)` is such that :math:`\sum_i p_i = 1`


    To make the solving numerically more stable and efficient, we reparameterize
    the problem to the following:

    .. math::

        \begin{array}{ll}
          &  v^* = \max_i v_i \\
          &  \alpha = v^* + c \beta \\
          &  u_i = \frac{v_i - v^*}{c} \\
          &  p_i = \frac{q_i}{\beta - u_i} \\
        \end{array}

    With this reparametrization, we need to find :math:`\beta>0` s.t.

    .. math::

        \sum_i \frac{q_i}{\beta - u_i} = 1

    We use Newton's method to update :math:`\beta` iteratively:

        .. math::

            \beta \leftarrow \beta - \frac{f(\beta)}{f'(\beta)}
            = \beta + \frac{\sum_i \frac{q_i}{\beta - v_i} - 1}{\sum_i \frac{q_i}{(\beta - v_i)^2}}

    where :math:`f(\beta) = \sum_i \frac{q_i}{\beta - u_i} - 1` and :math:`f'(\beta)`
    is the derivative of :math:`f(\beta)`. Since :math:`f(\beta)` is convex,
    starting the iteration with a :math:`\beta` s.t. :math:`f(\beta) > 0` gaurantees
    the convergence. In practice, we find that about 10 iterations can reach
    tolerance of 1e-6. Newton's method is much faster than binary search.

    Args:
        value (Tensor): [N, K] Tensor
        prior (Tensor): [N, K] Tensor
        c (Tensor): [N, 1] Tensor
        tol (float): Desired acurracy. The result satisfy :math:`|\sum_i p_i - 1| \le tol`
    Returns:
        tuple:
        - Tensor: [N, K], the exploration policy
        - int: the number of iterations
    """
    batch_size = value.shape[0]
    assert value.shape == prior.shape
    assert c.shape == (batch_size, 1)
    value[prior == 0] = -MAXIMUM_FLOAT_VALUE
    v_max = value.max(dim=1, keepdim=True)[0]
    u = (value - v_max) / c

    beta = (prior + u).max(dim=1, keepdim=True)[0]

    i = 0
    while True:
        i += 1
        beta_u = beta - u
        p = prior / beta_u
        sum_p = p.sum(dim=1, keepdim=True)
        diff = sum_p - 1
        if (diff < tol).all():
            break
        d = (p / beta_u).sum(dim=1, keepdim=True)
        beta = beta + diff / d
    return p, i


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
        # not make any difference since a difference of 1e-3 for visit probability
        # to about exp(1e-3*1e10) probability ratio.
        t[num_moves >= 30] = 1e-10
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
