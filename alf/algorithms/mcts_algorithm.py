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

import abc
from collections import OrderedDict
import math
import numpy
import torch
import torch.distributions as td
import torch.nn.functional as F
from typing import Callable

import alf
from alf.data_structures import AlgStep, namedtuple, TimeStep
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.utils import dist_utils
from alf import nest
from alf.trainers.policy_trainer import Trainer

ModelOutput = namedtuple(
    'ModelOutput',
    [
        'value',  # value for the player 0
        'reward',  # reward for the player 0
        'game_over',  # whether the game is over
        'action_distribution',
        'state'
    ])


class MCTSModel(abc.ABC):
    """The interface for the model used by MCTSAlgorithm."""

    @abc.abstractmethod
    def initial_inference(self, observation):
        """Generate the initial prediction given observation.

        Returns:
            ModelOutput: the prediction
        """

    @abc.abstractmethod
    def recurrent_inference(self, state, action):
        """Generate prediction given state and action.

        Args:
            state (Tensor): the latent state of the model. The state should be from
                previous call of ``initial_inference`` or ``recurrent_inference``.
            action (Tensor): the imagined action
        Returns:
            ModelOutput: the prediction
        """


MAXIMUM_FLOAT_VALUE = float('inf')


class Node(object):
    """Node in the MCTS tree."""

    def __init__(self, prior: float):
        self.visit_count = 0

        # the player who will take action from the current state
        self.to_play = 0

        self.prior = prior

        # value for player 0
        self.value_sum = 0

        self.children = {}
        self.model_state = None

        # reward for player 0
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


MAXIMUM_FLOAT_VALUE = float('inf')


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds):
        self.maximum = known_bounds[1] if known_bounds else -float('inf')
        self.minimum = known_bounds[0] if known_bounds else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def _nest_slice(nested, i):
    return nest.map_structure(lambda x: x[i], nested)


MCTSState = namedtuple("MCTSState", ["steps"])
MCTSInfo = namedtuple(
    "MCTSInfo",
    ["candidate_actions", "value", "candidate_action_visit_probabilities"])


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
       component of the score is 0 in the pseudocode. We use normlize(0) instead.
    4. The pseudocode intilizes the visit count of root to 0. We initialize it
       to 1 instead so that prior is not neglected in the first select_child().
       And this is consistent with how the visit_count of other nodes are initialized.
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
            num_sampled_actions: int = 16,  # for continuous action
            known_value_bounds=(None, None),
    ):
        r"""
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            num_simulations (int): the number of simulations per search (calls to model)
            root_dirichlet_alpha (float): alpha of dirichlet prior for exploration
            root_exploration_fraction (float): noise generated by the dirichlet distribution
                is combined with the action distribution from the model to used as
                the action prior for the root.
            pb_c_init (float): c1 of the pUCT rule in Appendix B, equation (2)
            pb_c_base (flaot): c2 of the pUCT rule in Appendix B, equation (2)
            discount (float): reward discount factor
            is_two_player_game (bool): whether this is a two player (zero-sum) game
            visit_softmax_temperature_fn (Callable): function for calculating the
                softmax temperature for sampling action based on the visit_counts
                of the children of the root. :math:`P(a) \propto \exp(visit_count/t)`
            num_sampled_actions (int): number of sampled action for expanding a node
                for continuous action distributions
            known_value_bounds (None, None): known bound of the value.
        """
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
                steps=alf.TensorSpec((), dtype=torch.int64)))

    def set_model(self, model: MCTSModel):
        """Set the model used by the algorithm."""
        self._model = model

    def predict_step(self, time_step: TimeStep, state: MCTSState):
        """Predict the action.

        Args:
            time_step (TimeStep): the time step structure
            state (MCTSState): state of the algorithm
        Returns:
            AlgOutput:
            - output (Tensor): the selected action
            - state (MCTSState): the state of the algorithm
            - info (MCTSInfo): info from the search. It might be used for
                training the model.
        """
        min_max_stats = MinMaxStats(self._known_value_bounds)

        batch_size = time_step.step_type.shape[0]
        roots = [Node(0) for _ in range(batch_size)]
        model_output = self._model.initial_inference(time_step.observation)
        model_output_spec = dist_utils.extract_spec(model_output)
        model_output = dist_utils.distributions_to_params(model_output)
        if self._is_two_player_game:
            to_plays = state.steps % 2
        else:
            to_plays = torch.zeros((batch_size), dtype=torch.int64)

        for i in range(batch_size):
            # paper pseudocode starts from 0
            # we start the root visit_count with 1 so that the first select_child
            # will be based on none-zero ucb_score
            roots[i].visit_count = 1
            self._expand_node(
                roots[i],
                to_play=to_plays[i],
                model_output=dist_utils.params_to_distributions(
                    _nest_slice(model_output, i), model_output_spec),
                dirichlet_alpha=self._root_dirichlet_alpha,
                exploration_fraction=self._root_exploration_fraction)

        for _ in range(self._num_simulations):
            search_paths = []
            last_actions = []
            last_to_plays = []
            for i in range(batch_size):
                to_play = to_plays[i]
                node = roots[i]
                search_path = [node]
                while node.expanded() and not node.game_over:
                    action, node = self._select_child(node, min_max_stats)
                    search_path.append(node)
                    to_play = self._is_two_player_game - to_play
                search_paths.append(search_path)
                last_actions.append(action)
                last_to_plays.append(to_play)

            model_states = [
                search_path[-2].model_state for search_path in search_paths
            ]
            model_state = nest.utils.stack_nests(model_states)
            action = torch.stack(last_actions)
            model_output = self._model.recurrent_inference(model_state, action)
            model_output = dist_utils.distributions_to_params(model_output)

            for i in range(batch_size):
                search_path = search_paths[i]
                self._expand_node(
                    search_path[-1],
                    to_play=last_to_plays[i],
                    model_output=dist_utils.params_to_distributions(
                        _nest_slice(model_output, i), model_output_spec))
                self._backup(
                    search_path=search_path,
                    value=model_output.value[i],
                    to_play=last_to_plays[i],
                    min_max_stats=min_max_stats)

        action = self._select_action(roots, state.steps)
        return AlgStep(
            output=action,
            state=MCTSState(steps=state.steps + 1),
            info=self._make_info(roots))

    def _print_tree(self, root, min_max_stats):
        """Helper function to visualize the search tree."""
        nodes = [(root, None)]
        while len(nodes) > 0:
            node, parent = nodes.pop(0)
            if node.model_state is not None:
                if parent is not None:
                    ucb_score = self._ucb_score(parent, node, min_max_stats)
                else:
                    ucb_score = 0.
                print(node.model_state, node.value(), node.prior, ucb_score,
                      node.visit_count)
                children = [(c, node) for c in node.children.values()]
                nodes.extend(children)

    def _make_info(self, roots):
        actions = []
        values = []
        probs = []
        for root in roots:
            actions.extend([a for a in root.children.keys()])
            values.append(root.value())
            probs.append([
                child.visit_count / root.visit_count
                for _, child in root.children.items()
            ])

        return MCTSInfo(
            candidate_actions=torch.stack(actions).reshape(len(roots), -1),
            value=torch.tensor(values),
            candidate_action_visit_probabilities=torch.tensor(probs))

    def _select_action(self, roots, steps):
        visit_counts = [[
            child.visit_count for _, child in root.children.items()
        ] for root in roots]
        t = self._visit_softmax_temperature_fn(steps)
        visit_counts = torch.tensor(visit_counts)
        action = torch.multinomial(
            F.softmax(visit_counts / t, dim=1), num_samples=1).squeeze(1)
        return action

    def _select_child(self, node: Node, min_max_stats: MinMaxStats):
        score_action_child = [(self._ucb_score(node, child, min_max_stats),
                               action, child)
                              for action, child in node.children.items()]
        _, action, child = max(score_action_child)
        return action, child

    def _ucb_score(self, parent: Node, child: Node,
                   min_max_stats: MinMaxStats):
        if child.prior == 0.:
            return torch.tensor(-MAXIMUM_FLOAT_VALUE)
        pb_c = math.log((parent.visit_count + self._pb_c_base + 1) /
                        self._pb_c_base) + self._pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = numpy.float32(pb_c) * child.prior

        value_score = child.reward + self._discount * child.value()
        value_score = min_max_stats.normalize(value_score)
        if parent.to_play != 0:
            value_score = -value_score
        return prior_score + value_score

    def _backup(self, search_path, value, to_play, min_max_stats: MinMaxStats):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())
            value = node.reward + self._discount * value

    def _expand_node(self,
                     node: Node,
                     to_play: int,
                     model_output: ModelOutput,
                     dirichlet_alpha=None,
                     exploration_fraction=None):
        node.to_play = to_play
        node.game_over = model_output.game_over
        node.model_state = model_output.state
        node.reward = model_output.reward
        node.policy = model_output.action_distribution
        if node.game_over:
            return

        if type(node.policy) == td.Categorical:
            probs = node.policy.probs
            actions = torch.arange(probs.shape[0])
        else:
            actions = node.policy.sample(
                sample_shape=(self._num_sampled_actions, ))
            # Since actions are sampled from the action distrution, the probability
            # is implicitly reflected in the sampled actions.
            # But we might want to use the actual log_prob of the actions
            probs = torch.ones((actions.shape[0], )) / actions.shape[0]

        if dirichlet_alpha is not None:
            noise = numpy.random.dirichlet(
                [dirichlet_alpha] * len(actions)).astype(numpy.float32)
            noise = torch.as_tensor(noise)
            noise = noise * (probs != 0)
            noise = noise / noise.sum()
            probs = exploration_fraction * noise + (
                1 - exploration_fraction) * probs

        node.children = OrderedDict(
            [(a, Node(prob)) for a, prob in zip(actions, probs)])


def create_atari_mcts(observation_spec, action_spec):
    """Helper function for creating MCTSAlgorithm for atari games."""

    def visit_softmax_temperature(num_moves):
        progress = Trainer.progress()
        if progress < 0.5:
            return 1.0
        elif progress < 0.75:
            return 0.5
        else:
            return 0.25

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


def create_board_game_mcts(observation_spec,
                           action_spec,
                           dirichlet_alpha: float,
                           pb_c_init=1.25,
                           num_simulations=800):
    """Helper function for creating MCTSAlgorithm for board games."""

    def visit_softmax_temperature(num_moves):
        if num_moves < 30:
            return 1.0
        else:
            # paper pseudocode uses 0.0
            # Current code does not support 0.0, so use a small value, which should
            # not make any difference since a difference of 1 for visit_count translates
            # to exp(1/0.0001) probability ratio.
            return 0.0001  # Play according to the max.

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
