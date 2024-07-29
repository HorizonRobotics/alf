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
"""Monte-Carlo Tree Search."""
from absl import logging

import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F
from typing import Callable, Optional

import alf
from alf import TensorSpec
from alf.data_structures import AlgStep, LossInfo, namedtuple, TimeStep
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.utils import dist_utils, summary_utils
from alf import nest
from alf.trainers.policy_trainer import Trainer
from alf.utils import common, tensor_utils, action_samplers
from .mcts_models import MCTSModel, ModelOutput

MAXIMUM_FLOAT_VALUE = float('inf')


class _MCTSTrees(object):
    def __init__(self, num_expansions, model_output, known_value_bounds,
                 num_best_children, discount, value_min_max_delta):
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
        self.discount = discount
        self.value_min_max_delta = value_min_max_delta

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
        self.original_value = torch.zeros_like(
            self.value_sum, dtype=model_output.value.dtype)

        # 0 for not expanded, value in range [0, num_expansions)
        self.children_index = torch.zeros(children_shape, dtype=torch.int64)
        self.model_state = common.zero_tensor_from_nested_spec(
            state_spec, parent_shape)

        # reward for player 0
        self.reward = None
        if isinstance(model_output.reward, torch.Tensor):
            # we use value.dtype because the reward.dtype from the intial_inference
            # can be different from the reward.dtype from the recurrent_inference
            self.reward = torch.zeros(
                parent_shape, dtype=model_output.value.dtype)

        self.action = None
        if isinstance(model_output.actions, torch.Tensor):
            # candidate actions for this state
            self.action = torch.zeros(
                children_shape + action_spec.shape, dtype=action_spec.dtype)

        self.game_over = None
        if isinstance(model_output.game_over, torch.Tensor):
            self.game_over = torch.zeros(parent_shape, dtype=torch.bool)

        shape = ()
        if num_best_children > 1:
            shape = (num_best_children, )
        # value in range [0, branch_factor)
        self.best_child_index = torch.zeros(
            parent_shape + shape, dtype=torch.int64)
        self.ucb_score = torch.zeros(children_shape)

    def update_value_stats(self, nodes, valid=None):
        """Update min max stats for each tree.

        Only valid values are used to update the stats.
        We maintain separate stats for each tree.
        The shapes of ``nodes`` and ``valid`` are [T, B] or [T, B, psims].
        """
        if self.fixed_bounds:
            return
        values = self.discount * self.calc_value(nodes)
        if self.reward is not None:
            values = self.reward[nodes] + values
        if valid is not None:
            invalid = ~valid
            values[invalid] = -MAXIMUM_FLOAT_VALUE
        max = values.max(dim=0)[0]
        if valid is not None:
            values[invalid] = MAXIMUM_FLOAT_VALUE
        min = values.min(dim=0)[0]
        if values.ndim == 3:
            max = max.max(dim=1)[0]
            min = min.min(dim=1)[0]
        self.maximum = torch.max(self.maximum, max)
        self.minimum = torch.min(self.minimum, min)

        # We normalize only when we have set the maximum and minimum values.
        normalize = self.maximum > self.minimum
        self.normalize_scale = torch.where(
            normalize, 1 /
            (self.maximum - self.minimum).clamp(min=self.value_min_max_delta),
            self.normalize_scale)
        self.normalize_base = torch.where(normalize, self.minimum,
                                          self.normalize_base)

    def normalize_value(self, value, batch_index):
        return self.normalize_scale[batch_index] * (
            value - self.normalize_base[batch_index])

    def calc_value(self, nodes):
        return self.value_sum[nodes] / self.visit_count[nodes]

    def get_children(self, parents):
        """
        Get the index tuple of the children of parents. It has one more rank of
        branch_factor than parents
        """
        children = self.children_index[parents]
        return (parents[0].unsqueeze(1), children)

    def get_model_state(self, nodes):
        return nest.map_structure(lambda x: x[nodes], self.model_state)

    def visualize(self, b: int):
        """Visualize the b-th tree as a directed graph.

        The graph is in Graphviz format and can be rendered

        1. Directly in Jupyter Notebook, or
        2. rendered as pdf, with ``trees.visualize(0).render(<path>)``.

        """
        try:
            import graphviz
        except ImportError:
            raise RuntimeError(
                'Need "graphviz" installed if you want to visualize MCTS')

        dot = graphviz.Digraph()
        dot.attr('node', shape='record')

        def _add_node(name: str, properties: dict):
            keys = "|".join(properties.keys())
            values = '|'.join([
                f'{x:.4f}' if type(x) is float else f'{x}'
                for x in properties.values()
            ])
            dot.node(name, label=f'{{{keys}}}|{{{values}}}')

        # Use a queue to BFS traverse the tree
        q = [(b, 0)]
        while len(q) > 0:
            node = q.pop(0)
            visit_count = self.visit_count[node].item()
            reward = self.reward[node].item()
            value = self.calc_value(node).item() + reward
            _add_node(
                str(node[1]), {
                    'visit': visit_count,
                    'value': value,
                    'reward': reward
                })

            children = self.children_index[node]
            prior = self.prior[node]
            # Only the child nodes whose index is not zero and prior > 0 are valid.
            valid = torch.logical_and(children != 0, prior > 0.0)
            children = children[valid].tolist()
            prior = prior[valid].tolist()
            ucb_score = self.ucb_score[node][valid].tolist()

            for c, u, p in zip(children, ucb_score, prior):
                q.append((b, c))
                dot.edge(
                    str(node[1]),
                    str(c),
                    label=f'<ucb {u:.4f} <br/> pri {p:.4f}>')
        return dot


MCTSState = namedtuple(
    "MCTSState",
    ["steps", "pred_state", "action_sampler_state", "next_predicted_reward"],
    default_value=())
MCTSInfo = namedtuple(
    "MCTSInfo", ["candidate_actions", "value", "candidate_action_policy"])


@alf.configurable
class MCTSAlgorithm(OffPolicyAlgorithm):
    r"""Monte-Carlo Tree Search algorithm.

    The code largely follows the pseudocode of
    `Schrittwieser et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model <https://arxiv.org/abs/1911.08265>`_.
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
       do the search/learn/act. This can be enabled by setting  ``act_with_exploration_policy``
       ``search_with_exploration_policy``, ``learn_with_exploration_policy`` to True.
       See `Grill et al. Monte-Carlo tree search as regularized policy optimization
       <https://arxiv.org/abs/2007.12509>`_ for reference.

    In addition to the original MuZero paper, we also implemented the method
    described in the following two paper:

    1. `Grill et al. Monte-Carlo tree search as regularized policy optimization
    <https://arxiv.org/abs/2007.12509>`_
      It can be enabled by setting (act/learn/search)_with_exploration_policy

    2. `Hubert et. al. Learning and Planning in Complex Action Spaces
    <https://arxiv.org/abs/2104.06303>`_
      It is enabled when SimpleMCTSModel.num_sampled_actions is set.

    The time spent on tree search is directly related to how many times the tree
    is expanded. To make it faster, we also support expanding multiple
    leaves simulaneously. In order to do this, we maintain num_parallel_sims best
    children for each node in the tree and use them to construct k=num_paralilel_sims
    paths. Note the k best children may have duplicates, which is desired because
    we want to expand the most promising path more often. Depending the value of
    ``search_with_exploration_policy``, this process is slightly different:

    * search_with_exploration_policy=True. The k best_children of each node are
      simply chosen by independently sampling the exploration policy k times.
      When constructing the search paths, the i-th search path is based on the
      i-th best child of each node.

    * search_with_exploration_policy=False. The best child is same the case k=1.
      The second best child is found by assuming the visit count of the best
      child and the parent are increased by 1 and applying the UCB criterion
      again. This is repeated k times to get k best children. Note that this is
      different from directly selecting the best k children based on the original
      UCB scores. The reason of not doing that is that if the highest score is
      much bigger than the second highest score, we want to both paths to select
      the same child. During the process of traversing from the root to construct
      k search paths, if several (let's say k') paths are exactly same so far,
      we will use best k' children of the last node of these k' paths to extend
      the paths so that The k' children (may contains duplicates) being selected to
      extend these k' paths are most promising according to the UCB scores.
    """

    def __init__(
            self,
            observation_spec,
            action_spec: alf.BoundedTensorSpec,
            num_simulations: int,
            root_dirichlet_alpha: float,
            root_exploration_fraction: float,
            pb_c_init: float,  # c1 in Appendix B (2)
            pb_c_base: float,  # c2 in Appendix B (2)
            discount: float,
            is_two_player_game: bool,
            visit_softmax_temperature_fn: Callable,
            model: Optional[MCTSModel] = None,
            keep_model_pred_state: bool = False,
            predict_action_sampler=action_samplers.MultinomialSampler(),
            rollout_action_sampler=action_samplers.MultinomialSampler(),
            learn_policy_temperature=1.0,
            reward_spec=TensorSpec(()),
            expand_all_children: bool = False,
            expand_all_root_children: bool = False,
            known_value_bounds=None,
            value_min_max_delta: float = 1e-30,
            ucb_break_tie_eps: float = 0.,
            ucb_parent_visit_count_minus_one: bool = False,
            unexpanded_value_score=0.5,
            act_with_exploration_policy=False,
            search_with_exploration_policy=False,
            learn_with_exploration_policy=False,
            exploration_policy_type: str = 'rkl',
            max_unroll_length=1000000,
            num_parallel_sims=1,
            checkpoint=None,
            debug_summaries=False,
            name="MCTSAlgorithm",
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
                   is the player to take the action. If not provided, steps % 2
                   will be used as to_play.
            action_spec (nested BoundedTensorSpec): representing the actions.
            num_simulations: the number of simulations per search (calls to model)
            root_dirichlet_alpha: alpha of dirichlet prior for exploration
            root_exploration_fraction: noise generated by the dirichlet distribution
                is combined with the action distribution from the model to be used as
                the action prior for the children of the root.
            pb_c_init: c1 of the pUCT rule in Appendix B, equation (2)
            pb_c_base: c2 of the pUCT rule in Appendix B, equation (2)
            discount: reward discount factor
            is_two_player_game (bool): whether this is a two player (zero-sum) game
            model: the model used by the algorithm. If not provided
                in the constructor. It should be specified using `set_model`
                before `predict_step` or `rollout_step` is used.
            keep_model_pred_state: whether to keep ModelOutput.state.pred_state
                returned from model.initial_predict as part of the state of this
                algorithm. If so previous pred_state will be used to call
                initial_predict.
            visit_softmax_temperature_fn (Callable): function for calculating the
                softmax temperature for sampling action based on the visit_counts
                of the children of the root. :math:`P(a) \propto \exp(visit\_count/t)`.
                This function is called as ``visit_softmax_temperature_fn(steps)``,
                where ``steps`` is a vector representing the number of steps in
                the games. And it is expected to return a float vector of the same
                shape as ``steps``.
            predict_action_sampler: available choices include ``CategoricalSeedSampler``,
                ``EpsilonGreedySampler``, ``MultinomialSampler``
            rollout_action_sampler: available choices include ``CategoricalSeedSampler``,
                ``EpsilonGreedySampler``, ``MultinomialSampler``
            learn_policy_temperature (float): transform the policy p found by
                MCTS by :math:`p^{1/learn_policy_temperature} / Z` as policy
                target for model learning, where Z is a normalization factor so
                that the resulting probabilities sum to one.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            expand_all_children (bool): If True, when a new leaf is selected, immediately
                expand all its children. With this option, the visit
                count does not truly reflect the quality of a node. Hence it
                should be used with (act/learn)_with_exploration_policy=True
            expand_all_root_children (bool): whether to expand all root children
                before search. This is described in Appendix A of "Learning and
                Planning in Complex Action Spaces". However, our implementation is
                different from the paper's. The paper initializes Q(s, a) for
                root s for all the action being sampled. We expand all sampled
                action for root s. With this option, the visit
                count does not truly reflect the quality of a node. Hence it
                should be used with (act/learn)_with_exploration_policy=True.
            known_value_bounds (tuple|None): known bound of the values.
            value_min_max_delta: when normalizing the value using the the min and
                max values, ``(max-min).clamp(min=value_min_max_delta)`` is used
                as the denominator.
            ucb_break_tie_eps: add a random number in the range of [0, ucb_break_tie_eps)
                to the UCB score to choose actions with close UCB score randomly.
                It is used only if at least one of ``act/search/learn_with_exploration_policy``
                is False.
            ucb_parent_visit_count_minus_one: This option effectively chooses the
                first child of a parent uniformly, which can increase exploration.
            unexpanded_value_score (float|str): The value score for an unexpanded
                child. If 'max'/'min'/'mean', will use the maximum/minimum/mean
                of the value scores of the expanded siblings. If 'mean_with_parent',
                will use the mean of the value scores of the expanded siblings and
                its parent (this is used in ELF OpenGo and EfficientZero). If 'none', when
                exploration policy is used, will keep the policy for the unexpanded
                children same as prior; when exploration is not used, 'none' behaves
                same as 'min'.
            act_with_exploration_policy (bool): If True, a policy calculated using
                reverse KL divergence will be used for generate action.
            search_with_exploration_policy (bool): If True, a policy calculated
                using reverse KL divergence will be used for tree search.
            learn_with_exploration_policy (bool): If True, a policy calculated
                using reverse KL divergence will be used for learning.
            exploration_policy_type: Type of exploration policy. Must be one of ('rkl', 'kl')
            max_unroll_length (int): maximal allowed unroll steps when building
                the search tree. If ``expand_all_children`` is False, the maximal
                allowed tree depth will be ``max_unroll_length``. Otherwise, the
                maximal allowed tree depth will be ``max_unroll_length-1``
            num_parallel_sims (int): expanding so many leaves at a time for one
                tree. ``num_simulations`` must be divisible by ``num_parallel_sims``.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            name (str): the name of the algorithm.
        """
        assert not nest.is_nested(
            action_spec), "nested action_spec is not supported"
        assert reward_spec.shape == (), "Only scalar reward is supported"
        self._num_simulations = num_simulations
        self._known_value_bounds = known_value_bounds
        self._value_min_max_delta = value_min_max_delta
        self._ucb_break_tie_eps = ucb_break_tie_eps
        self._ucb_parent_visit_count_minus_one = ucb_parent_visit_count_minus_one
        self._model = None
        self._pb_c_base = pb_c_base  # c2 in Appendix B (2)
        self._pb_c_init = pb_c_init  # c1 in Appendix B (2)
        self._discount = discount
        self._is_two_player_game = int(is_two_player_game)
        self._root_dirichlet_alpha = root_dirichlet_alpha
        self._root_exploration_fraction = root_exploration_fraction
        self._visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self._learn_policy_temperature = learn_policy_temperature
        assert exploration_policy_type in ('rkl', 'kl')
        if exploration_policy_type == 'rkl':
            self._calc_exploration_policy = calculate_exploration_policy
        else:
            self._calc_exploration_policy = calculate_kl_exploration_policy
        self._expand_all_children = expand_all_children
        self._expand_all_root_children = expand_all_root_children
        self._num_parallel_sims = num_parallel_sims
        self._parallel = num_parallel_sims > 1
        if expand_all_children:
            self._max_allowed_depth = max_unroll_length - 1
        else:
            self._max_allowed_depth = max_unroll_length
        predict_action_sampler_state_spec = ()
        if isinstance(predict_action_sampler, alf.nn.Network):
            predict_action_sampler_state_spec = predict_action_sampler.state_spec
        rollout_action_sampler_state_spec = ()
        if isinstance(rollout_action_sampler, alf.nn.Network):
            rollout_action_sampler_state_spec = rollout_action_sampler.state_spec

        assert not expand_all_children or num_parallel_sims == 1, (
            "num_parallel_sim > 1 is not supported when expand_all_children=True"
        )

        assert num_simulations % num_parallel_sims == 0, (
            "num_simulations must "
            "be divided by num_parallel_sims. Got %s and %s" %
            (self._num_simulations, num_parallel_sims))

        if (expand_all_children or expand_all_root_children) and not (
                learn_with_exploration_policy and act_with_exploration_policy):
            logging.warning(
                "Consider using (act/learn)_with_exploration_policy"
                "=True for expand_all_children or expand_all_root_children.")

        if isinstance(unexpanded_value_score, str):
            assert unexpanded_value_score in (
                'max', 'min', 'mean', 'mean_with_parent',
                'none'), ("Unsupported unexpanded_value_score=%s" %
                          unexpanded_value_score)
        self._unexpanded_value_score = unexpanded_value_score
        self._act_with_exploration_policy = act_with_exploration_policy
        self._search_with_exploration_policy = search_with_exploration_policy
        self._learn_with_exploration_policy = learn_with_exploration_policy
        pred_state_spec = ()
        if keep_model_pred_state:
            pred_state_spec = model.pred_state_spec
        self._keep_model_pred_state = keep_model_pred_state

        state_spec = MCTSState(
            steps=alf.TensorSpec((), dtype=torch.int64),
            pred_state=pred_state_spec)
        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            predict_state_spec=state_spec._replace(
                action_sampler_state=predict_action_sampler_state_spec),
            rollout_state_spec=state_spec._replace(
                next_predicted_reward=alf.TensorSpec(()),
                action_sampler_state=rollout_action_sampler_state_spec),
            train_state_spec=state_spec._replace(
                action_sampler_state=rollout_action_sampler_state_spec),
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)
        self._model = model
        self._rollout_action_sampler = rollout_action_sampler
        self._predict_action_sampler = predict_action_sampler

    def set_model(self, model: MCTSModel):
        """Set the model used by the algorithm."""
        self._model = model

    @property
    def discount(self):
        return self._discount

    @torch.no_grad()
    def _step(self, time_step: TimeStep, state: MCTSState, action_sampler):
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

        model_output = self._model.initial_predict(time_step.observation,
                                                   state.pred_state)

        if valid_action_mask is not None:
            # mask out invalid actions
            assert model_output.actions == ()
            model_output = model_output._replace(
                action_probs=model_output.action_probs *
                valid_action_mask.to(torch.float32))

        branch_factor = model_output.action_probs.shape[1]
        if self._expand_all_children:
            tree_size = (self._num_simulations + 1) * branch_factor + 1
        elif self._expand_all_root_children:
            tree_size = self._num_simulations + 1 + branch_factor
        else:
            tree_size = self._num_simulations + 1

        trees = _MCTSTrees(
            tree_size,
            model_output,
            self._known_value_bounds,
            self._num_parallel_sims,
            discount=self._discount,
            value_min_max_delta=self._value_min_max_delta)
        if self._is_two_player_game and to_plays is None:
            # We may need the environment to pass to_play and pass to_play to
            # model because players may not always alternate in some game.
            # And for many games, it is not always obvious to determine which
            # player is in the current turn by looking at the board.
            to_plays = steps % 2
        self._max_policy_iterations = 0
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
        trees.value_sum[roots] = model_output.value.to(trees.value_sum.dtype)
        if not self._expand_all_children:
            self._build_tree1(trees, to_plays)
        else:
            self._build_tree2(trees, to_plays)

        action_probs, info = self._select_action(trees, steps)
        pred_state = ()
        if self._keep_model_pred_state:
            pred_state = model_output.state.pred_state
        if isinstance(action_sampler, alf.nn.Network):
            action_id, action_sampler_state = action_sampler(
                action_probs, state.action_sampler_state)
        else:
            action_id, action_sampler_state = action_sampler(action_probs), ()
        if info.candidate_actions != ():
            action = info.candidate_actions[trees.B, action_id]
        else:
            action = action_id

        if common.is_rollout():
            curr_reward = time_step.reward
            curr_predicted_reward = trees.reward[roots]
            with alf.summary.scope(self._name):
                alf.summary.scalar(
                    'reward',
                    time_step.reward.mean(),
                    average_over_summary_interval=True)
                alf.summary.scalar(
                    'reward_next_predicted/value',
                    state.next_predicted_reward.mean(),
                    average_over_summary_interval=True)
                alf.summary.scalar(
                    'reward_predicted/value',
                    curr_predicted_reward.mean(),
                    average_over_summary_interval=True)

                def _summarize_error(name, tgt, pred):
                    error = (pred - tgt).abs()
                    error_z = error[tgt == 0]
                    error_nz = error[tgt != 0]
                    if error_z.numel() > 0:
                        alf.summary.scalar(
                            name + "/error_z",
                            error_z.mean(),
                            average_over_summary_interval=True)
                    else:
                        alf.summary.scalar(
                            name + "/error_z",
                            None,
                            average_over_summary_interval=True)
                    if error_nz.numel() > 0:
                        alf.summary.scalar(
                            name + "/error_nz",
                            error_nz.mean(),
                            average_over_summary_interval=True)
                    else:
                        alf.summary.scalar(
                            name + "/error_nz",
                            None,
                            average_over_summary_interval=True)

                _summarize_error("reward_next_predicted", curr_reward,
                                 state.next_predicted_reward)
                _summarize_error("reward_predicted", curr_reward,
                                 curr_predicted_reward)

            next_predicted_reward = trees.reward[
                trees.B, trees.children_index[roots][trees.B, action_id]]
        else:
            next_predicted_reward = ()

        return AlgStep(
            output=action,
            state=MCTSState(
                steps=state.steps + 1,
                pred_state=pred_state,
                next_predicted_reward=next_predicted_reward,
                action_sampler_state=action_sampler_state),
            info=info)

    def rollout_step(self, time_step: TimeStep, state: MCTSState):
        return self._step(time_step, state, self._rollout_action_sampler)

    def predict_step(self, time_step: TimeStep, state: MCTSState):
        return self._step(time_step, state, self._predict_action_sampler)

    def _get_best_child_index(self, trees, B, nodes, i):
        if self._parallel and not self._search_with_exploration_policy:
            # If several nodes are same, we should select actions from different
            # entries in best_child_index. So we need to find out which entry
            # of best_child_index a node should use. For search_with_exploration_policy,
            # the entries in best_child_index are sampled independently from
            # the calculated policy, so it is ok to use i'th best_child_index
            # for i-th node. But for UCB based selection, we want to use
            # the top entries in best_child_index when ever they are not used.
            #
            # For x>y, same_node[b, x, y] indicates where nodes[b, x] and nodes[b, y] are same
            # For x<=y, same_node[b, x, y] is 0.
            #
            # For example, suppose nodes[0] = [2,3,2,3,3], it means that the
            # current 5 parallel searches on tree 0 are at tree 0's node 2,
            # node 3, node 2, node 3 and node 3 respectively. According to
            # our algorithm, under UCB-based selection strategy, we want the
            # 3 searches at node 3 to go to different "next nodes" (i.e. the
            # top 3 best children of node 3) respectively. The same goes for the 2
            # searches at node 2 (they should go to the top 2 best children
            # of node 2 respectively). Therefore we want to use i to map nodes
            # index [2, 3, 2, 3, 3] to children index [0, 0, 1, 1, 2].
            #
            # With this example, same_node[0] will be:
            # [[0, 0, 0, 0, 0],
            #  [0, 0, 0, 0, 0],
            #  [1, 0, 0, 0, 0],
            #  [0, 1, 0, 0, 0],
            #  [0, 1, 0, 1, 0]]
            # [B, psims, psims]. `None` stands for unsqueezing at that dim.
            same_node = (nodes[:, :, None] == nodes[:, None, :]).tril(
                diagonal=-1)
            # With the above example, i[0] is [0, 0, 1, 1, 2]
            # [B, psims]
            i = (same_node.sum(-1), )
        return trees.best_child_index[(B, nodes) + i]

    def _build_tree1(self, trees, to_plays):
        """Build the tree by evaluating one node a time."""
        roots = (trees.B, trees.root_indices)
        sim0 = 1
        if self._expand_all_root_children:
            v = self._expand_children(trees, roots, 1, to_plays)
            branch_factor = trees.branch_factor
            # Backup the result from the children
            trees.visit_count[roots] += branch_factor
            trees.value_sum[roots] += branch_factor * v
            children = trees.get_children(roots)
            children = torch.broadcast_tensors(*children)
            b = torch.cat([roots[0], children[0].reshape(-1)])
            i = torch.cat([roots[1], children[1].reshape(-1)])
            self._update_best_child(trees, (b, i))
            sim0 += trees.branch_factor
        else:
            self._update_best_child(trees, roots)

        psims = self._num_parallel_sims
        parallel = self._parallel
        B = trees.B
        if parallel:
            B = B.unsqueeze(-1)
            i = (torch.arange(psims), )
        else:
            i = ()
        max_depth = torch.zeros(B.shape[0], dtype=torch.int64)
        avg_depth = 0.
        new_nodes = []
        for sim in range(sim0, self._num_simulations + sim0, psims):
            search_paths, path_lengths, last_to_plays = self._search(
                trees, to_plays)
            depth = path_lengths
            if parallel:
                depth = path_lengths.max(dim=1)[0]
            max_depth = torch.maximum(max_depth, depth)
            avg_depth = avg_depth + path_lengths.to(
                torch.float32).mean() * psims
            # [B] or [B, psims]
            prev_nodes = search_paths[(path_lengths - 2, B) + i]
            model_state = trees.get_model_state((B, prev_nodes))
            # [B] or [B, psims]
            best_child_index = self._get_best_child_index(
                trees, B, prev_nodes, i)
            if trees.action is None:
                action = best_child_index
            else:
                action = trees.action[B, prev_nodes, best_child_index]
            model_output = self._recurrent_inference(model_state, action)
            if parallel:
                sim = torch.arange(sim, sim + psims)
            self._expand_node(
                trees, sim, to_plays=last_to_plays, model_output=model_output)
            # If a child is expanded before, use its existing node
            # [B] or [B, psims]
            child_index = trees.children_index[B, prev_nodes, best_child_index]
            child_index = torch.where((child_index == 0) & (path_lengths > 1),
                                      sim, child_index)
            trees.children_index[B, prev_nodes, best_child_index] = child_index
            # The reason of not using child_index at the rhs of the following
            # assignment is that different search_paths may expand the same node
            # with same action. We want those duplicated expansions point to the
            # same node id.
            child_index = trees.children_index[B, prev_nodes, best_child_index]
            search_paths[(path_lengths - 1, B) + i] = child_index

            new_nodes.append(child_index)

            self._backup(
                trees,
                search_paths=search_paths,
                path_lengths=path_lengths,
                values=model_output.value)

        if self._debug_summaries and alf.summary.should_record_summaries():
            if not self._parallel:
                new_nodes = [x.unsqueeze(-1) for x in new_nodes]
                B = B.unsqueeze(-1)
            new_nodes = torch.cat(new_nodes, dim=1)
            node_i = B * trees.visit_count.shape[1] + new_nodes
            unique_node_i = torch.unique(node_i[:])
            with alf.summary.scope(self._name):
                alf.summary.scalar("avg_tree_size",
                                   unique_node_i.numel() / B.shape[0])

        if self._num_simulations > 0:
            avg_depth = avg_depth / self._num_simulations
        self._summarize(max_depth - 1, avg_depth - 1)

    def _summarize(self, max_unroll_length, avg_unroll_length):
        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("max_policy_iterations",
                                   self._max_policy_iterations)
                alf.summary.scalar("max_unroll_length",
                                   max_unroll_length.max())
                alf.summary.scalar("avg_max_unroll_length",
                                   max_unroll_length.to(torch.float32).mean())
                alf.summary.scalar("avg_unroll_length", avg_unroll_length)

    def _recurrent_inference(self, model_state, action):
        if not self._parallel:
            return self._model.recurrent_inference(model_state, action)

        batch_size, psims = action.shape[:2]

        def _flatten(x):
            return x.reshape(-1, *x.shape[2:])

        def _reshape(x):
            return x.reshape(batch_size, psims, *x.shape[1:])

        model_state = nest.map_structure(_flatten, model_state)
        action = _flatten(action)
        model_output = self._model.recurrent_inference(model_state, action)
        # reshape to [batch_size, self._num_parallel_sims, ...]
        model_output = model_output._replace(
            action_distribution=(), game_over_logit=())
        model_output = nest.map_structure(_reshape, model_output)
        return model_output

    def _build_tree2(self, trees, to_plays):
        """Build the tree by evaluating all the children of one node a time."""
        # expand the root node and backup the values of the children to root.
        branch_factor = trees.branch_factor
        roots = (trees.B, trees.root_indices)
        v = self._expand_children(trees, roots, 1, to_plays)
        trees.visit_count[roots] += branch_factor
        trees.value_sum[roots] += branch_factor * v
        self._update_best_child(trees, roots)

        max_depth = torch.zeros(trees.B.shape[0], dtype=torch.int64)
        avg_depth = 0.
        for sim in range(1, self._num_simulations + 1):
            search_paths, path_lengths, last_to_plays = self._search(
                trees, to_plays)
            max_depth = torch.maximum(max_depth, path_lengths)
            avg_depth = avg_depth + path_lengths.to(torch.float32).mean()
            prev_nodes = search_paths[path_lengths - 1, trees.B]
            if trees.game_over is None:
                values = self._expand_children(trees, (trees.B, prev_nodes),
                                               1 + sim * branch_factor,
                                               last_to_plays)
            else:
                # Do not expand game_over nodes.
                values = torch.zeros(trees.B.shape[0])
                game_over = trees.game_over[trees.B, prev_nodes]
                if not game_over.all():
                    not_over = ~game_over
                    nodes = (trees.B[not_over], prev_nodes[not_over])
                    values[not_over] = self._expand_children(
                        trees, nodes, 1 + sim * branch_factor,
                        last_to_plays[not_over]
                        if self._is_two_player_game else None)
            self._backup(
                trees,
                search_paths=search_paths,
                path_lengths=path_lengths,
                values=values,
                count=branch_factor)

        avg_depth = avg_depth / self._num_simulations
        self._summarize(max_depth, avg_depth)

    def _print_tree(self, trees: _MCTSTrees, b):
        """Helper function to visualize the b-th search tree."""
        nodes = [((b, 0), 0, 0)]
        np.set_printoptions(precision=4)

        def _get_item(nested, i):
            return alf.nest.map_structure(lambda x: x[i], nested)

        def _print_policy(policy):
            if policy.numel() <= 4:
                return policy
            policy = policy.cpu()
            indices = torch.argsort(policy, descending=True)
            s = ' '.join(
                ["%s:%0.3f" % (int(i), float(policy[i])) for i in indices[:4]])
            return '[' + s + ']'

        while len(nodes) > 0:
            node, action, depth = nodes.pop(0)
            print(
                '| ' * depth + "a=%s" % action, "node=%-3d" % node[1],
                "state=%-4d" %
                (hash(str(_get_item(trees.model_state, node))) % 10000), '--' *
                (self._max_allowed_depth - depth) + "original_value=%-7.4f" %
                trees.original_value[node].cpu().numpy(),
                "v=%-7.4f" % trees.calc_value(node).cpu().numpy(),
                "r=%-7.4f" % trees.reward[node].cpu().numpy(),
                "prior=%s" % _print_policy(trees.prior[node]),
                "policy=%s" % _print_policy(
                    self._calculate_policy(trees,
                                           (torch.tensor([node[0]]),
                                            torch.tensor([node[1]])))[0]),
                "visit_count=%s" % trees.visit_count[node].item())
            children = list(trees.children_index[node])
            prior = list(trees.prior[node])
            new_nodes = [((b, int(c)), a, depth + 1)
                         for a, (c, p) in enumerate(zip(children, prior))
                         if c != 0 and p != 0]
            nodes = new_nodes + nodes

    def _search(self, trees, to_plays):
        """
        Returns:
            tuple:
            - search_paths: [T, B] or [T, B, psims] int64 Tensor, where T is the max length of the
                search paths. For paths whose length is shorter than T, search_path
                is padded with the last node index of that path.
            - path_lengths: [B] or [B, psims] Tensor, length of each search path
            - last_to_plays: [B] or [B, psims], to_play for the last node of each path
        """
        children_index = trees.children_index
        game_over = trees.game_over
        psims = self._num_parallel_sims
        search_paths = []
        B = trees.B
        nodes = trees.root_indices
        i = ()
        if self._parallel:
            # If the p-th search path of the b-th tree is at node n, it will be
            # extended by the child indicated by best_child_index[b, n, i[p]].
            if self._search_with_exploration_policy:
                i = (torch.arange(psims), )
            B = B.unsqueeze(-1)
            nodes = trees.root_indices.unsqueeze(-1).expand(-1, psims)
            if self._is_two_player_game:
                to_plays = to_plays.unsqueeze(-1).expand_as(nodes)
        # [B] or [B, psims]
        path_lengths = torch.ones_like(nodes)
        search_paths = [nodes]
        if game_over is not None:
            done = game_over[B, nodes]
        else:
            done = torch.zeros_like(nodes, dtype=torch.bool)
        depth = 0
        while not torch.all(done) and depth < self._max_allowed_depth:
            depth += 1
            best_child_index = self._get_best_child_index(trees, B, nodes, i)
            # [B] or [B, psims]
            nodes = torch.where(done, nodes,
                                children_index[B, nodes, best_child_index])
            path_lengths[~done] += 1
            if self._is_two_player_game:
                to_plays = torch.where(done, to_plays,
                                       self._is_two_player_game - to_plays)
            if self._expand_all_children:
                # no children means node is not expanded
                done = (trees.children_index[B, nodes] == 0).all(dim=-1)
            else:
                # nodes == 0 means unexpanded node
                done = nodes == 0
            if game_over is not None:
                done = game_over[B, nodes] | done
            search_paths.append(nodes)

        search_paths = torch.stack(search_paths)
        return search_paths, path_lengths, to_plays

    def _select_action(self, trees, steps):
        roots = (trees.B, trees.root_indices)

        if self._act_with_exploration_policy or self._learn_with_exploration_policy:
            policy = self._calculate_policy(trees, roots)

        if not self._act_with_exploration_policy or not self._learn_with_exploration_policy:
            children = trees.get_children(roots)
            visit_counts = trees.visit_count[children]
            visit_counts[children[1] == 0] = 0

        if self._act_with_exploration_policy:
            probs = policy
        else:
            probs = visit_counts
        t = self._visit_softmax_temperature_fn(steps).unsqueeze(-1)
        probs = F.softmax((probs + 1e-30).log() / t, dim=1)
        probs = probs * (trees.prior[roots] > 0).to(torch.float32)

        if trees.action is not None:
            candidate_actions = trees.action[roots]
        else:
            candidate_actions = ()

        if not self._learn_with_exploration_policy:
            parent_visit_count = (trees.visit_count[roots] - 1.0).unsqueeze(-1)
            policy = visit_counts / parent_visit_count

        if self._learn_policy_temperature != 1.0:
            policy = F.softmax(
                (policy + 1e-30).log() / self._learn_policy_temperature, dim=1)

        info = MCTSInfo(
            candidate_actions=candidate_actions,
            value=trees.calc_value(roots),
            candidate_action_policy=policy,
        )

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                prior = trees.prior[roots]
                prior = prior + 1e-30
                policy = policy + 1e-30
                log_ratio = (prior / policy).log()
                rkld = (prior * log_ratio).sum(-1)
                kld = -(policy * log_ratio).sum(-1)
                policy_entropy = -(policy * policy.log()).sum(-1)
                prior_entropy = -(prior * prior.log()).sum(-1)
                summary_utils.add_mean_hist_summary("rkld", rkld)
                summary_utils.add_mean_hist_summary("kld", kld)
                summary_utils.add_mean_hist_summary("policy_entropy",
                                                    policy_entropy)
                summary_utils.add_mean_hist_summary("prior_entropy",
                                                    prior_entropy)
                summary_utils.add_mean_hist_summary("value", info.value)
                for i in range(policy.shape[1]):
                    summary_utils.add_mean_hist_summary(
                        'policy%s' % i, policy[:, i])

        return probs, info

    def _update_best_child(self, trees: _MCTSTrees, parents):
        if self._search_with_exploration_policy:
            self._sample_child(trees, parents)
        elif not self._parallel:
            self._ucb_child(trees, parents)
        else:
            self._ucb_child_parallel(trees, parents)

    def _ucb_child(self, trees: _MCTSTrees, parents):
        """Get child using UCB score."""
        ucb_scores = self._ucb_score(trees, parents)
        trees.ucb_score[parents] = ucb_scores

        best_child_index = ucb_scores.argmax(dim=1)
        trees.best_child_index[parents] = best_child_index

    def _ucb_child_parallel(self, trees: _MCTSTrees, parents):
        """Get child using UCB score."""
        prior = trees.prior[parents]
        value_score, child_visit_count = self._value_score(trees, parents)
        value_score[prior == 0] = -MAXIMUM_FLOAT_VALUE
        parent_visit_count = trees.visit_count[parents].unsqueeze(-1)
        ucb_scores = self._ucb_score1(parent_visit_count, child_visit_count,
                                      prior, value_score)
        trees.ucb_score[parents] = ucb_scores
        best_child_index = ucb_scores.argmax(dim=1)
        trees.best_child_index[parents + (0, )] = best_child_index

        n = torch.arange(child_visit_count.shape[0])
        for i in range(1, self._num_parallel_sims):
            child_visit_count[n, best_child_index] += 1
            parent_visit_count += 1
            ucb_scores = self._ucb_score1(
                parent_visit_count, child_visit_count, prior, value_score)
            best_child_index = ucb_scores.argmax(dim=1)
            trees.best_child_index[parents + (i, )] = best_child_index

    def _ucb_score(self, trees, parents):
        prior = trees.prior[parents]
        value_score, child_visit_count = self._value_score(trees, parents)
        value_score[prior == 0] = -MAXIMUM_FLOAT_VALUE
        parent_visit_count = trees.visit_count[parents].unsqueeze(-1)
        return self._ucb_score1(parent_visit_count, child_visit_count, prior,
                                value_score)

    def _ucb_score1(self, parent_visit_count, child_visit_count, prior,
                    value_score):
        if self._ucb_parent_visit_count_minus_one:
            parent_visit_count = parent_visit_count - 1
        pb_c = torch.log1p(
            (parent_visit_count + 1) / self._pb_c_base) + self._pb_c_init
        pb_c = pb_c * torch.sqrt(parent_visit_count.to(
            torch.float32)) / (child_visit_count + 1.)
        prior_score = pb_c * prior
        if self._ucb_break_tie_eps > 0:
            prior_score += self._ucb_break_tie_eps * torch.rand_like(
                prior_score)
        return prior_score + value_score

    def _value_score(self, trees: _MCTSTrees, parents):
        children = trees.get_children(parents)
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
        elif self._unexpanded_value_score == 'mean_with_parent':
            pvalue_score = trees.calc_value(parents)
            pvalue_score = trees.normalize_value(pvalue_score, parents[0])
            value_score[unexpanded] = 0.
            n = (~unexpanded).sum(dim=1, keepdim=True)
            mean_value_score = (pvalue_score.unsqueeze(1) + value_score.sum(
                dim=1, keepdim=True)) / (n + 1)
            value_score = torch.where(unexpanded, mean_value_score,
                                      value_score)

        return value_score, child_visit_count

    def _calculate_policy(self, trees: _MCTSTrees, parents):
        parent_visit_count = trees.visit_count[parents].unsqueeze(-1)
        if self._ucb_parent_visit_count_minus_one:
            parent_visit_count = parent_visit_count - 1
        c = torch.log1p(
            (parent_visit_count + 1) / self._pb_c_base) + self._pb_c_init
        c = c / parent_visit_count.to(torch.float32).sqrt()
        prior = trees.prior[parents]
        value_score, child_visit_count = self._value_score(trees, parents)
        if self._unexpanded_value_score != 'none':
            policy, iterations = self._calc_exploration_policy(
                value_score, prior, c)
        else:
            # For 'none', we keep the policy for the unexpanded children same
            # as prior.
            expanded = child_visit_count != 0
            expanded_prior = prior * (expanded.to(torch.float32) + 1e-30)
            sum_expanded = expanded_prior.sum(dim=1, keepdim=True)
            expanded_prior = expanded_prior / sum_expanded
            policy, iterations = self._calc_exploration_policy(
                value_score, expanded_prior, c)
            policy = torch.where(expanded, policy * sum_expanded, prior)
        if self._ucb_parent_visit_count_minus_one:
            # "parent_visit_count == 0" will get c=inf and policy will be equal to
            # prior. But according to ucb calculation (_ucb_score1()), we should
            # use uniform distribution in this case.
            uniform = (parent_visit_count == 0).unsqueeze(1)
            policy[uniform] = 1 / policy.shape[1]

        self._max_policy_iterations = max(self._max_policy_iterations,
                                          iterations)
        return policy

    def _sample_child(self, trees: _MCTSTrees, parents):
        """Get child by sampling from exploration policy."""
        if torch.are_deterministic_algorithms_enabled and self._parallel:
            # when using parallel expansions, there can be duplicated nodes in
            # parents and will cause the assignment at the end of the function
            # non-deterministic.
            num_expansions = trees.visit_count.shape[1]
            nodes_i = parents[0] * num_expansions + parents[1]
            nodes_i = nodes_i.unique()
            parents = (nodes_i // num_expansions, nodes_i % num_expansions)

        policy = self._calculate_policy(trees, parents)
        # We need replacement so that the promising children are searched more
        # often and value estimation is unbiased.
        action = torch.multinomial(
            policy, num_samples=self._num_parallel_sims, replacement=True)
        if not self._parallel:
            action = action.squeeze(1)
        trees.best_child_index[parents] = action

    def _backup(self,
                trees: _MCTSTrees,
                search_paths,
                path_lengths,
                values,
                count=1):
        r"""
        Notation:

            vsum: trees.value_sum
            vc: trees.visit_count
            v: value
            T: path_length
            c: count
            t: 0 <= t <= T-1

        The following shows the computation for each search path:

            G_t = \sum_{s=t+1}^{T-1} \gamma^{s-t-1} r_s + gamma^{T-t-1} v
            vsum_t += c G_t
            vc_t += c

        Args:
            search_paths: [T, B] or [T, B, psims]
            path_length: [B] or [B, psims]
            values: [B] or [B, psims]
        """
        psims = self._num_parallel_sims
        B = trees.B.unsqueeze(0)
        T = search_paths.shape[0]
        depth = torch.arange(T).unsqueeze(-1)
        parallel = self._parallel
        if parallel:
            B = B.unsqueeze(-1)
            depth = depth.unsqueeze(-1)
            i = (torch.arange(psims), )
        else:
            i = ()

        if trees.reward is not None:
            reward = trees.reward[B, search_paths]
            reward[depth > path_lengths] = 0.
            # [T+1, B] or [T+1, B, psims]
            reward = tensor_utils.tensor_extend_zero(reward)
            reward[(path_lengths, B[0]) + i] = values
            # [T+1, 1]
            discounts = (self._discount**torch.arange(
                T + 1, dtype=torch.float32)).unsqueeze(-1)
            if parallel:
                # [T+1, 1, 1]
                discounts = discounts.unsqueeze(-1)

            # discounted_return[t] = discount^t * reward[t]
            discounted_return = reward * discounts
            # discounted_return[t] = \sum_{s=t}^T discount^s * reward[s]
            discounted_return = discounted_return.flip(0).cumsum(dim=0).flip(0)

            # discounted_return[t] = \sum_{s=t}^T discount^(s-t) * reward[s]
            discounted_return = discounted_return / discounts
            discounted_return = discounted_return[1:]
        else:
            # [T, 1]
            steps = torch.arange(1, T + 1, dtype=torch.float32).unsqueeze(-1)
            if parallel:
                # [T, 1, psims]
                steps = steps.unsqueeze(-1)
            # [T, B] or [T, B, psims]
            discounts = self._discount**(path_lengths.unsqueeze(0) - steps)
            discounted_return = values.unsqueeze(0) * discounts

        # [T, B] or [T, B, psims]
        valid = depth < path_lengths
        value_sum_delta = count * discounted_return[valid]
        nodes = (B.expand_as(search_paths)[valid], search_paths[valid])
        if not parallel:
            trees.visit_count[nodes] += count
            trees.value_sum[nodes] += value_sum_delta
        else:
            if not torch.are_deterministic_algorithms_enabled:
                # This branch is not deterministic because of the possible duplications
                # in nodes_i.
                nodes_i = nodes[0] * trees.visit_count.shape[1] + nodes[1]
                trees.visit_count.view(-1).scatter_add_(
                    0, nodes_i,
                    torch.tensor(
                        [count],
                        dtype=trees.visit_count.dtype).expand_as(nodes_i))
                trees.value_sum.view(-1).scatter_add_(0, nodes_i,
                                                      value_sum_delta)
            else:
                depth = depth.squeeze(-1)
                TB = trees.B.unsqueeze(0).expand(T, -1)
                for i in range(psims):
                    # [T, B]
                    vld = valid[..., i]
                    nd = (TB[vld], search_paths[..., i][vld])
                    trees.visit_count[nd] += count
                    trees.value_sum[
                        nd] += count * discounted_return[..., i][vld]

        trees.update_value_stats((B, search_paths), valid)
        self._update_best_child(trees, nodes)

    def _expand_children(self, trees: _MCTSTrees, nodes, n, to_plays):
        """Expand all children of nodes.

        Args:
            nodes: tuple(trees.B, node_index). Nodes to be expanded.
            n (int): n-th expansion, zero-based
            to_plays: bool[batch_size], to_plays for nodes
        Returns:
            Tensor with shape of nodes. The weighted (prior as weight) average
            of the bootstrapped return from the children.
        """
        batch_size = nodes[0].shape[0]
        branch_factor = trees.branch_factor
        model_state = trees.get_model_state(nodes)

        def _repeat(x):
            return x.repeat_interleave(trees.branch_factor, dim=0)

        model_state = nest.map_structure(_repeat, model_state)
        if trees.action is None:
            action = torch.arange(branch_factor)[None, :].expand(
                batch_size, -1)
        else:
            action = trees.action[nodes]
        action = action.reshape(-1, *action.shape[2:])
        model_output = self._model.recurrent_inference(model_state, action)

        def _reshape(x):
            return x.reshape(batch_size, branch_factor, *x.shape[1:])

        # reshape to [batch_size, branch_factor, ...]
        model_output = model_output._replace(
            action_distribution=(), game_over_logit=())
        model_output = nest.map_structure(_reshape, model_output)
        # If a node is expanded before, use its existing children
        children_index0 = trees.children_index[nodes]
        children_index = torch.where(children_index0 == 0,
                                     torch.arange(n, n + branch_factor),
                                     children_index0)
        trees.children_index[nodes] = children_index
        children = (nodes[0].unsqueeze(-1), children_index)
        trees.value_sum[children] = model_output.value.to(
            trees.value_sum.dtype)
        trees.original_value[children] = model_output.value
        trees.visit_count[children] += 1
        trees.update_value_stats((children[0].t(), children[1].t()))
        if self._is_two_player_game:
            assert to_plays.shape == nodes[0].shape
            trees.to_play[children] = 1 - to_plays.unsqueeze(-1)
        if trees.game_over is not None:
            trees.game_over[children] = model_output.game_over

        def _set_tree_state(ts, s):
            ts[children] = s

        nest.map_structure(_set_tree_state, trees.model_state,
                           model_output.state)
        if trees.reward is not None:
            trees.reward[children] = model_output.reward
        if trees.action is not None:
            trees.action[children] = model_output.actions
        trees.prior[children] = model_output.action_probs
        value = self._discount * model_output.value
        if trees.reward is not None:
            value += model_output.reward
        value = (value * trees.prior[nodes]).sum(dim=-1)
        return value

    def _expand_node(
            self,
            trees: _MCTSTrees,
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
            # model_output.reward.dtype may be different from initial_inference
            trees.reward[:, n] = model_output.reward.to(trees.reward.dtype)
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
        trees.original_value[:, n] = model_output.value


def calculate_exploration_policy(value, prior, c, tol=1e-6):
    r"""Calculate exploration policy.

    The policy is based on
    `Grill et al. Monte-Carlo tree search as regularized policy optimization
    <https://arxiv.org/abs/2007.12509>`_

    Notation:

        q: prior policy

        p: sampling probability

        v: value

    The exploration policy is found by minimizing the following:

    .. math::

        p = \arg\min_p \left[ -E_p(v) + c KL(q\|p) \right]

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
    starting the iteration with a :math:`\beta` s.t. :math:`f(\beta) > 0` guarantees
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

    converged = False
    i = 0
    while i < 100:
        i += 1
        beta_u = beta - u
        p = prior / beta_u
        sum_p = p.sum(dim=1, keepdim=True)
        diff = sum_p - 1
        if (diff < tol).all():
            converged = True
            break
        d = (p / beta_u).sum(dim=1, keepdim=True)
        beta = beta + diff / d

    if not converged:
        bad = ~(diff < tol).squeeze(1)
        raise RuntimeError(
            "calculate_exploration_policy() cannot converge. "
            "value=%s prior=%s c=%s" % (value[bad], prior[bad], c[bad]))

    return p, i


def calculate_kl_exploration_policy(value, prior, c):
    r"""Calculate exploration policy.

    This is similar to ``calculate_exploration_policy``, but using :math:`KL(p\|q)`
    instead of :math:`KL(q\|p)` for regularization.

    Notation:

        q: prior policy

        p: sampling probability

        v: value

    The exploration policy is found by minimizing the following:

    .. math::

        p = \arg\min_p \left[ -E_p(v) + c KL(p\|q) \right]

    which leads to the following solution:

    .. math::

        p_i = \frac{q_i \exp(v_i/c)}{Z}

    where :math:`Z` is such that :math:`\sum_i p_i = 1`

    Args:
        value (Tensor): [N, K] Tensor
        prior (Tensor): [N, K] Tensor
        c (Tensor): [N, 1] Tensor
    Returns:
        tuple:
        - Tensor: [N, K], the exploration policy
        - int: always 0 (to conform with the signature of calculate_exploration_policy)
    """
    batch_size = value.shape[0]
    assert value.shape == prior.shape
    assert c.shape == (batch_size, 1)
    p = prior * (value / c).exp()
    p = p / p.sum(dim=-1, keepdim=True)

    return p, 0


@alf.configurable
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


@alf.configurable
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


@alf.configurable
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
        # translates to about exp(1e-3*1e10) probability ratio.
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


@alf.configurable
def create_go_mcts(observation_spec, action_spec, debug_summaries):
    return create_board_game_mcts(
        observation_spec,
        action_spec,
        dirichlet_alpha=0.03,
        debug_summaries=debug_summaries)


@alf.configurable
def create_chess_mcts(observation_spec, action_spec, debug_summaries):
    return create_board_game_mcts(
        observation_spec,
        action_spec,
        dirichlet_alpha=0.3,
        debug_summaries=debug_summaries)


@alf.configurable
def create_shogi_mcts(observation_spec, action_spec, debug_summaries):
    return create_board_game_mcts(
        observation_spec,
        action_spec,
        dirichlet_alpha=0.15,
        debug_summaries=debug_summaries)


@alf.configurable
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


@alf.repr_wrapper
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
