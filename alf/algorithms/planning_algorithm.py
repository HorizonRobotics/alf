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

from collections import namedtuple
import gin

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import (AlgStep, Experience, LossInfo, namedtuple,
                                 TimeStep)
from alf.nest import nest
from alf.optimizers.traj_optimizers import RandomOptimizer

PlannerState = namedtuple("PlannerState", ["policy"], default_value=())
PlannerInfo = namedtuple("PlannerInfo", ["policy"])
PlannerLossInfo = namedtuple('PlannerLossInfo', ["policy"])


@gin.configurable
class PlanAlgorithm(OffPolicyAlgorithm):
    """Planning Module

    This module plans for actions based on initial observation
    and specified reward and dynamics functions
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 train_state_spec,
                 planning_horizon=25,
                 upper_bound=None,
                 lower_bound=None,
                 name="PlanningAlgorithm"):
        """Create a PlanningAlgorithm.

        Args:
            planning_horizon (int): planning horizon in terms of time steps
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
        """
        super().__init__(
            feature_spec,
            action_spec,
            train_state_spec=train_state_spec,
            name=name)

        flat_action_spec = nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, "doesn't support nested action_spec"

        flat_feature_spec = nest.flatten(feature_spec)
        assert len(
            flat_feature_spec) == 1, "doesn't support nested feature_spec"

        action_spec = flat_action_spec[0]

        assert action_spec.is_continuous, "only support \
                                                    continious control"

        self._num_actions = action_spec.shape[-1]

        self._action_spec = action_spec
        self._feature_spec = feature_spec
        self._planning_horizon = planning_horizon
        self._upper_bound = action_spec.maximum if upper_bound is None \
                                                else upper_bound
        self._lower_bound = action_spec.minimum if lower_bound is None \
                                                else lower_bound

        self._reward_func = None
        self._dynamics_func = None
        self._step_eval_func = None  # per step evaluation function

    def train_step(self, time_step: TimeStep, state):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            TrainStep:
                output: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        pass

    def set_reward_func(self, reward_func):
        """Set per-time-step reward function used for planning
        Args:
            reward_func (Callable): the reward function to be used for planning.
            reward_func takes (obs, action) as input
        """
        self._reward_func = reward_func

    def set_dynamics_func(self, dynamics_func):
        """Set the dynamics function for planning
        Args:
            dynamics_func (Callable): reward function to be used for planning.
            dynamics_func takes (time_step, state) as input and returns
            next_time_step (TimeStep) and the next_state
        """
        self._dynamics_func = dynamics_func

    def generate_plan(self, time_step: TimeStep, state, epsilon_greedy):
        """Compute the plan based on the provided observation and action
        Args:
            time_step (TimeStep): input data for next step prediction
            state: input state next step prediction
        Returns:
            action: planned action for the given inputs
            state: mbrl state
        """
        pass


@gin.configurable
class RandomShootingAlgorithm(PlanAlgorithm):
    """Random Shooting-based planning method.
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 population_size,
                 planning_horizon,
                 upper_bound=None,
                 lower_bound=None,
                 hidden_size=256,
                 name="RandomShootingAlgorithm"):
        """Create a RandomShootingAlgorithm.

        Args:
            population_size (int): the size of polulation for random shooting
            planning_horizon (int): planning horizon in terms of time steps
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
            hidden_size (int|tuple): size of hidden layer(s)
        """
        super().__init__(
            feature_spec=feature_spec,
            action_spec=action_spec,
            train_state_spec=(),
            planning_horizon=planning_horizon,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            name=name)

        flat_action_spec = nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, ("RandomShootingAlgorithm doesn't "
                                            "support nested action_spec")

        flat_feature_spec = nest.flatten(feature_spec)
        assert len(flat_feature_spec) == 1, ("RandomShootingAlgorithm doesn't "
                                             "support nested feature_spec")

        self._population_size = population_size
        solution_size = self._planning_horizon * self._num_actions
        self._plan_optimizer = RandomOptimizer(
            solution_size,
            self._population_size,
            upper_bound=action_spec.maximum,
            lower_bound=action_spec.minimum)

    def train_step(self, time_step: TimeStep, state):
        """
        Args:
            time_step (TimeStep): input data for planning
            state: state for planning (previous observation)
        Returns:
            TrainStep:
                output: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        return AlgStep(output=(), state=(), info=())

    def generate_plan(self, time_step: TimeStep, state, epsilon_greedy):
        assert self._reward_func is not None, ("specify reward function "
                                               "before planning")

        assert self._dynamics_func is not None, ("specify dynamics function "
                                                 "before planning")

        self._plan_optimizer.set_cost(self._calc_cost_for_action_sequence)
        opt_action = self._plan_optimizer.obtain_solution(time_step, state)
        action = opt_action[:, 0]
        action = torch.reshape(action, [time_step.observation.shape[0], -1])
        return action, state

    def _expand_to_population(self, data):
        """Expand the input tensor to a population of replications
        Args:
            data (Tensor): input data with shape [batch_size, ...]
        Returns:
            data_population (Tensor) with shape
                                    [batch_size * self._population_size, ...].
            For example data tensor [[a, b], [c, d]] and a population_size of 2,
            we have the following data_population tensor as output
                                    [[a, b], [a, b], [c, d], [c, d]]
        """
        data_population = torch.repeat_interleave(
            data, self._population_size, dim=0)
        return data_population

    def _calc_cost_for_action_sequence(self, time_step: TimeStep, state,
                                       ac_seqs):
        """
        Args:
            time_step (TimeStep): input data for next step prediction
            state (MbrlState): input state for next step prediction
            ac_seqs: action_sequence (Tensor) of shape [batch_size,
                    population_size, solution_dim]), where
                    solution_dim = planning_horizon * num_actions
        Returns:
            cost (tf.Tensor) with shape [batch_size, population_size]
        """
        obs = time_step.observation
        batch_size = obs.shape[0]

        ac_seqs = torch.reshape(
            ac_seqs,
            [batch_size, self._population_size, self._planning_horizon, -1])

        ac_seqs = ac_seqs.permute(2, 0, 1, 3)
        ac_seqs = torch.reshape(
            ac_seqs, (self._planning_horizon, -1, self._num_actions))

        state = state._replace(dynamics=state.dynamics._replace(feature=obs))
        init_obs = self._expand_to_population(obs)
        state = nest.map_structure(self._expand_to_population, state)

        obs = init_obs
        cost = 0
        for i in range(ac_seqs.shape[0]):
            action = ac_seqs[i]
            time_step = time_step._replace(prev_action=action)
            time_step, state = self._dynamics_func(time_step, state)
            next_obs = time_step.observation
            reward_step = self._reward_func(next_obs, action)
            cost = cost - reward_step
            obs = next_obs

        # reshape cost back to [batch size, population_size]
        cost = torch.reshape(cost, [batch_size, -1])
        return cost

    def after_update(self, training_info):
        pass
