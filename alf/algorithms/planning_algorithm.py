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
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import (AlgStep, Experience, LossInfo, namedtuple,
                                 TimeStep, StepType)
from alf.nest import nest
from alf.optimizers.traj_optimizers import RandomOptimizer, CEMOptimizer
from alf.utils import common

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

    def train_step(self, time_step: TimeStep, state: PlannerState):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            AlgStep:
                output: empty tuple ()
                state (PlannerState): state for training
                info (PlannerInfo):
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
            state (MbrlState): input state for next step prediction. It will
                be used by the next step prediction function set by
                ```set_dynamics_func```.
        Returns:
            action: planned action for the given inputs
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
                 name="RandomShootingAlgorithm"):
        """Create a RandomShootingAlgorithm.

        Args:
            population_size (int): the size of polulation for random shooting
            planning_horizon (int): planning horizon in terms of time steps
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
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
        self._solution_size = solution_size

        assert (() == action_spec.maximum.shape) and \
                (() == action_spec.minimum.shape), \
                    "Only support scalar action maximum and minimum bound"

        self._plan_optimizer = RandomOptimizer(
            solution_size,
            self._population_size,
            upper_bound=np.asscalar(action_spec.maximum),
            lower_bound=np.asscalar(action_spec.minimum))

    def train_step(self, time_step: TimeStep, state):
        """
        Args:
            time_step (TimeStep): input data for planning
            state: state for planning (previous observation)
        Returns:
            AlgStep:
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
        return action

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
            cost (Tensor) with shape [batch_size, population_size]
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
            # Note: currently using (next_obs, action), might need to
            # consider (obs, action) in order to be more compatible
            # with the conventional definition of the reward function
            reward_step, state = self._reward_func(next_obs, action, state)
            cost = cost - reward_step
            obs = next_obs

        # reshape cost back to [batch size, population_size]
        cost = torch.reshape(cost, [batch_size, -1])
        return cost

    def after_update(self, training_info):
        pass


@gin.configurable
class CEMPlanAlgorithm(RandomShootingAlgorithm):
    """CEM-based planning method.
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 population_size,
                 planning_horizon,
                 elite_size=50,
                 max_iter_num=5,
                 epsilon=0.01,
                 tau=0.9,
                 upper_bound=None,
                 lower_bound=None,
                 name="CEMPlanAlgorithm"):
        """Create a CEMPlanAlgorithm.

        Args:
            population_size (int): the size of polulation for optimization
            planning_horizon (int): planning horizon in terms of time steps
            elite_size (int): the number of elites selected in each round
            max_iter_num (int|Tensor): the maximum number of CEM iterations
            epsilon (float): a minimum variance threshold. If the variance of
                the population falls below it, the CEM iteration will stop.
            tau (float): a value in (0, 1) for softly updating the population
                mean and variance:
                    mean = (1 - tau) * mean + tau * new_mean
                    var = (1 - tau) * var + tau * new_var
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
        """
        super().__init__(
            feature_spec=feature_spec,
            action_spec=action_spec,
            planning_horizon=planning_horizon,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            name=name)

        solution_size = planning_horizon * self._num_actions

        assert (() == action_spec.maximum.shape) and \
                (() == action_spec.minimum.shape), \
                    "Only support scalar action maximum and minimum bound"

        self._plan_optimizer = CEMOptimizer(
            solution_size,
            self._population_size,
            upper_bound=np.asscalar(action_spec.maximum),
            lower_bound=np.asscalar(action_spec.minimum),
            elite_size=elite_size,
            max_iter_num=max_iter_num,
            epsilon=epsilon,
            tau=tau)

        self._default_solution = None
        self._prev_solution = None

    def generate_plan(self, time_step: TimeStep, state, epsilon_greedy):
        assert self._reward_func is not None, ("specify reward function "
                                               "before planning")

        assert self._dynamics_func is not None, ("specify dynamics function "
                                                 "before planning")

        # init prev_solution and reset if necessary
        batch_size = time_step.observation.shape[0]
        if self._prev_solution is None:
            self._prev_solution = torch.zeros(batch_size, self._solution_size)
            self._default_solution = self._prev_solution.clone()
        else:
            self._prev_solution = common.reset_state_if_necessary(
                self._prev_solution, self._default_solution,
                time_step.step_type == StepType.FIRST)

        self._plan_optimizer.set_cost(self._calc_cost_for_action_sequence)

        init_mean = self._prev_solution.unsqueeze(1)
        opt_action = self._plan_optimizer.obtain_solution(
            time_step,
            state,
            init_mean=init_mean,
            init_var=torch.ones_like(init_mean) * 1)

        # [B, pop_size=1, horizon * action_dim] -> [B, horizon, action_dim]
        opt_action = torch.reshape(
            opt_action,
            [opt_action.shape[0], self._planning_horizon, self._num_actions])

        # [B, horizon, action_dim]
        self._prev_solution = torch.cat(
            (opt_action[:, 1:], torch.zeros_like(opt_action[:, 0:1])), 1)
        # [B, horizon, action_dim] -> [B, horizon*action_dim]
        self._prev_solution = self._prev_solution.reshape(
            self._prev_solution.shape[0], -1)

        action = opt_action[:, 0]
        return action
