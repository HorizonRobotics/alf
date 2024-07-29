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

import torch
from typing import Callable

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import AlgStep, namedtuple, TimeStep, StepType
from alf.nest import nest
from alf.optimizers.traj_optimizers import RandomOptimizer, CEMOptimizer
from alf.tensor_specs import TensorSpec

PlannerState = namedtuple("PlannerState", ["prev_plan"], default_value=())
PlannerInfo = namedtuple("PlannerInfo", ["planner"])


@alf.configurable
class PlanAlgorithm(OffPolicyAlgorithm):
    """Planning Module

    This module plans for actions based on initial observation
    and specified reward and dynamics functions
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 reward_spec=TensorSpec(()),
                 planning_horizon=25,
                 upper_bound=None,
                 lower_bound=None,
                 name="PlanningAlgorithm"):
        """Create a PlanningAlgorithm.

        Args:
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            planning_horizon (int): planning horizon in terms of time steps
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
            particles_per_replica (int): number of particles used for each replica
        """
        super().__init__(
            feature_spec,
            action_spec,
            reward_spec=reward_spec,
            train_state_spec=PlannerState(
                prev_plan=TensorSpec((planning_horizon,
                                      action_spec.shape[-1]))),
            name=name)

        flat_action_spec = nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, "doesn't support nested action_spec"

        action_spec = flat_action_spec[0]

        assert action_spec.is_continuous, "only support \
                                                    continuous control"

        self._num_actions = action_spec.shape[-1]

        self._action_spec = action_spec
        self._feature_spec = feature_spec
        self._planning_horizon = planning_horizon

        self._upper_bound = torch.Tensor(action_spec.maximum) \
                        if upper_bound is None else upper_bound
        self._lower_bound = torch.Tensor(action_spec.minimum) \
                        if lower_bound is None else lower_bound

        self._action_seq_cost_func = None

    def train_step(self,
                   time_step: TimeStep,
                   state: PlannerState,
                   rollout_info=None):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (PlannerState): input planner state
        Returns:
            AlgStep:
                output: empty tuple ()
                state (PlannerState): updated planner state
                info (PlannerInfo):
        """
        pass

    def set_action_sequence_cost_func(self, action_seq_cost_func):
        """Set a function for evaluating the action sequences for planning
        Args:
            action_seq_cost_func (Callable): cost function to be used for planning.
            action_seq_cost_func takes initial observation and action sequences
            of the shape [B, population, unroll_steps, action_dim] as input
            and returns the accumulated cost along the unrolled trajectory, with
            the shape of [B, population]
        """
        self._action_seq_cost_func = action_seq_cost_func

    def predict_plan(self, time_step: TimeStep, state: PlannerState,
                     epsilon_greedy):
        """Compute the plan based on the provided observation and action
        Args:
            time_step (TimeStep): input data for next step prediction
            state (PlannerState): input planner state
        Returns:
            action: planned action for the given inputs
        """
        pass


@alf.configurable
class RandomShootingAlgorithm(PlanAlgorithm):
    """Random Shooting-based planning method.

    This method uses a Random Shooting approach to optimize an action
    trajectory by minimizing a given cost function. The optimized action
    trajectory is termed as a 'plan' which can be used by other components
    such as a MPC-based controller. It has been used in `Neural Network Dynamics
    for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning
    <https://arxiv.org/abs/1708.02596>`_
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 population_size,
                 reward_spec=TensorSpec(()),
                 planning_horizon=25,
                 upper_bound=None,
                 lower_bound=None,
                 name="RandomShootingAlgorithm"):
        """Create a RandomShootingAlgorithm.

        Args:
            population_size (int): the size of population for random shooting
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            planning_horizon (int): planning horizon in terms of time steps
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
        """
        super().__init__(
            feature_spec=feature_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            planning_horizon=planning_horizon,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            name=name)

        flat_action_spec = nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, ("RandomShootingAlgorithm doesn't "
                                            "support nested action_spec")

        self._population_size = population_size

        solution_size = self._planning_horizon * self._num_actions
        self._solution_size = solution_size

        # expand action bound to solution bound
        solution_upper_bound = self._upper_bound.unsqueeze(0).expand(
            planning_horizon, *self._upper_bound.shape).reshape(-1)
        solution_lower_bound = self._lower_bound.unsqueeze(0).expand(
            planning_horizon, *self._lower_bound.shape).reshape(-1)

        self._plan_optimizer = RandomOptimizer(
            solution_size,
            self._population_size,
            upper_bound=solution_upper_bound,
            lower_bound=solution_lower_bound,
            cost_func=self._calc_cost_for_action_sequence)

    def train_step(self, time_step: TimeStep, state, rollout_info=None):
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
        return AlgStep(output=(), state=state, info=())

    def predict_plan(self, time_step: TimeStep, state: PlannerState,
                     epsilon_greedy):
        assert self._action_seq_cost_func is not None, (
            "specify "
            "action sequence cost function before planning")

        opt_action = self._plan_optimizer.obtain_solution(
            time_step.observation)
        # [B, horizon * action_dim] -> [B, horizon, action_dim]
        opt_action = torch.reshape(
            opt_action,
            [opt_action.shape[0], self._planning_horizon, self._num_actions])
        action = opt_action[:, 0]
        return action, state

    def _calc_cost_for_action_sequence(self, obs, ac_seqs):
        """
        Args:
            obs (Tensor): initial observation to start the rollout for the
                evaluation of ac_seqs.
            ac_seqs(Tensor): action_sequence of shape [batch_size,
                    population_size, solution_dim]), where
                    solution_dim = planning_horizon * num_actions
        Returns:
            cost (Tensor) with shape [batch_size, population_size]
        """
        ac_seqs = ac_seqs.reshape(*ac_seqs.shape[0:2], self._planning_horizon,
                                  -1)
        cost = self._action_seq_cost_func(obs, ac_seqs)
        return cost

    def after_update(self, root_inputs, info):
        pass


@alf.configurable
class CEMPlanAlgorithm(RandomShootingAlgorithm):
    """CEM-based planning method.

    This method uses a Cross-Entropy Method (CEM) to optimize an action
    trajectory by minimizing a given cost function. The optimized action
    trajectory is termed as a 'plan' which can be used by other components
    such as a MPC-based controller. This has been used by some MBRL works
    such as `Deep Reinforcement Learning in a Handful of Trials using
    Probabilistic Dynamics Models <https://arxiv.org/abs/1805.12114>`_

    To speedup, when possible, we have used the plan obtained at the previous
    time step to initialize the the mean of the plan distribution at the current
    time step, after proper shifting and padding.
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 population_size,
                 planning_horizon,
                 reward_spec=TensorSpec(()),
                 elite_size=50,
                 max_iter_num=5,
                 epsilon=0.01,
                 tau=0.9,
                 scalar_var=None,
                 upper_bound=None,
                 lower_bound=None,
                 name="CEMPlanAlgorithm"):
        """Create a CEMPlanAlgorithm.

        Args:
            population_size (int): the size of population for optimization
            planning_horizon (int): planning horizon in terms of time steps
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s.)
            elite_size (int): the number of elites selected in each round
            max_iter_num (int|Tensor): the maximum number of CEM iterations
            epsilon (float): a minimum variance threshold. If the variance of
                the population falls below it, the CEM iteration will stop.
            tau (float): a value in (0, 1) for softly updating the population
                mean and variance:

                .. code-block:: python

                    mean = (1 - tau) * mean + tau * new_mean
                    var = (1 - tau) * var + tau * new_var

            scalar_var (None|float): the value that will be used to construct
                the initial diagonal covariance matrix of the multi-dimensional
                Gaussian used by the CEM optimizer. If value is None,
                0.5 * (upper_bound - lower_bound) is used.
            upper_bound (int): upper bound for elements in solution;
                action_spec.maximum will be used if not specified
            lower_bound (int): lower bound for elements in solution;
                action_spec.minimum will be used if not specified
        """
        super().__init__(
            feature_spec=feature_spec,
            action_spec=action_spec,
            population_size=population_size,
            reward_spec=reward_spec,
            planning_horizon=planning_horizon,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            name=name)

        solution_size = planning_horizon * self._num_actions

        self._plan_optimizer = CEMOptimizer(
            solution_size,
            self._population_size,
            upper_bound=self._upper_bound,
            lower_bound=self._lower_bound,
            cost_func=self._calc_cost_for_action_sequence,
            elite_size=elite_size,
            max_iter_num=max_iter_num,
            epsilon=epsilon,
            tau=tau)

        if scalar_var is None:
            self._scalar_var = (self._upper_bound - self._lower_bound) / 2.
        else:
            self._scalar_var = scalar_var

    def predict_plan(self, time_step: TimeStep, state: PlannerState,
                     epislon_greedy):
        prev_plan = state.prev_plan
        # [B, horizon, action_dim] -> [B, horizon*action_dim]
        prev_solution = prev_plan.reshape(prev_plan.shape[0], -1)

        prev_solution = prev_solution.clone()
        prev_solution[time_step.step_type == StepType.FIRST] = torch.full(
            (self._solution_size, ),
            (self._upper_bound + self._lower_bound) / 2.)

        init_mean = prev_solution.unsqueeze(1)

        opt_action = self._plan_optimizer.obtain_solution(
            time_step.observation,
            init_mean=init_mean,
            init_var=torch.ones_like(init_mean) * self._scalar_var)

        # [B, horizon * action_dim] -> [B, horizon, action_dim]
        opt_action = torch.reshape(
            opt_action,
            [opt_action.shape[0], self._planning_horizon, self._num_actions])

        # [B, horizon, action_dim]
        temporally_shifted_plan = torch.cat(
            (opt_action[:, 1:], opt_action.mean(
                dim=(0, 1), keepdim=True).expand(opt_action.shape[0], 1, -1)),
            1)

        action = opt_action[:, 0]
        new_state = state._replace(prev_plan=temporally_shifted_plan)

        return action, new_state
