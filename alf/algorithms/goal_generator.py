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

import gin
import numpy as np
import functools

import torch

import alf
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import (TimeStep, Experience, LossInfo, namedtuple,
                                 AlgStep, StepType)
from alf.optimizers.traj_optimizers import CEMOptimizer
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
import alf.utils.common as common

GoalState = namedtuple(
    "GoalState", ["goal", "steps_since_last_goal"], default_value=())
GoalInfo = namedtuple(
    "GoalInfo", ["goal", "original_goal", "loss"], default_value=())


@gin.configurable
class ConditionalGoalGenerator(RLAlgorithm):
    """Conditional Goal Generation Module.

    This module generates a random categorical goal for the agent
    in the beginning of every episode.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 train_with_goal="rollout",
                 train_state_spec=(),
                 name="ConditionalGoalGenerator"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested TensorSpec): representing the action.
            train_with_goal (str): source of goal during training: "rollout" for
                rollout_info generated during unroll by goal generator, or "exp"
                for the "desired_goal" field in observation.  Use "exp" if using
                hindsight relabeling.
            train_state_spec (nested tensorSpec): representing the train state.
            name (str): name of the algorithm.
        """
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            train_state_spec=train_state_spec,
            name=name)
        self._train_with_goal = train_with_goal

    def update_condition(self, observation, step_type, state):
        """Condition to update the goals.

        Args:
            observation (nested Tensor): the observation at the current time step
            step_type (StepType): step type for current observation
            state (nested Tensor): state of this goal generator

        Returns:
            - update mask (Tensor or None): representing the envs to update
                goals for.  When None, no updates will be made.
            - state (nest): new goal generator state.
        """
        raise NotImplementedError()

    def generate_goal(self, observation, state):
        """Generate new goals.

        Args:
            observation (nested Tensor): the observation at the current time step.
            state (nested Tensor): state of this goal generator.

        Returns:
            Tensor: a batch of goal tensors.
        """
        raise NotImplementedError()

    def _update_goal(self, observation, state, step_type):
        """Update the goal if the episode just beginned; otherwise keep using
        the goal in ``state``.

        Args:
            observation (nested Tensor): the observation at the current time step
            state (nested Tensor): state of this goal generator
            step_type (StepType): step type for current observation

        Returns:
            - goal (Tensor): a batch of one-hot tensors representing the updated goals.
            - state (nest).
        """
        new_goal_mask, state = self.update_condition(observation, step_type,
                                                     state)
        if new_goal_mask is not None:
            generated_goal = self.generate_goal(observation, state)
            new_goal = torch.where(new_goal_mask, generated_goal, state.goal)
            return new_goal, state
        else:
            return state.goal, state

    def _step(self, time_step: TimeStep, state):
        """Perform one step of rollout or prediction.

        Note that as ``RandomCategoricalGoalGenerator`` is a non-trainable module,
        and it will randomly generate goals for episode beginnings.

        Args:
            time_step (TimeStep): input time_step data.
            state (nested Tensor): consistent with ``train_state_spec``.
        Returns:
            AlgStep:
            - output (Tensor); one-hot goal vectors.
            - state (nested Tensor):
            - info (GoalInfo): storing any info that will be put into a replay
              buffer (if off-policy training is used.
        """
        observation = time_step.observation
        step_type = time_step.step_type
        new_goal, state = self._update_goal(observation, state, step_type)
        return AlgStep(
            output=new_goal,
            state=GoalState(
                goal=new_goal,
                steps_since_last_goal=state.steps_since_last_goal),
            info=GoalInfo(goal=new_goal))

    def rollout_step(self, time_step: TimeStep, state):
        return self._step(time_step, state)

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        return self._step(time_step, state)

    def train_step(self, exp: Experience, state):
        """For off-policy training, the current output goal should be taken from
        the goal in ``exp.rollout_info`` (historical goals generated during rollout).

        Note that we cannot take the goal from ``state`` and pass it down because
        the first state might be a zero vector. And we also cannot resample
        the goal online because that might be inconsistent with the sampled
        experience trajectory.

        Args:
            exp (Experience): the experience data whose ``rollout_info`` has been
                replaced with goal generator ``rollout_info``.
            state (nested Tensor):

        Returns:
            AlgStep:
            - output (Tensor); one-hot goal vectors
            - state (nested Tensor):
            - info (GoalInfo): for training.
        """
        if self._train_with_goal == 'rollout':
            goal = exp.rollout_info.goal
        elif self._train_with_goal == 'exp':
            goal = exp.observation["desired_goal"]
        elif self._train_with_goal == 'orig':
            goal = exp.rollout_info.original_goal
        else:
            raise NotImplementedError()
        return AlgStep(output=goal, state=state, info=GoalInfo(goal=goal))

    def calc_loss(self, experience, info: GoalInfo):
        return LossInfo()


@gin.configurable
class RandomCategoricalGoalGenerator(ConditionalGoalGenerator):
    """Random Goal Generation Module.

    This module generates a random categorical goal for the agent
    in the beginning of every episode.
    """

    def __init__(self,
                 observation_spec,
                 num_of_goals,
                 name="RandomCategoricalGoalGenerator"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            num_of_goals (int): total number of goals the agent can sample from.
            name (str): name of the algorithm.
        """
        goal_spec = TensorSpec((num_of_goals, ))
        train_state_spec = GoalState(goal=goal_spec)
        super().__init__(
            observation_spec=observation_spec,
            action_spec=BoundedTensorSpec(
                shape=(num_of_goals, ),
                dtype='float32',
                minimum=0.,
                maximum=1.),
            train_state_spec=train_state_spec,
            name=name)
        self._num_of_goals = num_of_goals

    def generate_goal(self, observation, state):
        """Generate new goals.

        Args:
            observation (nested Tensor): the observation at the current time step.
            state (nested Tensor): state of this goal generator.

        Returns:
            Tensor: a batch of one-hot goal tensors.
        """
        batch_size = alf.nest.get_nest_batch_size(observation)
        goals = torch.randint(
            high=self._num_of_goals, size=(batch_size, ), dtype=torch.int64)
        goals_onehot = torch.nn.functional.one_hot(
            goals, self._num_of_goals).to(torch.float32)
        return goals_onehot

    def update_condition(self, observation, step_type, state):
        """Condition to update the goals.

        Args:
            observation (nested Tensor): current observation
            step_type (StepType): step type for current observation
            state (nested Tensor): state of this goal generator

        Returns:
            - update mask (Tensor or None): representing the envs to update
                goals for.  When None, no updates will be made.
            - state (nest): new goal generator state.
        """
        new_goal_mask = torch.unsqueeze((step_type == StepType.FIRST), dim=-1)
        return new_goal_mask, state


@gin.configurable
class SubgoalPlanningGoalGenerator(ConditionalGoalGenerator):
    """Subgoal Planning Goal Generation Module.

    This module generates the next subgoal for the agent every n steps, by
    generating a path of subgoals from current state to the goal state using
    the goal conditioned value function of the rl_algorithm.
    """

    def __init__(self,
                 observation_spec,
                 num_subgoals,
                 action_dim,
                 action_bounds,
                 value_fn=None,
                 value_state_spec=(),
                 max_subgoal_steps=10,
                 plan_margin=0.,
                 min_goal_cost_to_use_plan=0.,
                 name="SubgoalPlanningGoalGenerator"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            num_subgoals (int): number of subgoals in a planned path.
            action_dim (int): number of dimensions of the (goal) action.
            action_bounds (pair of float Tensors or lists): min and max bounds of
                each action dimension.
            value_fn (Callable or None): value function to measure distance between states.
                When value_fn is None, it can also be set via set_value_fn.
            value_state_spec (nested TensorSpec): spec of the state of value function.
            max_subgoal_steps (int): number of steps to execute any subgoal before
                updating it.
            plan_margin (float): how much larger value than baseline does the plan need,
                to be adopted.
            min_goal_cost_to_use_plan (float): cost of original goal must be above
                this threshold for the plan to be accepted.
            name (str): name of the algorithm.
        """
        goal_shape = (action_dim, )
        goal_spec = TensorSpec(goal_shape)
        steps_spec = TensorSpec((), dtype=torch.int32)
        train_state_spec = GoalState(
            goal=goal_spec, steps_since_last_goal=steps_spec)
        assert len(action_bounds) == 2, "specify min and max bounds"
        super().__init__(
            observation_spec,
            action_spec=BoundedTensorSpec(
                shape=goal_shape,
                dtype='float32',
                minimum=torch.min(
                    torch.as_tensor(action_bounds[0],
                                    dtype=torch.float32)).item(),
                maximum=torch.max(
                    torch.as_tensor(action_bounds[1],
                                    dtype=torch.float32)).item()),
            train_state_spec=train_state_spec,
            name=name)
        self._num_subgoals = num_subgoals
        self._action_dim = action_dim
        if value_fn:
            self._value_fn = value_fn
        self._max_subgoal_steps = max_subgoal_steps
        self._plan_margin = plan_margin
        self._min_goal_cost_to_use_plan = min_goal_cost_to_use_plan

        self._value_state_spec = value_state_spec
        self._opt = CEMOptimizer(
            planning_horizon=num_subgoals,
            action_dim=action_dim,
            bounds=action_bounds)

        def _costs_agg_dist(time_step, state, samples):
            assert self._value_fn, "no value function provided."
            (batch_size, pop_size, horizon, action_dim) = samples.shape
            start = time_step.observation["achieved_goal"].reshape(
                batch_size, 1, 1, action_dim).expand(batch_size, pop_size, 1,
                                                     action_dim)
            end = time_step.observation["desired_goal"].reshape(
                batch_size, 1, 1, action_dim).expand(batch_size, pop_size, 1,
                                                     action_dim)
            samples_e = torch.cat([start, samples, end], dim=2)
            stack_obs = time_step.observation.copy()
            stack_obs["observation"] = time_step.observation[
                "observation"].reshape(batch_size, 1, 1, -1).expand(
                    batch_size, pop_size, horizon + 1, -1).reshape(
                        batch_size * pop_size * (horizon + 1), -1)
            stack_obs["achieved_goal"] = samples_e[:, :, :-1, :].reshape(
                batch_size * pop_size * (horizon + 1), -1)
            stack_obs["desired_goal"] = samples_e[:, :, 1:, :].reshape(
                batch_size * pop_size * (horizon + 1), -1)

            if self._value_state_spec != ():
                raise NotImplementedError()
            values, _unused_value_state = self._value_fn(stack_obs, ())
            dists = -values.reshape(batch_size, pop_size, horizon + 1)
            agg_dist = torch.sum(dists, dim=2)
            return agg_dist

        self._opt.set_cost(_costs_agg_dist)

    def set_value_fn(self, value_fn):
        self._value_fn = value_fn

    def generate_goal(self, observation, state):
        """Generate new goals.

        Args:
            observation (nested Tensor): the observation at the current time step.
            state (nested Tensor): state of this goal generator.

        Returns:
            Tensor: a batch of goal tensors.
        """
        ts = TimeStep(observation=observation)
        assert self._value_fn, "set_value_fn before generate_goal"
        values, _v_state = self._value_fn(observation, ())
        init_costs = -values
        goals, costs = self._opt.obtain_solution(ts, ())
        subgoal = goals[:, 0, :]  # the first subgoal in the plan
        # Assumes costs are positive, at least later on during training,
        # Otherwise, don't use planner.
        # We also require goal to be > min_goal_cost_to_use_plan away.
        # If goal is within the min_cost range, just use original goal.
        plan_success = (init_costs > self._min_goal_cost_to_use_plan) & (
            costs > 0) & (init_costs > costs * (1. + self._plan_margin))
        alf.summary.scalar(
            "planner/plan_adoption_rate" + "." + common.exe_mode_name(),
            torch.mean(plan_success.float()))
        alf.summary.scalar(
            "planner/cost_mean_planned" + "." + common.exe_mode_name(),
            torch.mean(costs))
        alf.summary.scalar(
            "planner/cost_mean_orig_goal" + "." + common.exe_mode_name(),
            torch.mean(init_costs))
        subgoal = torch.where(plan_success, subgoal,
                              observation["desired_goal"])
        return subgoal

    def update_condition(self, observation, step_type, state):
        """Condition to update the goals.

        Args:
            observation (nested Tensor): the observation at the current time step
            step_type (StepType): step type for current observation
            state (nested Tensor): state of this goal generator

        Returns:
            - update mask (Tensor or None): representing the envs to update
                goals for.  When None, no updates will be made.
            - state (nest): new goal generator state.
        """
        inc_steps = state.steps_since_last_goal + 1
        steps_since_last_goal = torch.where(
            step_type == StepType.FIRST,
            torch.as_tensor(10000, dtype=torch.int32), inc_steps)
        if torch.all(steps_since_last_goal > self._max_subgoal_steps):
            # This is an efficiency trick so planning only happens for all envs
            # together.
            state = state._replace(
                steps_since_last_goal=torch.zeros_like(inc_steps))
            return torch.ones(
                step_type.shape, dtype=torch.uint8).unsqueeze(-1), state
        else:
            goals = torch.where((step_type == StepType.FIRST).unsqueeze(-1),
                                observation["desired_goal"], state.goal)
            state = state._replace(
                steps_since_last_goal=steps_since_last_goal, goal=goals)
            return None, state
