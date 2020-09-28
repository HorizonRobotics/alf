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

from absl import logging
import gin
import numpy as np
import functools

import torch

import alf
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import (TimeStep, Experience, LossInfo, namedtuple,
                                 AlgStep, StepType)
from alf.experience_replayers.replay_buffer import l2_dist_close_reward_fn
from alf.optimizers.traj_optimizers import CEMOptimizer
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
import alf.utils.common as common

GoalState = namedtuple(
    "GoalState", [
        "goal", "full_plan", "steps_since_last_plan", "subgoals_achieved",
        "steps_since_last_goal"
    ],
    default_value=())
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
            a tuple
                - goal (Tensor): a batch of one-hot goal tensors.
                - new_state (nested Tensor): a potentially updated state.
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
            generated_goal, state = self.generate_goal(observation, state)
            new_goal = torch.where(new_goal_mask, generated_goal, state.goal)
            state = state._replace(goal=new_goal)
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
                full_plan=state.full_plan,
                steps_since_last_goal=state.steps_since_last_goal,
                subgoals_achieved=state.subgoals_achieved,
                steps_since_last_plan=state.steps_since_last_plan),
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
            if "aux_desired" in exp.observation:
                goal = torch.cat((goal, exp.observation["aux_desired"]), dim=1)
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
            a tuple
                - goal (Tensor): a batch of one-hot goal tensors.
                - new_state (nested Tensor): a potentially updated state.
        """
        batch_size = alf.nest.get_nest_batch_size(observation)
        goals = torch.randint(
            high=self._num_of_goals, size=(batch_size, ), dtype=torch.int64)
        goals_onehot = torch.nn.functional.one_hot(
            goals, self._num_of_goals).to(torch.float32)
        return goals_onehot, state

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
                 always_adopt_plan=False,
                 next_goal_on_success=False,
                 reward_fn=l2_dist_close_reward_fn,
                 max_subgoal_steps=10,
                 normalize_goals=False,
                 bound_goals=False,
                 value_fn=None,
                 value_state_spec=(),
                 max_replan_steps=10,
                 plan_margin=0.,
                 min_goal_cost_to_use_plan=0.,
                 use_aux_achieved=False,
                 aux_dim=0,
                 control_aux=False,
                 name="SubgoalPlanningGoalGenerator"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            num_subgoals (int): number of subgoals in a planned path.
            action_dim (int): number of dimensions of the (goal) action.
            action_bounds (pair of float Tensors or lists): min and max bounds of
                each action dimension.
            always_adopt_plan (bool): always adopt plan instead of having to meet
                certain conditions like more than plan_margin.
            next_goal_on_success (bool): switch to the next goal in plan after
                current subgoal is achieved.  When True, if max_subgoal_steps
                is reached, re-plan and switch to the next goal regardless of whether
                current subgoal is achieved or not.
            reward_fn (Callable): arguments: (achieved_goal, goal, device="cpu"), returns
                reward -1 or 0.  Arguments are supposed to be of shape
                ``(batch_size, batch_length, action_dim)``, so it can be the same as
                hindsight relabeler's reward_fn.
            normalize_goals (bool): whether to use normalizer to record stats for goal
                dimensions, and sample only within observed stats.
            bound_goals (bool): whether to use normalizer stats to bound planned goals.
            value_fn (Callable or None): value function to measure distance between states.
                When value_fn is None, it can also be set via set_value_fn.
            value_state_spec (nested TensorSpec): spec of the state of value function.
            max_replan_steps (int): number of steps to execute a plan before
                replanning.
            plan_margin (float): how much larger value than baseline does the plan need,
                to be adopted.
            min_goal_cost_to_use_plan (float): cost of original goal must be above
                this threshold for the plan to be accepted.
            use_aux_achieved (bool): whether to plan auxiliary achieved states like
                agent's speed, pose etc. in the field ``aux_achieved``.
            aux_dim (int): number of dimensions to plan for ``aux_achieved`` field.
            control_aux (bool): whether to output aux_achieved as part of goal,
                in which case, goal generator needs to output aux_desired as part of
                observation.  This is achieved by concatenating ``desired_goal``
                (of ``action_dim``) and ``aux_desired`` (of ``aux_dim``) to output
                ``state.goal`` (``action_dim + aux_dim``). When ``control_aux`` is
                ``False``, ``state.goal`` is of shape ``(batch_size, action_dim)``.
            name (str): name of the algorithm.
        """
        self._num_subgoals = num_subgoals
        self._always_adopt_plan = always_adopt_plan
        self._next_goal_on_success = next_goal_on_success
        self._max_subgoal_steps = max_subgoal_steps
        self._action_dim = action_dim
        self._max_replan_steps = max_replan_steps
        self._plan_margin = plan_margin
        self._min_goal_cost_to_use_plan = min_goal_cost_to_use_plan
        self._use_aux_achieved = use_aux_achieved
        self._aux_dim = aux_dim
        self._control_aux = control_aux
        if control_aux:
            assert use_aux_achieved
        goal_dim = action_dim
        if control_aux:
            goal_dim += aux_dim
        goal_shape = (goal_dim, )
        goal_spec = TensorSpec(goal_shape)
        plan_horizon = self._compute_plan_horizon()
        full_plan_shape = (plan_horizon, goal_dim)
        full_plan_spec = TensorSpec(full_plan_shape)
        steps_spec = TensorSpec((), dtype=torch.int32)
        train_state_spec = GoalState(
            goal=goal_spec,
            full_plan=full_plan_spec,
            steps_since_last_plan=steps_spec,
            subgoals_achieved=steps_spec,
            steps_since_last_goal=steps_spec)
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
        self._goal_achieved_fn = lambda achieved, goal: reward_fn(
            achieved.unsqueeze(1), goal.unsqueeze(1), device=goal.device
        ).squeeze(1) >= 0
        if value_fn:
            self._value_fn = value_fn

        self._value_state_spec = value_state_spec
        if use_aux_achieved:
            assert "aux_achieved" in observation_spec, (
                "aux_achieved not in observation_spec. "
                "Set use_aux_achieved flag in env?")
            # aux_dim bounds are assumed to be action_bounds[:][0]
            if isinstance(action_bounds[0], list):
                for _ in range(aux_dim):
                    action_bounds[0].append(action_bounds[0][0])
                    action_bounds[1].append(action_bounds[1][0])
        self._action_bounds = action_bounds
        self._normalizer = None
        if normalize_goals:
            self._normalizer = alf.utils.normalizers.EMNormalizer(
                observation_spec, name="observation/goal_gen_obs_normalizer")
        self._bound_goals = bound_goals
        if bound_goals:
            assert normalize_goals

    @property
    def control_aux(self):
        """Whether auxiliary dimensions are also planned.
        """
        return self._control_aux

    def set_value_fn(self, value_fn):
        self._value_fn = value_fn

    def _summarize_tensor_dims(self, name, t):
        for i in range(t.shape[1]):
            alf.summary.scalar(
                "{}_{}_mean.{}".format(name, i, common.exe_mode_name()),
                torch.mean(t[:, i]))

    def _costs_agg_dist(self, time_step, state, samples, info=None):
        assert self._value_fn, "no value function provided."
        (batch_size, pop_size, horizon, plan_dim) = samples.shape
        if self.control_aux:
            horizon -= 1
        assert (plan_dim == self._action_dim
                + self._aux_dim), "{} != {} + {}".format(
                    plan_dim, self._action_dim, self._aux_dim)
        action_dim = self._action_dim
        start = time_step.observation["achieved_goal"].reshape(
            batch_size, 1, 1, action_dim).expand(batch_size, pop_size, 1,
                                                 action_dim)
        end = time_step.observation["desired_goal"].reshape(
            batch_size, 1, 1, action_dim).expand(batch_size, pop_size, 1,
                                                 action_dim)
        if self._use_aux_achieved or self.control_aux:
            ag_samples = samples[:, :, :, :action_dim]
            if self.control_aux:
                # remove last step which is only used for control_aux
                ag_samples = ag_samples[:, :, :-1, :]
        else:
            ag_samples = samples
        if horizon > 0:
            samples_e = torch.cat([start, ag_samples, end], dim=2)
        else:
            samples_e = torch.cat([start, end], dim=2)
        stack_obs = time_step.observation.copy()
        stack_obs["observation"] = time_step.observation[
            "observation"].reshape(batch_size, 1, 1, -1).expand(
                batch_size, pop_size, horizon + 1, -1).reshape(
                    batch_size * pop_size * (horizon + 1), -1)
        stack_obs["achieved_goal"] = samples_e[:, :, :-1, :].reshape(
            batch_size * pop_size * (horizon + 1), action_dim)
        stack_obs["desired_goal"] = samples_e[:, :, 1:, :].reshape(
            batch_size * pop_size * (horizon + 1), action_dim)
        if self._use_aux_achieved:
            aux_start = time_step.observation["aux_achieved"].reshape(
                batch_size, 1, 1, self._aux_dim).expand(
                    batch_size, pop_size, 1, self._aux_dim)
            if horizon > 0:
                aux_start = torch.cat([
                    aux_start, samples[:, :, :horizon, action_dim:].reshape(
                        batch_size, pop_size, horizon, self._aux_dim)
                ],
                                      dim=2)
            stack_obs["aux_achieved"] = aux_start.reshape(
                batch_size * pop_size * (horizon + 1), self._aux_dim)
            if self.control_aux:
                stack_obs[
                    "aux_desired"] = samples[:, :, :, action_dim:].reshape(
                        batch_size * pop_size * (horizon + 1), self._aux_dim)
        if self._value_state_spec != ():
            raise NotImplementedError()
        values, _unused_value_state = self._value_fn(stack_obs, ())
        dists = -values.reshape(batch_size, pop_size, horizon + 1, -1)
        # values can be multi dimensional, need to sum to dim 3.
        if info is not None:
            info["all_population_costs"] = alf.math.sum_to_leftmost(
                dists, dim=3)
        agg_dist = alf.math.sum_to_leftmost(dists, dim=2)
        return agg_dist

    def _compute_plan_horizon(self):
        plan_horizon = self._num_subgoals
        if self.control_aux:
            # part of last subgoal is used to generate final goal state's aux_achieved
            plan_horizon = self._num_subgoals + 1
        return plan_horizon

    def generate_goal(self, observation, state):
        """Generate new goals.

        Args:
            observation (nested Tensor): the observation at the current time step.
            state (nested Tensor): state of this goal generator.

        Returns:
            a tuple
                - goal (Tensor): a batch of one-hot goal tensors.
                - new_state (nested Tensor): a potentially updated state.
        """
        if common.is_rollout() and self._normalizer:
            self._normalizer.update(observation)

        if self._use_aux_achieved or self.control_aux:
            # for control_aux, part of last subgoal is used to generate
            # the final goal state's aux_achieved
            plan_dim = self._action_dim + self._aux_dim
        else:
            plan_dim = self._action_dim
        plan_horizon = self._compute_plan_horizon()
        opt = CEMOptimizer(
            planning_horizon=plan_horizon,
            action_dim=plan_dim,
            bounds=self._action_bounds)
        opt.set_cost(self._costs_agg_dist)
        if self._normalizer:
            means = self._normalizer._mean_averager.get()
            m2s = self._normalizer._m2_averager.get()
            mean = means["desired_goal"]
            m2 = m2s["desired_goal"]
            if self._use_aux_achieved:
                mean = torch.cat((mean, means["aux_achieved"]))
                m2 = torch.cat((m2, m2s["aux_achieved"]))
            opt.set_initial_distributions(mean, m2, self._bound_goals)
        ts = TimeStep(observation=observation)
        assert self._value_fn, "set_value_fn before generate_goal"
        info = {}
        goals, costs = opt.obtain_solution(ts, (), info)
        if self.control_aux:
            goals[:, -1, :self._action_dim] = observation["desired_goal"]
        subgoal = goals[:, 0, :]  # the first subgoal in the plan
        if self._next_goal_on_success:
            # In reality, subgoal_achieved can be different for different ENVs.  Here,
            # we simply plan as if no subgoal is achieved, and adopt the right subgoal
            # based on the value of subgoals_achieved.
            # TODO: Fix the achieved subgoal's dimensions after CEM sampling and
            # only sample unachieved states.
            subgoal = goals[(torch.arange(goals.shape[0]),
                             state.subgoals_achieved.squeeze().long())]
        if self._num_subgoals > 0 and alf.summary.should_record_summaries():
            if "segment_costs" in info:
                self._summarize_tensor_dims("planner/segment_cost",
                                            info["segment_costs"])
            full_dist = torch.norm(
                observation["desired_goal"] - observation["achieved_goal"],
                dim=1)
            alf.summary.scalar(
                "planner/distance_full." + common.exe_mode_name(),
                torch.mean(full_dist))
            dist_ag_subgoal = torch.norm(
                subgoal[:, :self._action_dim] - observation["achieved_goal"],
                dim=1)
            alf.summary.scalar(
                "planner/distance_ag_to_subgoal." + common.exe_mode_name(),
                torch.mean(dist_ag_subgoal / full_dist))
            dist_subgoal_g = torch.norm(
                subgoal[:, :self._action_dim] - observation["desired_goal"],
                dim=1)
            alf.summary.scalar(
                "planner/distance_subgoal_to_goal." + common.exe_mode_name(),
                torch.mean(dist_subgoal_g / full_dist))

        if self._use_aux_achieved and not self.control_aux:
            subgoal = subgoal[:, :self._action_dim]
        if self._use_aux_achieved and self.control_aux:
            observation["aux_desired"] = goals[:, -1, self._action_dim:]
        # _value_fn relies on calling Q function with predicted action, but
        # with aux_control, lower level policy's input doesn't contain
        # aux_desired, and cannot predict action.
        # Properly handle this case of init_costs would probably add
        # another CEM process to predict goal aux dimensions first,
        # maximizing value_fn.
        values, _v_state = self._value_fn(observation, ())
        init_costs = -values
        # to deal with multi dim reward case
        init_costs = alf.math.sum_to_leftmost(init_costs, dim=2)
        # Assumes costs are positive, at least later on during training,
        # Otherwise, don't use planner.
        # We also require goal to be > min_goal_cost_to_use_plan away.
        # If goal is within the min_cost range, just use original goal.
        plan_success = (init_costs > self._min_goal_cost_to_use_plan) & (
            costs > 0) & (init_costs > costs * (1. + self._plan_margin))
        if self._always_adopt_plan:
            plan_success |= torch.tensor(1, dtype=torch.bool)
        if alf.summary.should_record_summaries():
            alf.summary.scalar(
                "planner/adoption_rate_planning." + common.exe_mode_name(),
                torch.mean(plan_success.float()))
            alf.summary.scalar(
                "planner/cost_mean_orig_goal." + common.exe_mode_name(),
                torch.mean(init_costs))
            alf.summary.scalar(
                "planner/cost_mean_planning." + common.exe_mode_name(),
                torch.mean(costs))
        orig_desired = observation["desired_goal"]
        if self.control_aux:
            orig_desired = goals[:, -1, :]
        self._summarize_tensor_dims("planner/planned_subgoal", subgoal)
        self._summarize_tensor_dims("planner/planned_orig_desired",
                                    orig_desired)
        if common.is_play():
            torch.set_printoptions(precision=2)
            if plan_success[0] > 0:
                outcome = "plan SUCCESS: "
            else:
                outcome = "plan fail: "
            init_costs_str = "init_cost: " + str(init_costs)
            ach = observation["achieved_goal"]
            if self._use_aux_achieved:
                ach = torch.cat((ach, observation["aux_achieved"]), dim=1)
            plan_horizon = goals.shape[1]
            if self.control_aux:
                plan_horizon -= 1
            if plan_horizon > 0:
                subgoal_str = str(goals[:, :plan_horizon, :]) + " ->\n"
            else:
                subgoal_str = ""
            logging.info(outcome + init_costs_str + " plan_cost:" +
                         str(costs) + ":\n" + str(ach) + " ->\n" +
                         subgoal_str + str(orig_desired))
            if self._num_subgoals > 0:
                logging.info("full_dist = {}, ag_sg = {}, sg_g = {}".format(
                    full_dist, dist_ag_subgoal, dist_subgoal_g))
        subgoal = torch.where(plan_success, subgoal, orig_desired)
        # Add full plan to state so we can get final goal's aux_desired,
        # as well as all the subgoals for the case when next_goal_on_success.
        if self._next_goal_on_success or self.control_aux:
            state = state._replace(full_plan=goals)
        return subgoal, state

    def _goal_achieved(self, observation, goal):
        achieved = observation["achieved_goal"]
        if self.control_aux:
            achieved = torch.cat((achieved, observation["aux_achieved"]),
                                 dim=1)
        return self._goal_achieved_fn(achieved, goal)

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
        inc_steps = state.steps_since_last_plan + 1
        steps_since_last_plan = torch.where(
            step_type == StepType.FIRST,
            torch.as_tensor(10000, dtype=torch.int32), inc_steps)
        update_cond = steps_since_last_plan > self._max_replan_steps
        if self._next_goal_on_success:
            inc_sg_steps = torch.where(step_type == StepType.FIRST,
                                       torch.zeros_like(inc_steps),
                                       state.steps_since_last_goal + 1)
            goal_achieved = self._goal_achieved(observation, state.goal)
            max_subgoal_steps_reached = inc_sg_steps > self._max_subgoal_steps
            skip_subgoal = max_subgoal_steps_reached | goal_achieved
            subgoals_achieved = torch.where(
                step_type == StepType.FIRST, torch.zeros_like(inc_steps),
                torch.min(
                    torch.as_tensor(self._num_subgoals, dtype=torch.int32),
                    state.subgoals_achieved + skip_subgoal))
            state = state._replace(
                subgoals_achieved=subgoals_achieved,
                steps_since_last_goal=inc_sg_steps)
            if alf.summary.should_record_summaries():
                alf.summary.scalar(
                    "planner/curr_goal_achieved." + common.exe_mode_name(),
                    torch.mean(goal_achieved.float()))
                alf.summary.scalar(
                    "planner/curr_goal_skipped." + common.exe_mode_name(),
                    torch.mean(max_subgoal_steps_reached.float()))
                alf.summary.scalar(
                    "planner/subgoal_index." + common.exe_mode_name(),
                    torch.mean(subgoals_achieved.float()))
                alf.summary.scalar(
                    "planner/steps_since_last_goal." + common.exe_mode_name(),
                    torch.mean(inc_sg_steps.float()))

        if torch.all(update_cond) or (
            (self.control_aux or self._next_goal_on_success)
                and torch.any(update_cond)):
            # Need to replan:
            # This is an efficiency trick so planning only happens for all envs
            # together at certain timesteps, instead of for some envs for every
            # timestep.
            state = state._replace(
                steps_since_last_plan=torch.zeros_like(inc_steps))
            if self._next_goal_on_success:
                alf.summary.scalar(
                    "planner/steps_since_last_goal-replan." +
                    common.exe_mode_name(), torch.mean(inc_sg_steps.float()))
            return torch.ones(
                step_type.shape, dtype=torch.uint8).unsqueeze(-1), state
        else:
            alf.summary.scalar(
                "planner/steps_since_last_plan-keepplan." +
                common.exe_mode_name(),
                torch.mean(steps_since_last_plan.float()))
            # No need to replan:
            if self.control_aux:
                # We can safely assume there are no FIRST steps due to
                # the if condition above.
                goals = state.goal
                # Populate aux_desired from the existing plan.
                observation["aux_desired"] = goals[:, self._action_dim:]
            else:
                goals = torch.where(
                    (step_type == StepType.FIRST).unsqueeze(-1),
                    observation["desired_goal"], state.goal)

            if self._next_goal_on_success:
                # There are no FIRST steps in this if branch.
                goals = state.full_plan[(torch.arange(
                    goals.shape[0]), state.subgoals_achieved.squeeze().long())]
                alf.summary.scalar(
                    "planner/steps_since_last_goal-keepplan." +
                    common.exe_mode_name(), torch.mean(inc_sg_steps.float()))

            state = state._replace(
                steps_since_last_plan=steps_since_last_plan, goal=goals)
            return None, state
