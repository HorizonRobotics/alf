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
        "goal", "full_plan", "steps_since_last_plan", "subgoals_index",
        "steps_since_last_goal", "final_goal", "replan", "switched_goal",
        "retain_old", "advanced_goal"
    ],
    default_value=())
GoalInfo = namedtuple(
    "GoalInfo",
    ["goal", "original_goal", "final_goal", "switched_goal", "loss"],
    default_value=())


@gin.configurable
class ConditionalGoalGenerator(RLAlgorithm):
    """Conditional Goal Generation Module.

    This module generates a random categorical goal for the agent
    in the beginning of every episode.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 train_with_goal="exp",
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

    def post_process(self, observation, state, step_type, epsilon_greedy=1.):
        """Post process after update_condition and generate_goal calls.

        Args:
            observation (nested Tensor): the observation at the current time step
            state (nested Tensor): state of this goal generator, with state.goal being the new goal
            step_type (StepType): step type for current observation
            epsilon_greedy (float): whether to add noise to goal

        Returns:
            - goal (Tensor): a batch of one-hot tensors representing the updated goals.
            - state (nest).
        """
        return state.goal, state

    def _step(self, time_step: TimeStep, state, epsilon_greedy):
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
        new_goal, state = self.post_process(observation, state, step_type,
                                            epsilon_greedy)
        return AlgStep(
            output=new_goal,
            state=GoalState(
                goal=new_goal,
                full_plan=state.full_plan,
                final_goal=state.final_goal,
                steps_since_last_goal=state.steps_since_last_goal,
                subgoals_index=state.subgoals_index,
                steps_since_last_plan=state.steps_since_last_plan),
            info=GoalInfo(goal=new_goal, final_goal=state.final_goal))

    def rollout_step(self, time_step: TimeStep, state):
        return self.predict_step(time_step, state, epsilon_greedy=1.)

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy=1.):
        return self._step(time_step, state, epsilon_greedy)

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
            if "final_goal" in exp.observation:
                state = state._replace(
                    final_goal=exp.observation["final_goal"])
            state = state._replace(
                switched_goal=exp.rollout_info.switched_goal)
        elif self._train_with_goal == 'orig':
            goal = exp.rollout_info.original_goal
        else:
            raise NotImplementedError()
        return AlgStep(
            output=goal,
            state=state,
            info=GoalInfo(
                goal=goal,
                final_goal=state.final_goal,
                switched_goal=state.switched_goal))

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
                 final_goal=False,
                 reward_fn=l2_dist_close_reward_fn,
                 sparse_reward=False,
                 max_subgoal_steps=50,
                 normalize_goals=False,
                 bound_goals=False,
                 value_fn=None,
                 combine_her_nonher_value_weight=0.,
                 value_state_spec=(),
                 max_replan_steps=10,
                 plan_margin=0.,
                 plan_with_goal_value_only=False,
                 plan_cost_ln_norm=1,
                 goal_speed_zero=False,
                 min_goal_cost_to_use_plan=0.,
                 use_aux_achieved=False,
                 aux_dim=0,
                 control_aux=False,
                 gou_stddev=0.,
                 gou_damping=0.15,
                 multi_dim_goal_reward=False,
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
            final_goal (bool): put whether current goal is final into observation.
            reward_fn (Callable): arguments: (achieved_goal, goal, device="cpu"), returns
                reward -1 or 0.  Arguments are supposed to be of shape
                ``(batch_size, batch_length, action_dim)``, so it can be the same as
                hindsight relabeler's reward_fn.
            sparse_reward (bool): Whether to accept 0/1 reward as input.  Requires
                the planning algorithm to take logarithm before summing segment costs.
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
            plan_cost_ln_norm (int): ln norm applied to segment costs before adding
                segment costs to form the full plan cost.
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
            gou_stddev (float): if non-zero, stddev of the ou noise added to each
                dim of output goal in the default collect policy.
            gou_damping (float): Damping factor for the OU noise added in the
                default collect policy.
            multi_dim_goal_reward (bool): number of dimensions of goal reward.
            name (str): name of the algorithm.
        """
        self._num_subgoals = num_subgoals
        self._always_adopt_plan = always_adopt_plan
        self._next_goal_on_success = next_goal_on_success
        if next_goal_on_success:
            assert always_adopt_plan, "TODO: replan after achieving subgoal."
        self._final_goal = final_goal
        self._sparse_reward = sparse_reward
        self._max_subgoal_steps = max_subgoal_steps
        self._action_dim = action_dim
        self._max_replan_steps = max_replan_steps
        self._plan_margin = plan_margin
        self._plan_with_goal_value_only = plan_with_goal_value_only
        self._plan_cost_ln_norm = plan_cost_ln_norm
        self._goal_speed_zero = goal_speed_zero
        self._min_goal_cost_to_use_plan = min_goal_cost_to_use_plan
        self._use_aux_achieved = use_aux_achieved
        self._aux_dim = aux_dim
        self._control_aux = control_aux
        if control_aux:
            assert use_aux_achieved
        self._multi_dim_goal_reward = multi_dim_goal_reward
        self._goal_dim = action_dim
        if control_aux:
            self._goal_dim += aux_dim
        goal_shape = (self._goal_dim, )
        goal_spec = TensorSpec(goal_shape)
        plan_horizon = self._compute_plan_horizon()
        full_plan_shape = (plan_horizon, self._goal_dim)
        full_plan_spec = TensorSpec(full_plan_shape)
        final_goal_spec = ()
        if final_goal:
            final_goal_spec = TensorSpec((1, ))
        steps_spec = TensorSpec((), dtype=torch.int32)
        train_state_spec = GoalState(
            goal=goal_spec,
            full_plan=full_plan_spec,
            final_goal=final_goal_spec,
            steps_since_last_plan=steps_spec,
            subgoals_index=steps_spec,
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

        def _goal_ach_fn(achieved, goal):
            ach = reward_fn(
                achieved.unsqueeze(1),
                goal.unsqueeze(1),
                multi_dim_goal_reward=multi_dim_goal_reward,
                device=goal.device).squeeze(1) >= 0
            if multi_dim_goal_reward:
                ach = torch.min(ach, dim=1)[0]
            return ach

        self._goal_achieved_fn = _goal_ach_fn
        if value_fn:
            self._value_fn = value_fn
        self._combine_her_nonher_value_weight = combine_her_nonher_value_weight

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
        if not self._normalizer:
            alf.utils.checkpoint_utils.enable_checkpoint(self, False)
        self._ou_process = common.create_ou_process(goal_spec, gou_stddev,
                                                    gou_damping)
        self._gou_stddev = gou_stddev

    def goal_dim(self):
        """Number of dimensions of a goal.
        """
        return self._goal_dim

    @gin.configurable
    def reward_spec(self, env_reward_spec=None):
        if env_reward_spec is None:
            env_reward_spec = common.get_reward_spec()
        if self._multi_dim_goal_reward:
            env_dim = 1
            if env_reward_spec.shape:
                env_dim = env_reward_spec.shape[0]
            goal_shape = (self.goal_dim() + env_dim - 1, )
            reward_spec = TensorSpec(goal_shape)
            return reward_spec
        else:
            return env_reward_spec

    @property
    def sparse_reward(self):
        """Using 0/1 reward
        """
        return self._sparse_reward

    @property
    def num_subgoals(self):
        """Number of subgoals to plan.
        """
        return self._num_subgoals

    @property
    def control_aux(self):
        """Whether auxiliary dimensions are also planned.
        """
        return self._control_aux

    def set_value_fn(self, value_fn):
        self._value_fn = value_fn

    def _wrapped_value_fn(self, obs, state):
        # input tensors can be 2 (batch_size, batch_length) or 3 dimensional
        # (batch_size, population, plan_horizon).
        # This function consolidates multi dim value functions into a single dimension.
        values, state = self._value_fn(obs, state)
        obs_dim = len(alf.nest.get_nest_shape(obs))
        mdim = obs_dim <= values.ndim
        if self.sparse_reward:
            # 0/1 reward, value cannot be negative, or too close to zero.
            # Sum all dims before taking log:
            if mdim and self._plan_with_goal_value_only:
                values = values[..., :-1]  # last dim is distraction reward
            values = alf.math.sum_to_leftmost(values, dim=obs_dim)
            # values = torch.min(values, torch.tensor(1.))  # is this needed?
            values = torch.log(torch.max(values, torch.tensor(1.e-10)))
            # Multiply all dims (sum after log):
            # if mdim and not self._plan_with_goal_value_only:
            #     # multi dim reward
            #     goal_v = values[..., :-1]  # last dim is distraction reward
            # else:
            #     goal_v = values
            # goal_v = torch.max(goal_v, torch.tensor(1.e-10))
            # goal_v = torch.log(goal_v)
            # if mdim:
            #     values[..., :-1] = goal_v
            # else:
            #     values = goal_v
        else:
            # -1/0 reward, value cannot be positive.
            values = torch.min(values, torch.tensor([0.]))
            if mdim and self._plan_with_goal_value_only:
                values = values[..., :-1]
            values = alf.math.sum_to_leftmost(values, dim=obs_dim)
        return values, state

    def _summarize_tensor_dims(self, name, t):
        if alf.summary.should_record_summaries():
            for i in range(t.shape[1]):
                alf.summary.scalar(
                    "{}_{}_mean.{}".format(name, i, common.exe_mode_name()),
                    torch.mean(t[:, i]))
                alf.summary.scalar(
                    "{}_{}_var.{}".format(name, i, common.exe_mode_name()),
                    torch.var(t[:, i]))

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
                # remove last step which is only used for control_aux,
                # and will be filled with observation["desired_goal"].
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
        if self._final_goal:
            stack_obs["final_goal"] = torch.zeros(batch_size, pop_size,
                                                  horizon + 1)
            stack_obs["final_goal"][:, :, -1] = torch.ones(())
            stack_obs["final_goal"] = stack_obs["final_goal"].reshape(-1, 1)
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
                if self._goal_speed_zero:
                    samples[:, :, -1, action_dim:action_dim +
                            2] = 0  # x, y speed
                    samples[:, :, -1, action_dim + 5] = 0  # yaw speed
                stack_obs[
                    "aux_desired"] = samples[:, :, :, action_dim:].reshape(
                        batch_size * pop_size * (horizon + 1), self._aux_dim)
        if self._value_state_spec != ():
            raise NotImplementedError()
        values, _unused_value_state = self._wrapped_value_fn(stack_obs, ())
        dists = -values.reshape(batch_size, pop_size, horizon + 1, -1)
        if info is not None:
            info["all_population_costs"] = dists
        if self._plan_cost_ln_norm > 1:
            if self._plan_cost_ln_norm < 10:
                dists = dists**self._plan_cost_ln_norm
            else:
                dists = torch.max(dists, dim=2)[0]  # L-inf norm
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
        batch_size = goals.shape[0]
        if self.control_aux:
            # This portion of the CEM samples were never used in the
            # _costs_agg_dist function, and were filled with desired_goal.
            goals[:, -1, :self._action_dim] = observation["desired_goal"]
        else:
            goals = torch.cat((goals[:, :, :self._action_dim],
                               observation["desired_goal"].unsqueeze(1)),
                              dim=1)
        old_plan = state.full_plan
        if not self.control_aux:
            assert False, "We give up on non-speed control mode."
        old_costs = opt.cost_function(ts, state, old_plan.unsqueeze(1))
        retain_old = old_costs < costs
        goals = torch.where(retain_old.reshape(-1, 1, 1), old_plan, goals)
        state = state._replace(
            replan=state.replan & ~retain_old, retain_old=retain_old)

        subgoal = goals[:, 0, :]  # the first subgoal in the plan
        if self._next_goal_on_success:
            # In reality, subgoals_index can be different for different ENVs.  Here,
            # we simply plan all subgoals, and choose the subgoal specified by
            # the subgoals_index.
            # TODO: only calc costs for subgoals to be achieved.
            cap_subgoals_index = torch.min(
                torch.as_tensor(self._num_subgoals, dtype=torch.int32),
                state.subgoals_index)

            subgoal = goals[(torch.arange(batch_size),
                             cap_subgoals_index.squeeze().long())]

        if self._num_subgoals > 0 and (alf.summary.should_record_summaries()
                                       or common.is_play()):
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
                torch.mean(dist_ag_subgoal))
            dist_subgoal_g = torch.norm(
                subgoal[:, :self._action_dim] - observation["desired_goal"],
                dim=1)
            alf.summary.scalar(
                "planner/distance_subgoal_to_goal." + common.exe_mode_name(),
                torch.mean(dist_subgoal_g))

        if self._use_aux_achieved and not self.control_aux:
            subgoal = subgoal[:, :self._action_dim]
        # _value_fn relies on calling Q function with predicted action, but
        # with aux_control, lower level policy's input doesn't contain
        # aux_desired, and cannot predict action.
        # Properly handle this case of init_costs would probably add
        # another CEM process to predict goal aux dimensions first,
        # maximizing value_fn.
        if self._use_aux_achieved and self.control_aux:
            observation["aux_desired"] = goals[:, -1, self._action_dim:]
        if self._final_goal:
            # To compute overall cost of base policy, all goals are final.
            observation["final_goal"] = torch.ones((batch_size, 1))
        values, _v_state = self._wrapped_value_fn(observation, ())
        init_costs = -values
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
        self._summarize_tensor_dims("observation/planned_subgoal", subgoal)
        self._summarize_tensor_dims("observation/planned_orig_desired",
                                    orig_desired)
        if common.is_play():
            torch.set_printoptions(
                precision=2, sci_mode=False, linewidth=110, profile="full")
            if plan_success[0] > 0:
                outcome = "plan SUCCESS:"
            else:
                outcome = "plan fail:"
            ach = observation["achieved_goal"]
            if self._use_aux_achieved:
                ach = torch.cat((ach, observation["aux_achieved"]), dim=1)
            plan_horizon = goals.shape[1]
            if self.control_aux:
                plan_horizon -= 1
            if plan_horizon > 0:
                subgoal_str = "subg: " + "\n      ".join(
                    [str(goals[:, i, :])
                     for i in range(plan_horizon)]) + " ->\n"
            else:
                subgoal_str = ""
            logging.info(
                "%s init_cost: %f plan_cost: %f:\nobs: %s\nachv: %s ->\n"
                "%sgoal: %s", outcome, init_costs, costs,
                str(observation["observation"]), str(ach), subgoal_str,
                str(orig_desired))
            if self._num_subgoals > 0 and "segment_costs" in info:
                logging.info(
                    "full_dist=%f; ag_sg=%f, cost=%f; sg_g=%f, cost=%f",
                    full_dist, dist_ag_subgoal, info["segment_costs"][0, 0],
                    dist_subgoal_g, info["segment_costs"][0, 1])
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
        first_steps = step_type == StepType.FIRST
        steps_since_last_plan = state.steps_since_last_plan + torch.tensor(
            1, dtype=torch.int32)
        update_cond = (steps_since_last_plan >
                       self._max_replan_steps) | first_steps
        steps_since_last_plan = torch.where(update_cond,
                                            torch.tensor(0, dtype=torch.int32),
                                            steps_since_last_plan)
        state = state._replace(
            steps_since_last_plan=steps_since_last_plan,
            replan=update_cond,
            retain_old=torch.zeros_like(update_cond))
        if torch.all(update_cond) or (
            (self.control_aux or self._next_goal_on_success)
                and torch.any(update_cond)):
            # Need to replan:
            return update_cond.unsqueeze(-1), state
        else:
            # Not replan:
            if self.control_aux:
                assert not torch.any(first_steps), (
                    "replan condition should ensure no first steps here")
                goals = state.goal
            else:
                goals = torch.where(
                    (step_type == StepType.FIRST).unsqueeze(-1),
                    observation["desired_goal"], state.goal)
            state = state._replace(goal=goals)
            return None, state

    def _unreal(self, s):
        return torch.any(self._unreal_batch(s))

    def _unreal_batch(self, s):
        return ((torch.abs(s[:, 0]) > 5.7 + 0.5)
                | (torch.abs(s[:, 1]) > 5.7 + 0.5)
                | (abs(s[:, 7]) > 7.5)
                | (abs(s[:, 11]) > 3.15 + 0.5)
                |
                (torch.norm(torch.cat(
                    (s[:, 4:7], s[:, 8:11]), dim=1), dim=1) >= 3))

    def post_process(self, observation, state, step_type, epsilon_greedy=1.):
        """Post process after update_condition and generate_goal calls.

        This is needed because generate_goal isn't always called for every update_condition call.

        Args:
            observation (nested Tensor): the observation at the current time step
            state (nested Tensor): state of this goal generator, with state.goal being the new goal
            step_type (StepType): step type for current observation

        Returns:
            - goal (Tensor): a batch of one-hot tensors representing the updated goals.
            - state (nest).
        """
        new_goal = state.goal
        if self._final_goal:
            final_goal = torch.zeros(new_goal.shape[0])
        if self._next_goal_on_success:
            if self._final_goal:
                final_goal = (state.subgoals_index >= self._num_subgoals)
            # Judge whether we reached the subgoal
            first_steps = step_type == StepType.FIRST
            goal_achieved = self._goal_achieved(observation, new_goal)
            # Judge whether we reached max subgoal steps, and need to skip current subgoal
            # Even at the first step of a new plan, goal may be achieved.
            sg_steps = state.steps_since_last_goal
            # NOTE: when index < num_subgoals, it's a subgoal; when index == num_subgoals,
            # it's the final goal; when index == num_subgoals + 1, final goal has been achieved;
            # when index == num_subgoals + 2, exited final goal region.
            current_subgoal_index = state.subgoals_index
            max_subgoal_steps_reached = sg_steps > self._max_subgoal_steps
            curr_goal_skipped = (max_subgoal_steps_reached & ~goal_achieved
                                 & ~first_steps)
            next_subgoal = max_subgoal_steps_reached | goal_achieved
            # Compute index of subgoal
            subgoals_index = torch.where(
                first_steps, torch.zeros_like(sg_steps),
                torch.min(
                    torch.as_tensor(self._num_subgoals + 1, dtype=torch.int32),
                    current_subgoal_index + next_subgoal))
            cap_subgoals_index = torch.min(
                torch.as_tensor(self._num_subgoals, dtype=torch.int32),
                subgoals_index)
            advanced_goal = subgoals_index > current_subgoal_index
            exit_goal = ~goal_achieved & (
                current_subgoal_index == self._num_subgoals + 1)
            subgoals_index += exit_goal

            # Get the subgoal from full plan
            batch_size = step_type.shape[0]
            new_goal = state.full_plan[(torch.arange(batch_size),
                                        cap_subgoals_index.squeeze().long())]
            new_sg_steps = torch.where(first_steps | next_subgoal,
                                       torch.zeros_like(sg_steps), sg_steps)
            new_sg_steps += 1
            # Reset plan steps counter if subgoal is reached.
            steps_since_last_plan = torch.where(
                advanced_goal, torch.tensor(0, dtype=torch.int32),
                state.steps_since_last_plan)
            state = state._replace(
                goal=new_goal,
                subgoals_index=subgoals_index,
                steps_since_last_goal=new_sg_steps,
                steps_since_last_plan=steps_since_last_plan,
                switched_goal=state.replan | advanced_goal,
                advanced_goal=advanced_goal)

        if self._final_goal:
            state = state._replace(final_goal=final_goal.float().unsqueeze(1))
        # Populate aux_desired into observation, very important for agent to see!
        if self.control_aux:
            observation["aux_desired"] = new_goal[:, self._action_dim:]

        if common.is_play():
            ach = observation["achieved_goal"]
            desire = observation["desired_goal"]
            if self.control_aux:
                ach = torch.cat((ach, observation["aux_achieved"]), dim=1)
                desire = torch.cat((desire, observation["aux_desired"]), dim=1)
            ach_str = str(ach)
            if self._unreal(ach):
                logging.info("Reached Unreal: %s", ach_str)
            if self._unreal(desire):
                logging.info("Unreal Goal: %s", str(desire))

        if self._next_goal_on_success:
            if alf.summary.should_record_summaries():
                alf.summary.scalar(
                    "planner/curr_goal_achieved." + common.exe_mode_name(),
                    torch.mean(goal_achieved.float()))
                alf.summary.scalar(
                    "planner/curr_goal_skipped." + common.exe_mode_name(),
                    torch.mean(curr_goal_skipped.float()))
                reached_all_steps_sum = torch.sum(
                    torch.where(goal_achieved, sg_steps,
                                torch.zeros_like(sg_steps)))
                if reached_all_steps_sum > 0:
                    alf.summary.scalar(
                        "planner/steps_to_reach_all_goals." +
                        common.exe_mode_name(), reached_all_steps_sum /
                        torch.sum(goal_achieved.float()))
                subgoal_achieved = goal_achieved & (
                    current_subgoal_index < self._num_subgoals) & advanced_goal
                reached_sg_steps_sum = torch.sum(
                    torch.where(subgoal_achieved, sg_steps,
                                torch.zeros_like(sg_steps)))
                alf.summary.scalar(
                    "planner/rate_to_reach_sub_goal." + common.exe_mode_name(),
                    torch.mean(subgoal_achieved.float()))
                if reached_sg_steps_sum > 0:
                    alf.summary.scalar(
                        "planner/steps_to_reach_subgoal." +
                        common.exe_mode_name(), reached_sg_steps_sum /
                        torch.sum(subgoal_achieved.float()))
                fgoal_achieved = goal_achieved & (
                    current_subgoal_index >=
                    self._num_subgoals) & advanced_goal
                alf.summary.scalar(
                    "planner/rate_to_reach_final_goal." +
                    common.exe_mode_name(), torch.mean(fgoal_achieved.float()))
                reached_fg_steps_sum = torch.sum(
                    torch.where(fgoal_achieved, sg_steps,
                                torch.zeros_like(sg_steps)))
                if reached_fg_steps_sum > 0:
                    alf.summary.scalar(
                        "planner/steps_to_reach_final_goal." +
                        common.exe_mode_name(), reached_fg_steps_sum /
                        torch.sum(fgoal_achieved.float()))
            if common.is_play():
                if goal_achieved & advanced_goal:
                    logging.info("REACHED GOAL:%d in %d steps at\nachv: %s",
                                 current_subgoal_index, sg_steps, ach_str)
                    if current_subgoal_index[0] < self._num_subgoals:
                        logging.info("New Goal:\ngoal: %s", str(new_goal))
                elif curr_goal_skipped & advanced_goal:
                    logging.info("Skipped Goal:%d in %d steps at\nachv: %s",
                                 current_subgoal_index, sg_steps, ach_str)
                elif ~goal_achieved & (
                        current_subgoal_index == self._num_subgoals + 1):
                    logging.info("Exiting Final Goal at\nachv: %s", ach_str)

        if self._gou_stddev > 0 and epsilon_greedy >= 1:
            new_goal += self._ou_process()

        return new_goal, state

    def summarize_rollout(self, experience):
        """Generate summaries for rollout.

        Args:
            experience (Experience): experience collected from ``rollout_step()``.
        """
        if not alf.summary.should_record_summaries() or not hasattr(
                experience.state,
                "goal_generator") or experience.state.goal_generator == ():
            return
        state = experience.state.goal_generator
        if state.replan != ():
            replan = state.replan
            retain = state.retain_old
            alf.summary.scalar("planner/rate_replan_adopted",
                               torch.mean(replan.float()))
            alf.summary.scalar("planner/rate_replan_rejected",
                               torch.mean(retain.float()))
            alf.summary.scalar("planner/rate_replan_requested",
                               torch.mean(retain | replan).float())
        if state.switched_goal != ():
            alf.summary.scalar(
                "planner/rate_switched_goal." + common.exe_mode_name(),
                torch.mean(state.switched_goal.float()))
        if state.steps_since_last_plan != ():
            alf.summary.scalar("planner/steps_since_last_plan",
                               torch.mean(state.steps_since_last_plan.float()))
            alf.summary.scalar("planner/steps_since_last_goal",
                               torch.mean(state.steps_since_last_goal.float()))
            goal_shape = state.goal.shape
            goal = state.goal.reshape(goal_shape[0] * goal_shape[1], -1)
            alf.summary.scalar(
                "planner/rate_real_goal",
                1. - torch.mean(self._unreal_batch(goal).float()))
        if state.final_goal != ():
            alf.summary.scalar("planner/rate_final_goal",
                               torch.mean(state.final_goal.float()))
        if state.subgoals_index != ():
            for i in range(self._num_subgoals + 3):
                alf.summary.scalar(
                    "planner/rate_subgoal_{}".format(i),
                    torch.mean((state.subgoals_index == i).float()))
                if state.advanced_goal != ():
                    alf.summary.scalar(
                        "planner/rate_advanced_to_subgoal_{}".format(i),
                        torch.mean((state.advanced_goal &
                                    (state.subgoals_index == i)).float()))
