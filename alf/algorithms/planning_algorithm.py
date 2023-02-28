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
import torch.distributions as td
import einops

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import alf
from alf.algorithms.mcts_algorithm import MCTSState
from alf.algorithms.mcts_models import MCTSModel, ModelOutput
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import AlgStep, namedtuple, TimeStep, StepType
from alf.nest import nest
from alf.optimizers.traj_optimizers import RandomOptimizer, CEMOptimizer
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils import common, summary_utils

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
                                                    continious control"

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
            population_size (int): the size of polulation for random shooting
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
            population_size (int): the size of polulation for optimization
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


class MPPIState(NamedTuple):
    traj_means: torch.Tensor = torch.tensor([])  # [B, H, ...]
    pred_state: Any = ()
    next_predicted_reward: Any = ()

    # ---- Committing Mechanism ----

    # Here, committed_steps[i] == 6 means that after taking the current decided
    # steps, there are still 6 steps need to be committed, and it == 0 means a
    # new decision need to be made for taking the next action.
    committed_steps: Any = ()  # [B,]
    committed_actions: Any = ()  # [B, H, ...]

    def inherit_traj_distribution(self,
                                  is_first: torch.Tensor,
                                  bounds: Tuple[torch.Tensor, torch.Tensor],
                                  desired_horizon: Optional[int] = None):
        # [B, H, ...]
        new_means = einops.repeat(
            (bounds[0] + bounds[1]) / 2.0,
            "... -> B H ...",
            B=self.traj_means.shape[0],
            H=self.traj_means.shape[1]).clone()
        new_stds = einops.repeat(
            (bounds[1] - bounds[0]) / 2.0,
            "... -> B H ...",
            B=self.traj_means.shape[0],
            H=self.traj_means.shape[1])

        is_first = einops.rearrange(
            is_first, "B -> B" + (" ()" * (self.traj_means.ndim - 1)))

        new_means[:, :-1] = torch.where(is_first, new_means[:, :-1],
                                        self.traj_means[:, 1:])

        if desired_horizon is not None:
            assert desired_horizon <= self.traj_means.shape[1]
            new_means = new_means[:, :desired_horizon]
            new_stds = new_stds[:, :desired_horizon]

        return new_means, new_stds


class MPPIInfo(NamedTuple):
    candidate_actions: Any = ()
    value: Any = ()
    candidate_action_policy: Any = ()
    candidate_prior: Any = ()
    candidate_advantage: Any = ()


@alf.configurable
class MPPIPlanner(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 discount: float,
                 reward_spec=TensorSpec(()),
                 action_margin: float = 0.02,
                 horizon: int = 10,
                 num_elites: int = 64,
                 num_trajs: int = 512,
                 num_iters: int = 6,
                 momentum: float = 0.99,
                 temperature: float = 0.05,
                 policy_guided_ratio: float = 0.05,
                 use_elite_as_policy_target: bool = False,
                 weighted_policy_target: bool = False,
                 min_std: float = 0.05,
                 model: Optional[MCTSModel] = None,
                 predict_without_planning: bool = False,
                 rollout_commit_prob: float = 0.0,
                 use_value: bool = True,
                 use_reward: bool = True,
                 debug_summaries: bool = False,
                 name: str = "MPPIAlgorithm"):
        assert reward_spec.shape == (), "Only scalar reward is supported"
        state_spec = MPPIState(
            traj_means=action_spec.replace(
                shape=(horizon, *action_spec.shape)),
            committed_steps=TensorSpec((), dtype=torch.int64),
            committed_actions=action_spec.replace(
                shape=(horizon, *action_spec.shape)),
            next_predicted_reward=TensorSpec(()))

        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            predict_state_spec=state_spec,
            rollout_state_spec=state_spec,
            train_state_spec=state_spec,
            debug_summaries=debug_summaries,
            name=name)

        # Compute the bounds of the action spec, broadcasted.
        self._action_bounds = (
            torch.broadcast_to(
                torch.tensor(action_spec.minimum) + action_margin,
                action_spec.shape),
            torch.broadcast_to(
                torch.tensor(action_spec.maximum) - action_margin,
                action_spec.shape))

        self._model = model
        self._discount = discount
        self._H = horizon
        self._num_elites = num_elites
        self._num_iters = num_iters
        self._num_guided_trajs = int(policy_guided_ratio * num_trajs)
        self._num_random_trajs = num_trajs - self._num_guided_trajs
        self._use_elite_as_policy_target = use_elite_as_policy_target
        self._weighted_policy_target = weighted_policy_target
        self._min_std = min_std
        self._momentum = momentum
        self._temperature = temperature
        self._predict_without_planning = predict_without_planning
        self._rollout_commit_prob = rollout_commit_prob
        self._use_value = use_value
        self._use_reward = use_reward

    def set_model(self, model: MCTSModel):
        self._model = model

    @torch.no_grad()
    def debug_print(self, actions: torch.Tensor, returns: torch.Tensor,
                    topk: int, initial: ModelOutput):
        assert self._model is not None
        B, H = actions.shape[1:3]
        assert B == 1

        # [j, H, ...]
        actions = actions[:, 0]
        # [j]
        returns = returns[:, 0]

        # [topk]
        indices = torch.topk(returns, topk).indices
        selected_returns = returns[indices]
        selected_actions = actions[indices]

        def _repeat(x):
            return einops.repeat(x, "B ... -> (j B) ...", j=topk)

        state = nest.map_structure(_repeat, initial.state)

        # [topk]
        g = torch.zeros_like(selected_returns)
        reward = torch.zeros(topk, H)
        discount = self._discount
        current = initial
        for h in range(H):
            current = self._model.recurrent_inference(
                state=state, action=selected_actions[:, h])
            state = current.state
            if self._use_reward:
                g = g + discount * current.reward
            reward[:, h] = current.reward
            discount *= self._discount

        if self._use_value:
            g = g + discount * current.value

        for k in range(topk):
            print(f"===== Top {k} =====")
            for h in range(H):
                print(f"  action = {selected_actions[k, h].cpu().numpy()}"
                      f"  reward = {reward[k, h].item():.4f}")
            print(
                f"return = {selected_returns[k].item()}, recomputed = {g[k].item()}"
            )

    @torch.no_grad()
    def _sample_trajs_from_policy(
            self,
            num_trajs: int,
            H: int,
            initial: ModelOutput,
            first_action: Optional[torch.Tensor] = None,  # [num_trajs, B, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample trajectories using the MCTS Model.

        Returs the action sequence and the reward sequence.
        """
        assert self._model is not None, (
            "Need to call `set_model` before `predict_step` or `rollout_step`")

        def _repeat(x):
            return einops.repeat(x, "B ... -> (j B) ...", j=num_trajs)

        state = nest.map_structure(_repeat, initial.state)
        actions = []
        # [B * num_trajs]
        g = einops.repeat(
            torch.zeros_like(initial.value), "B -> (j B)", j=num_trajs)
        current = initial
        discount = self._discount

        for h in range(H):
            # [num_trajs, B, ...]
            if h == 0:
                if first_action is None:
                    action = current.action_distribution.sample((num_trajs, ))
                    actions.append(action)
                else:
                    action = first_action
                action = einops.repeat(action, "j B ... -> (j B) ...")

            else:
                action = current.action_distribution.sample()
                actions.append(
                    einops.repeat(action, "(j B) ... -> j B ...", j=num_trajs))

            current = self._model.recurrent_inference(
                state=state, action=action)
            state = current.state
            if self._use_reward:
                g = g + discount * current.reward
            discount *= self._discount
        if self._use_value:
            # [B * num_trajs]
            g = g + discount * current.value

        # [j, B, H, ...]
        actions = torch.stack(actions, dim=2)
        returns = einops.rearrange(g, "(j B) -> j B", j=num_trajs)
        return actions, returns

    @torch.no_grad()
    def _evaluate_returns_of_trajs(
            self,
            action_sequence: torch.Tensor,
            initial: ModelOutput,
    ) -> torch.Tensor:
        """Roll out the trajectory to get the imaginary returns.

        Args:

            action_sequence: Of shape [j, B, H, ...].

        Returns:

            The returns of all the trajectories for all the batches.
            Shape should be [j, B].

        """
        assert self._model is not None, (
            "Need to call `set_model` before `predict_step` or `rollout_step`")

        num_trajs, _, H = action_sequence.shape[:3]
        actions = einops.rearrange(action_sequence, "j B H ... -> (j B) H ...")

        def _repeat(x):
            return einops.repeat(x, "B ... -> (j B) ...", j=num_trajs)

        state = nest.map_structure(_repeat, initial.state)
        g = einops.repeat(
            torch.zeros_like(initial.value), "B -> (j B)", j=num_trajs)
        current = initial
        discount = 1.0
        for h in range(H):
            current = self._model.recurrent_inference(
                state=state, action=actions[:, h])
            state = current.state
            g = g + discount * current.reward
            discount *= self._discount
        g = g + discount * current.value

        actions = einops.rearrange(
            actions, "(j B) H ... -> j B H ...", j=num_trajs)
        return einops.rearrange(g, "(j B) -> j B", j=num_trajs)

    @torch.no_grad()
    def _sample_trajs_from_distribution(self,
                                        traj_means: torch.Tensor,
                                        traj_stds: torch.Tensor,
                                        num_trajs: int,
                                        initial: ModelOutput,
                                        bias: Optional[torch.Tensor] = None
                                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._model is not None, (
            "Need to call `set_model` before `predict_step` or `rollout_step`")

        # Shape of the sampled actions will be [j, B, H, ...]
        actions = torch.randn(num_trajs, *traj_means.shape)
        if bias is not None:
            actions = actions + bias

        actions = torch.clamp(actions * traj_stds + traj_means,
                              *self._action_bounds)
        returns = self._evaluate_returns_of_trajs(
            action_sequence=actions, initial=initial)

        return actions, returns

    @torch.no_grad()
    def _sample_trajs_from_high_level_policy(
            self,
            num_trajs: int,
            initial: ModelOutput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = initial.action_distribution.sample((num_trajs, ))
        actions = einops.rearrange(
            actions,
            "j B (H a) ... -> j B H a ...",
            a=self._action_spec.shape[0])
        actions = torch.clamp(actions, *self._action_bounds)
        returns = self._evaluate_returns_of_trajs(actions, initial)
        return actions, returns

    @torch.no_grad()
    def _step(self,
              time_step: TimeStep,
              state: MPPIState,
              num_random: int,
              num_guided: int = 0,
              exploration_mode: bool = True,
              log_metrics: bool = False,
              force_commit: bool = False):
        assert self._model is not None, (
            "Need to call `set_model` before `predict_step` or `rollout_step`")

        B = time_step.observation.shape[0]
        b = torch.arange(B)

        initial = self._model.initial_predict(time_step.observation,
                                              state.pred_state)
        policy_is_high_level = (initial.action_distribution.event_shape !=
                                self._action_spec.shape)

        # Sample guided trajectories from the policy
        guided_actions = None
        guided_returns = None
        if num_guided > 0:
            if policy_is_high_level:
                # [num_guided, B, H, ...]
                guided_actions, guided_returns = self._sample_trajs_from_high_level_policy(
                    num_trajs=num_guided, initial=initial)
            else:
                # [num_guided, B, H, ...]
                guided_actions, guided_returns = self._sample_trajs_from_policy(
                    num_trajs=num_guided, H=self._H, initial=initial)

        # The CEM iterations
        traj_means, traj_stds = state.inherit_traj_distribution(
            time_step.is_first(),
            bounds=self._action_bounds,
            desired_horizon=self._H)
        ndims_traj = traj_means.ndim - 1
        for i in range(self._num_iters):
            # 1. Sample from the trajectory distribution
            # [num_random, B, H, ...]
            actions, returns = self._sample_trajs_from_distribution(
                traj_means=traj_means,
                traj_stds=traj_stds,
                num_trajs=num_random,
                initial=initial)

            # 2. Append the guided trajectories when needed
            if num_guided > 0:
                assert guided_actions is not None and guided_returns is not None
                actions = torch.cat([actions, guided_actions], dim=0)
                returns = torch.cat([returns, guided_returns], dim=0)

            # 3. Rank and pick the topK elite trajectories and compute the
            #    weight based on their returns.

            # [num_elites, B]
            temperature = self._temperature() if callable(
                self._temperature) else self._temperature
            elites = torch.topk(returns, self._num_elites, dim=0).indices
            elite_returns = returns[elites, b]  # [num_elite, B]
            best_return = torch.amax(elite_returns, dim=0)  # [B]
            scores = torch.exp(temperature * (elite_returns - best_return))
            # [num_elite, B]
            weights = scores / scores.sum(dim=0)
            w = weights.view(*weights.shape, *([1] * ndims_traj))

            # 4. Update the Gaussian trajectory distribution
            elite_actions = actions[elites, b]  # [num_elite, B, H, ...]
            new_means = torch.sum(elite_actions * w, dim=0)
            new_vars = torch.sum(
                torch.square(elite_actions - new_means) * w, dim=0)
            traj_stds = torch.sqrt(new_vars).clamp_(
                torch.tensor(self._min_std),
                self._action_bounds[1] - self._action_bounds[0])
            traj_means = self._momentum * traj_means + (
                1.0 - self._momentum) * new_means

            if log_metrics:
                with alf.summary.scope(self._name):
                    summary_utils.add_mean_hist_summary(
                        f"iter{i}/traj_means/0", traj_means[:, 0])
                    summary_utils.add_mean_hist_summary(
                        f"iter{i}/traj_means/1+", traj_means[:, 1:])
                    summary_utils.add_mean_hist_summary(
                        f"iter{i}/traj_stds/0", traj_stds[:, 0])
                    summary_utils.add_mean_hist_summary(
                        f"iter{i}/traj_stds/1+", traj_stds[:, 1:])
                    summary_utils.add_mean_hist_summary(
                        f"iter{i}/returns", returns)
                    summary_utils.add_mean_hist_summary(
                        f"iter{i}/elite_returns", elite_returns)

        # self.debug_print(actions=actions, returns=returns, topk=7, initial=initial)

        num_candidates = num_guided + num_random

        def _generate_samples(means, stds, j=num_candidates):
            actions = einops.repeat(means, "B ... -> j B ...", j=j)
            actions = actions + stds * torch.randn_like(actions)
            actions = actions.clamp(*self._action_bounds)
            return actions

        if policy_is_high_level:
            # [j, B, H, ...]
            if self._use_elite_as_policy_target:
                candidate_actions = elite_actions  # [num_elite, B, H, ...]
            else:
                candidate_actions = _generate_samples(traj_means, traj_stds)
            candidate_actions = einops.rearrange(
                candidate_actions, "j B H a ... -> j B (H a) ...")
        else:
            # [j, B, ...]
            candidate_actions = _generate_samples(traj_means[:, 0],
                                                  traj_stds[:, 0])

        if self._use_elite_as_policy_target and self._weighted_policy_target:
            candidate_action_policy = weights
        else:
            # Assign uniform weights to the candidate actions
            candidate_action_policy = torch.ones(
                candidate_actions.shape[:2],
                device=candidate_actions.device) / candidate_actions.shape[0]

        info = nest.map_structure(
            lambda x: einops.rearrange(x, "j B ... -> B j ..."),
            MPPIInfo(
                candidate_actions=candidate_actions,  # [j, B, ...]
                value=(),
                candidate_action_policy=candidate_action_policy,
                candidate_prior=candidate_action_policy,
                candidate_advantage=returns - initial.value))

        info = info._replace(value=elite_returns.mean(dim=0))

        if (common.is_rollout() and self._rollout_commit_prob > 0.0) or force_commit:
            # Handle the commitment of actions. Note that this will not be
            # executed for reanalyze because of ``common.is_rollout()``.
            committed_steps = state.committed_steps.clone()
            committed_steps[time_step.is_first()] = 0
        else:
            committed_steps = torch.zeros_like(state.committed_steps)

        fresh = committed_steps == 0

        committed_actions = state.committed_actions.clone()
        if exploration_mode and common.is_rollout():
            committed_actions[fresh] = _generate_samples(
                traj_means[fresh], traj_stds[fresh], 1)[0]
        else:
            committed_actions[fresh] = traj_means[fresh]

        final_action = committed_actions[b, -committed_steps]
        committed_steps[~fresh] = committed_steps[~fresh] - 1

        if self._rollout_commit_prob > 0.0 or force_commit:
            start_commit = fresh if force_commit else torch.logical_and(
                fresh,
                torch.rand_like(time_step.discount) <
                self._rollout_commit_prob)
            committed_steps[start_commit] = self._H - 1

        if log_metrics:
            with alf.summary.scope(self._name):
                curr_reward = time_step.reward  # [B]
                curr_predicted_reward = initial.reward  #[B]

                alf.summary.scalar(
                    'reward',
                    time_step.reward.mean(),
                    average_over_summary_interval=True)
                alf.summary.scalar(
                    'reward_last_predicted/value',
                    state.next_predicted_reward.mean(),
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

                _summarize_error("reward_predicted", curr_reward,
                                 curr_predicted_reward)
                _summarize_error("reward_next_predicted", curr_reward,
                                 state.next_predicted_reward)

                # How many of guided policies elected as elites?
                if num_guided > 0:
                    summary_utils.add_mean_hist_summary(
                        "guided_elite_ratio",
                        (elites >= num_random).sum(dim=0) / num_guided)

                # Summarize the fresh and continued steps
                alf.summary.scalar("commit_ratio", 1.0 - fresh.sum() / B)

        next_prediction = self._model.recurrent_inference(
            state=initial.state, action=final_action)

        return AlgStep(
            output=final_action,
            state=MPPIState(
                traj_means=traj_means,
                # Note that we do not keep the model state here.
                pred_state=(),
                next_predicted_reward=next_prediction.reward,
                committed_steps=committed_steps,
                committed_actions=committed_actions),
            info=info)

    def predict_step(self, time_step: TimeStep, state: MPPIState):
        if self._predict_without_planning:
            initial = self._model.initial_predict(time_step.observation,
                                                  state.pred_state)
            action = initial.action_distribution.sample()
            traj_means, _ = state.inherit_traj_distribution(
                time_step.is_first(),
                bounds=self._action_bounds,
                desired_horizon=self._H)
            return AlgStep(
                output=action,
                state=MPPIState(traj_means=traj_means,
                                pred_state=(),
                                next_predicted_reward=state.next_predicted_reward,
                                committed_steps=state.committed_steps,
                                committed_actions=state.committed_actions),
                info=())

        return self._step(
            time_step,
            state,
            num_random=self._num_random_trajs,
            num_guided=self._num_guided_trajs,
            exploration_mode=False,
            force_commit=False)

    def rollout_step(self, time_step: TimeStep, state: MPPIState):
        return self._step(
            time_step,
            state,
            num_random=self._num_random_trajs,
            num_guided=self._num_guided_trajs,
            exploration_mode=True,
            log_metrics=True)
