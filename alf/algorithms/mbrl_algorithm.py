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
"""Model-based RL Algorithm."""

from functools import partial
import numpy as np
import gin

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import (AlgStep, Experience, LossInfo, namedtuple,
                                 TimeStep)
from alf.nest import nest
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, tensor_utils
from alf.utils.math_ops import add_ignore_empty

from alf.algorithms.dynamics_learning_algorithm import DynamicsLearningAlgorithm
from alf.algorithms.reward_learning_algorithm import RewardEstimationAlgorithm
from alf.algorithms.planning_algorithm import PlanAlgorithm

MbrlState = namedtuple("MbrlState", ["dynamics", "reward", "planner"])
MbrlInfo = namedtuple(
    "MbrlInfo", ["dynamics", "reward", "planner"], default_value=())


@gin.configurable
class MbrlAlgorithm(OffPolicyAlgorithm):
    """Model-based RL algorithm
    """

    def __init__(self,
                 observation_spec,
                 feature_spec,
                 action_spec,
                 dynamics_module: DynamicsLearningAlgorithm,
                 reward_module: RewardEstimationAlgorithm,
                 planner_module: PlanAlgorithm,
                 particles_per_replica=1,
                 env=None,
                 config: TrainerConfig = None,
                 dynamics_optimizer=None,
                 reward_optimizer=None,
                 planner_optimizer=None,
                 debug_summaries=False,
                 name="MbrlAlgorithm"):
        """Create an MbrlAlgorithm.
        The MbrlAlgorithm takes as input the following set of modules for
        making decisions on actions based on the current observation:
        1) learnable/fixed dynamics module
        2) learnable/fixed reward module
        3) learnable/fixed planner module

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            dynamics_module (DynamicsLearningAlgorithm): module for learning to
                predict the next feature based on the previous feature and action.
                It should accept input with spec [feature_spec,
                encoded_action_spec] and output a tensor of shape
                feature_spec. For discrete action, encoded_action is an one-hot
                representation of the action. For continuous action, encoded
                action is same as the original action.
            reward_module (RewardEstimationAlgorithm): module for calculating
                the reward, i.e.,  evaluating the reward for a (s, a) pair
            planner_module (PlanAlgorithm): module for generating planned action
                based on specified reward function and dynamics function
            particles_per_replica (int): number of particles for each replica
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs `train_iter()` by
                itself.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.

        """
        train_state_spec = MbrlState(
            dynamics=dynamics_module.train_state_spec,
            reward=reward_module.train_state_spec,
            planner=planner_module.train_state_spec)

        super().__init__(
            feature_spec,
            action_spec,
            train_state_spec=train_state_spec,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        flat_action_spec = nest.flatten(action_spec)
        action_spec = flat_action_spec[0]

        assert action_spec.is_continuous, "only support \
                                                    continious control"

        num_actions = action_spec.shape[-1]

        flat_feature_spec = nest.flatten(feature_spec)
        assert len(flat_feature_spec) == 1, "Mbrl doesn't support nested \
                                             feature_spec"

        self._action_spec = action_spec
        self._num_actions = num_actions

        if dynamics_optimizer is not None:
            self.add_optimizer(dynamics_optimizer, [dynamics_module])

        if planner_optimizer is not None:
            self.add_optimizer(planner_optimizer, [planner_module])

        if reward_optimizer is not None:
            self.add_optimizer(reward_optimizer, [reward_module])

        self._dynamics_module = dynamics_module
        self._reward_module = reward_module
        self._planner_module = planner_module
        self._planner_module.set_action_sequence_cost_func(
            self._predict_multi_step_cost)
        if dynamics_module is not None:
            self._num_dynamics_replicas = dynamics_module.num_replicas
        self._particles_per_replica = particles_per_replica

    def _predict_next_step(self, time_step, dynamics_state):
        """Predict the next step (observation and state) based on the current
            time step and state
        Args:
            time_step (TimeStep): input data for next step prediction
            dynamics_state: input dynamics state next step prediction
        Returns:
            next_time_step (TimeStep): updated time_step with observation
                predicted from the dynamics module
            next_dynamic_state: updated dynamics state from the dynamics module
        """
        with torch.no_grad():
            dynamics_step = self._dynamics_module.predict_step(
                time_step, dynamics_state)
            pred_obs = dynamics_step.output
            next_time_step = time_step._replace(observation=pred_obs)
            next_dynamic_state = dynamics_step.state
        return next_time_step, next_dynamic_state

    def _expand_to_population(self, data, population_size):
        """Expand the input tensor to a population of replications
        Args:
            data (Tensor): input data with shape [batch_size, ...]
        Returns:
            data_population (Tensor) with shape
                                    [batch_size * population_size, ...].
            For example data tensor [[a, b], [c, d]] and a population_size of 2,
            we have the following data_population tensor as output
                                    [[a, b], [a, b], [c, d], [c, d]]
        """
        data_population = torch.repeat_interleave(data, population_size, dim=0)
        return data_population

    def _expand_to_particles(self, inputs):
        """Expand the inputs of shape [B, ...] to [B*p, n, ...] if n > 1,
            or to [B*p, ...] if n = 1, where n is the number of replicas
            and p is the number of particles per replica.
        """
        # [B, ...] -> [B*p, ...]
        inputs = torch.repeat_interleave(
            inputs, self._particles_per_replica, dim=0)
        if self._num_dynamics_replicas > 1:
            # [B*p, ...] -> [B*p, n, ...]
            inputs = inputs.unsqueeze(1).expand(
                -1, self._num_dynamics_replicas, *inputs.shape[1:])

        return inputs

    @torch.no_grad()
    def _predict_multi_step_cost(self, observation, actions):
        """Compute the total cost by unrolling multiple steps according to
            the given initial observation and multi-step actions.
        Args:
            observation: the current observation for predicting quantities of
                future time steps
            actions (Tensor): a set of action sequences to
                shape [B, population, unroll_steps, action_dim]
        Returns:
            cost (Tensor): negation of accumulated predicted reward, with
                the shape of [B, population]
        """
        batch_size, population_size, num_unroll_steps = actions.shape[0:3]

        state = self.get_initial_predict_state(batch_size)
        time_step = TimeStep()
        dyn_state = state.dynamics._replace(feature=observation)
        dyn_state = nest.map_structure(
            partial(
                self._expand_to_population, population_size=population_size),
            dyn_state)

        # expand to particles
        dyn_state = nest.map_structure(self._expand_to_particles, dyn_state)
        reward_state = state.reward
        reward = 0
        for i in range(num_unroll_steps):
            action = actions[:, :, i, ...].view(-1, actions.shape[3])
            action = self._expand_to_particles(action)
            time_step = time_step._replace(prev_action=action)
            time_step, dyn_state = self._predict_next_step(
                time_step, dyn_state)
            next_obs = time_step.observation
            # Note: currently using (next_obs, action), might need to
            # consider (obs, action) in order to be more compatible
            # with the conventional definition of the reward function
            reward_step, reward_state = self._calc_step_reward(
                next_obs, action, reward_state)
            reward = reward + reward_step
        cost = -reward
        # reshape cost
        # [B*par, n] -> [B, par*n]
        cost = cost.reshape(
            -1, self._particles_per_replica * self._num_dynamics_replicas)
        cost = cost.mean(-1)

        # reshape cost back to [batch size, population_size]
        cost = torch.reshape(cost, [batch_size, -1])

        return cost

    def _calc_step_reward(self, obs, action, reward_state):
        """Calculate the step reward based on the given observation, action
            and state.
        Args:
            obs (Tensor): observation
            action (Tensor): action
            state: state for reward calculation
        Returns:
            reward (Tensor): compuated reward for the given input
            updated_state: updated state from the reward module
        """
        reward, reward_state = self._reward_module.compute_reward(
            obs, action, reward_state)
        return reward, reward_state

    def _predict_with_planning(self, time_step: TimeStep, state: MbrlState,
                               epsilon_greedy):

        action, planner_state = self._planner_module.predict_plan(
            time_step, state.planner, epsilon_greedy)

        dynamics_state = self._dynamics_module.update_state(
            time_step, state.dynamics)

        return AlgStep(
            output=action,
            state=state._replace(
                dynamics=dynamics_state, planner=planner_state),
            info=MbrlInfo())

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy=0.0):
        return self._predict_with_planning(time_step, state, epsilon_greedy)

    def rollout_step(self, time_step: TimeStep, state):
        # note epsilon_greedy
        # 0.1 for random exploration
        return self._predict_with_planning(
            time_step, state, epsilon_greedy=0.0)

    def train_step(self, exp: Experience, state: MbrlState):
        action = exp.action
        dynamics_step = self._dynamics_module.train_step(exp, state.dynamics)
        reward_step = self._reward_module.train_step(exp, state.reward)
        plan_step = self._planner_module.train_step(exp, state.planner)
        state = MbrlState(
            dynamics=dynamics_step.state,
            reward=reward_step.state,
            planner=plan_step.state)
        info = MbrlInfo(
            dynamics=dynamics_step.info,
            reward=reward_step.info,
            planner=plan_step.info)
        return AlgStep(action, state, info)

    def calc_loss(self, experience, training_info: MbrlInfo):
        loss_dynamics = self._dynamics_module.calc_loss(training_info.dynamics)
        loss = loss_dynamics.loss
        loss = add_ignore_empty(loss, training_info.reward)
        loss = add_ignore_empty(loss, training_info.planner)
        return LossInfo(loss=loss, scalar_loss=loss_dynamics.scalar_loss)

    def after_update(self, experience, training_info):
        self._planner_module.after_update(
            training_info._replace(planner=training_info.planner))
