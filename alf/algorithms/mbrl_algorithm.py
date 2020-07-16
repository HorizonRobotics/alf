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
            planner_module (PLANAlgorithm): module for generating planned action
                based on specified reward function and dynamics function
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
        self._planner_module.set_reward_func(self._calc_step_reward)
        self._planner_module.set_dynamics_func(self._predict_next_step)

    def _predict_next_step(self, time_step, state):
        """Predict the next step (observation and state) based on the current
            time step and state
        Args:
            time_step (TimeStep): input data for next step prediction
            state (MbrlState): input state next step prediction
        Returns:
            next_time_step (TimeStep): updated time_step with observation
                predicted from the dynamics module
            next_state (MbrlState): updated state from the dynamics module
        """
        with torch.no_grad():
            dynamics_step = self._dynamics_module.predict_step(
                time_step, state.dynamics)
            pred_obs = dynamics_step.output
            next_time_step = time_step._replace(observation=pred_obs)
            next_state = state._replace(dynamics=dynamics_step.state)
        return next_time_step, next_state

    def _calc_step_reward(self, obs, action):
        reward = self._reward_module.compute_reward(obs, action)
        return reward

    def _predict_with_planning(self, time_step: TimeStep, state,
                               epsilon_greedy):
        # full state in
        action, mbrl_state = self._planner_module.generate_plan(
            time_step, state, epsilon_greedy)
        dynamics_state = self._dynamics_module.update_state(
            time_step, state.dynamics)

        return AlgStep(
            output=action,
            state=mbrl_state._replace(dynamics=dynamics_state),
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
        loss = training_info.dynamics.loss

        loss = add_ignore_empty(loss, training_info.reward)
        loss = add_ignore_empty(loss, training_info.planner)
        return LossInfo(loss=loss.loss, extra=(loss.loss))

    def after_update(self, experience, training_info):
        self._planner_module.after_update(
            training_info._replace(planner=training_info.planner))
