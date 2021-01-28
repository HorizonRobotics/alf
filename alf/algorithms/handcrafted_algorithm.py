# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Handcrafted Algorithm."""

import numpy as np
import gin
import torch

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import AlgStep, Experience, LossInfo, TimeStep
from alf.tensor_specs import BoundedTensorSpec


@gin.configurable
class HandcraftedAlgorithm(OffPolicyAlgorithm):
    """An Algorithm with handcrafted computational logic.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 env=None,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="Handcrafted"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=(),
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

    def _policy_func(self, observation):
        """A function calculating action based on the input observation.
        Each subclass needs to define this function

        Args:
            observation (nested Tensor): input observation that is compatible
                with observation_spec
        Returns:
            nested Tensor: action that is compatible with action spec
        """
        raise NotImplementedError('Must define _policy_func member '
                                  'function for the class')

    def _predict_action(self, observation, state):
        """Predict action based on observation
        """
        return self._policy_func(observation)

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy=1.0):
        action = self._predict_action(time_step.observation, state=state)
        return AlgStep(output=action, state=state)

    def rollout_step(self, time_step: TimeStep, state):
        action = self._predict_action(time_step.observation, state=state)
        return AlgStep(output=action, state=state)

    def train_step(self, exp: Experience, state):
        return AlgStep()

    def calc_loss(self, experience, train_info):
        return LossInfo()


@gin.configurable
class SimpleCarlaAlgorithm(HandcraftedAlgorithm):
    """A simple controller for Carla environment.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 env=None,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="SimpleCarlaAlgorithm"):

        super().__init__(
            observation_spec,
            action_spec,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

    def _policy_func(self, observation):
        """A naive hand-crafted policy for Carla environment.

        Args:
            observation (nested Tensor): input observation that is compatible
                with observation_spec
        Returns
            Tensor: action computed based on the observation
        """
        nav = alf.nest.get_field(observation, 'observation.navigation')
        goal = alf.nest.get_field(observation, 'observation.goal')

        waypoints = nav
        if waypoints.shape[-1] > 1:
            wp_vector = waypoints[:, 1]
        else:
            wp_vector = waypoints[:, 0]

        direction = torch.atan2(wp_vector[..., 1], wp_vector[..., 0])
        direction = direction / np.pi

        action = torch.zeros(waypoints.shape[0], self._action_spec.shape[0])
        action[:, 1] = direction

        distance_to_goal = torch.norm(goal)

        if distance_to_goal > 50:
            action[:, 0] = 1
        elif distance_to_goal > 1 and distance_to_goal < 50:
            action[:, 0] = distance_to_goal / 50.0
        else:
            action[:, 0] = 0

        return action
