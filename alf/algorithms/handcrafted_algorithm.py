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
import torch

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import AlgStep, LossInfo, TimeStep
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


@alf.configurable
class HandcraftedAlgorithm(OffPolicyAlgorithm):
    """A base class for algorithms with handcrafted computational logic.
    Note that a concrete algorithm should subclass from this and implement the
    computational logic in ``_policy_func``. See ``SimpleCarlaAlgorithm`` for
    an example.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 env=None,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="Handcrafted"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
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
            reward_spec=reward_spec,
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

    def predict_step(self, inputs: TimeStep, state):
        action = self._predict_action(inputs.observation, state=state)
        return AlgStep(output=action, state=state)

    def rollout_step(self, inputs: TimeStep, state):
        action = self._predict_action(inputs.observation, state=state)
        return AlgStep(output=action, state=state)

    def train_step(self, inputs: TimeStep, state, rollout_info):
        return AlgStep()

    def calc_loss(self, info):
        return LossInfo()


@alf.configurable
class SimpleCarlaAlgorithm(HandcraftedAlgorithm):
    """A simple controller for Carla environment.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 distance_to_decelerate=50.0,
                 distance_to_stop=1.0,
                 env=None,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="SimpleCarlaAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            distance_to_decelerate (float|int): the distance in meter to goal
                from which to start decreasing the speed
            distance_to_stop (float|int): the distance in meter to goal
                from which to start to make a stop
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
            reward_spec=reward_spec,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        self._distance_to_decelerate = distance_to_decelerate
        self._distance_to_stop = distance_to_stop

    def _policy_func(self, observation):
        """A naive hand-crafted policy for Carla environment.

        Args:
            observation (nested Tensor): input observation that is compatible
                with observation_spec
        Returns
            Tensor: action computed based on the observation
        """
        # waypoints is a [B, k, 3] shaped tensor, which contains the batched
        # positions of a number of future waypoints in the route, relative to
        # the coordinate system of the respective vehicle. waypoints[:, 0]
        # is the closest waypoint and waypoints[:, -1] is the farthest one.
        # Each waypoint has 3 elements corresponding to the x, y, z values
        # relative to the vehicle's coordinate system.
        waypoints = alf.nest.get_field(observation, 'observation.navigation')

        # goal is a [B, 3] shaped tensor, with each 3D vector contains the
        # x, y, z positions of the goal, relative to  to the vehicle's
        # coordinate system.
        goal = alf.nest.get_field(observation, 'observation.goal')

        if waypoints.shape[1] > 1:
            wp_vector = waypoints[:, 1]
        else:
            wp_vector = waypoints[:, 0]

        direction = torch.atan2(wp_vector[..., 1], wp_vector[..., 0])
        direction = direction / np.pi

        # action is a [B, 3] tensor with each 3D vector corresponding to
        # [speed, direction, reverse].
        # speed: 1.0 corresponding to maximally allowed speed
        # direction: relative to the vehicle's heading, with 0 being front,
        # -0.5 being left and 0.5 being right
        # reverse: values greater than 0.5 corresponding to going backward.

        action = torch.zeros(waypoints.shape[0], self._action_spec.shape[0])

        distance_to_goal = torch.norm(goal, dim=1)

        # here we adjust the speed based on the distance to goal
        action[distance_to_goal > self._distance_to_decelerate, 0] = 1
        ind = (distance_to_goal > self._distance_to_stop) & (
            distance_to_goal <= self._distance_to_decelerate)
        action[ind, 0] = distance_to_goal[ind] / self._distance_to_decelerate
        action[distance_to_goal <= self._distance_to_stop, 0] = 0

        # direction is computed based on the waypoint
        action[:, 1] = direction

        return action
