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

from typing import Callable

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, TimeStep


@alf.configurable
class RewardEstimationAlgorithm(Algorithm):
    """Reward Estimation Module

    This module is responsible for computing/predicting rewards
    """

    def __init__(self, name="RewardEstimationAlgorithm"):
        """Create a RewardEstimationAlgorithm.
        """
        super().__init__(train_state_spec=(), name=name)

    def train_step(self, time_step: TimeStep, state, rollout_info=None):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            AlgStep
        """
        pass

    def compute_reward(self, obs, action, state):
        """Compute reward based on the provided observation and action
        Args:
            obs (Tensor): observation
            action (Tensor): action
            state ()
        Returns:
            reward (Tensor): compuated reward for the given input
        """
        pass


@alf.configurable
class FixedRewardFunction(RewardEstimationAlgorithm):
    """Fixed Reward Estimation Module with hand-crafted computational rules.
    """

    def __init__(self, reward_func: Callable, name="FixedRewardFunction"):
        """

        Args:
            reward_func (Callable): a function for computing reward.
                It takes as input:

                (1) observation (Tensor of shape [batch_size, observation_dim])
                (2) action (Tensor of shape [batch_size, num_actions])
                    and returns a reward Tensor of shape [batch_size]
        """
        super().__init__(name=name)
        self._reward_func = reward_func

    def train_step(self, time_step: TimeStep, state=(), rollout_info=None):
        """
        Args:
            time_step (TimeStep): input data for dynamics learning
            state: state for reward learning
        Returns:
            AlgStep
        """
        return AlgStep(output=(), state=state, info=())

    def compute_reward(self, obs, action, state):
        """Compute reward based on current observation and action
        Args:
            obs (Tensor): observation
            action (Tensor): action
            state: state for reward calculation
        Returns:
            reward (Tensor): compuated reward for the given input
            state: updated state, currently simply passing the input state
        """
        return self._reward_func(obs, action), state
