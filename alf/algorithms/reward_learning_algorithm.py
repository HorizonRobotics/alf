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

from collections import namedtuple
from typing import Callable

import gin.tf
import tensorflow as tf

from tf_agents.networks.network import Network
import tf_agents.specs.tensor_spec as tensor_spec

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.data_structures import ActionTimeStep, namedtuple


@gin.configurable
class RewardEstimationAlgorithm(Algorithm):
    """Reward Estimation Module

    This module is responsible to compute/predict rewards
    """

    def __init__(self, name="RewardEstimationAlgorithm"):
        """Create a RewardEstimationAlgorithm.
        """
        super().__init__(name=name)

    def train_step(self, time_step: ActionTimeStep, state):
        """
        Args:
            time_step (ActionTimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        pass

    def calc_loss(self, info):
        loss = tf.nest.map_structure(tf.reduce_mean, info.loss)
        return LossInfo(
            loss=info.loss, scalar_loss=loss.loss, extra=loss.extra)

    def compute_reward(self, obs, action):
        """Compute reward based on the provided observation and action
        Args:
            obs (Tensor): observation
            action (Tensor): action
        Returns:
            reward (Tensor): compuated reward for the given input
        """
        pass


@gin.configurable
class FixedRewardFunction(RewardEstimationAlgorithm):
    """Fixed Reward Estimation Module with hand-crafted computational rules.
    """

    def __init__(self, reward_func: Callable, name="FixedRewardFunction"):
        """Create a FixedRewardFunction.

        Args:
            reward_func (Callable): a function for computing reward. It takes
                as input:
                (1) observation (tf.Tensor of shape [batch_size, observation_dim])
                (2) action (tf.Tensor of shape [batch_size, num_actions])
                and returns a reward Tensor of shape [batch_size]
        """
        super().__init__(name=name)
        self._reward_func = reward_func

    def train_step(self, time_step: ActionTimeStep, state):
        """
        Args:
            time_step (ActionTimeStep): input data for dynamics learning
            state (Tensor): state for dynamics learning (previous observation)
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state (DynamicsState): state for training
                info (DynamicsInfo):
        """
        return AlgorithmStep(outputs=(), state=(), info=())

    def compute_reward(self, obs, action):
        """Compute reward based on current observation and action
        """
        return self._reward_func(obs, action)
