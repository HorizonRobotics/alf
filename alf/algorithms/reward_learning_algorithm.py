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

import gin.tf
import tensorflow as tf

from tf_agents.networks.network import Network
import tf_agents.specs.tensor_spec as tensor_spec

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.data_structures import ActionTimeStep, namedtuple


@gin.configurable
class REAlgorithm(Algorithm):
    """Reward Estimation Module

    This module is responsible to compute/predict rewards
    """

    def __init__(self,
                 feature_spec,
                 action_spec,
                 hidden_size=256,
                 reward_network: Network = None,
                 name="RewardEstimationAlgorithm"):
        """Create a RewardEstimationAlgorithm.

        Args:
            hidden_size (int|tuple): size of hidden layer(s)
            reward_network (Network): network for predicting reward
                based on the current feature and action. It should accept
                input with spec [feature_spec, encoded_action_spec] and output a scalar tensor. For discrete action,
                encoded_action is an one-hot representation of the action.
                For continuous action, encoded action is the original action.
        """
        super().__init__(name=name)

        flat_action_spec = tf.nest.flatten(action_spec)
        assert len(flat_action_spec) == 1, "doesn't support nested action_spec"

        flat_feature_spec = tf.nest.flatten(feature_spec)
        assert len(
            flat_feature_spec) == 1, "doesn't support nested feature_spec"

        action_spec = flat_action_spec[0]

        if tensor_spec.is_discrete(action_spec):
            self._num_actions = action_spec.maximum - action_spec.minimum + 1
        else:
            self._num_actions = action_spec.shape[-1]

        self._action_spec = action_spec
        self._feature_spec = feature_spec

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


def _reward_function_for_pendulum(obs, action):
    def _observation_cost(obs):
        # gym pendulum
        c_theta, s_theta, d_theta = obs[:, :1], obs[:, 1:2], obs[:, 2:3]
        # theta = tf.sign(tf.math.acos(c_theta)) * tf.math.asin(s_theta)
        theta = tf.math.acos(c_theta)
        cost = tf.reduce_sum(
            tf.square(theta) + 0.1 * tf.square(d_theta), axis=1)
        cost = tf.where(tf.math.is_nan(cost), 1e6 * tf.ones_like(cost), cost)
        return cost

    def _action_cost(action):
        return 0.001 * tf.reduce_sum(tf.square(action), axis=1)

    cost = _observation_cost(obs) + _action_cost(action)
    # negative cost as reward
    reward = -cost
    return reward


@gin.configurable
class FREAlgorithm(REAlgorithm):
    """Fixed Reward Estimation Module with hand-crafted computational rules.
    """

    def __init__(self,
                 env_name,
                 feature_spec,
                 action_spec,
                 hidden_size=256,
                 reward_network: Network = None,
                 name="FixedRewardEstimationAlgorithm"):
        """Create a FixedRewardEstimationAlgorithm.

        Args:
            env_name (str): the name of the environment
            hidden_size (int|tuple): size of hidden layer(s)
            reward_network (Network): network for predicting reward
                based on the current feature and action. It should accept
                input with spec [feature_spec, encoded_action_spec] and output a scalar tensor. For discrete action,
                encoded_action is an one-hot representation of the action.
                For continuous action, encoded action is the original action.
        """
        reward_for_envs = {}
        # implement the respective reward functions for desired environments
        reward_for_envs['pendulum-v0'] = _reward_function_for_pendulum

        env_name_lower = env_name.lower()
        assert env_name_lower in reward_for_envs.keys(
        ), "implement the reward \
            function for %s first" % (env_name)

        super().__init__(
            feature_spec=feature_spec,
            action_spec=action_spec,
            hidden_size=hidden_size,
            reward_network=reward_network,
            name=name)
        self._env_name = env_name_lower
        self._reward_for_envs = reward_for_envs

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
        return self._reward_for_envs[self._env_name](obs, action)
