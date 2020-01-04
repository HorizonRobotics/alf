# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""PPO algorithm."""

from collections import namedtuple
import gin
import tensorflow as tf

from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.data_structures import Experience, ActionTimeStep, TrainingInfo
from alf.utils import common, value_ops

PPOInfo = namedtuple("PPOInfo",
                     ["action_distribution", "returns", "advantages"])


@gin.configurable
class PPOAlgorithm(ActorCriticAlgorithm):
    """PPO Algorithm.
    Implement the simplified surrogate loss in equation (9) of "Proximal
    Policy Optimization Algorithms" https://arxiv.org/abs/1707.06347

    It works with ppo_loss.PPOLoss. It should have same behavior as
    baselines.ppo2.
    """

    def preprocess_experience(self, exp: Experience):
        """Compute advantages and put it into exp.rollout_info."""
        advantages = value_ops.generalized_advantage_estimation(
            rewards=exp.reward,
            values=exp.rollout_info.value,
            step_types=exp.step_type,
            discounts=exp.discount * self._loss._gamma,
            td_lambda=self._loss._lambda,
            time_major=False)
        advantages = tf.concat([
            advantages,
            tf.zeros(
                shape=common.concat_shape(tf.shape(advantages)[:-1], [1]),
                dtype=advantages.dtype)
        ],
                               axis=-1)
        returns = exp.rollout_info.value + advantages
        return exp._replace(
            rollout_info=PPOInfo(exp.rollout_info.action_distribution, returns,
                                 advantages))
