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
from alf.algorithms.actor_critic_algorithm import create_ac_algorithm
from alf.algorithms.on_policy_algorithm import OffPolicyAdapter
from alf.algorithms.off_policy_algorithm import Experience
from alf.algorithms.rl_algorithm import ActionTimeStep, TrainingInfo
from alf.utils import common, value_ops

PPOInfo = namedtuple("PPOInfo", ["returns", "advantages"])


@gin.configurable
class PPOAlgorithm(OffPolicyAdapter):
    """PPO Algorithm.
    Implement the simplified surrogate loss in equation (9) of "Proximal
    Policy Optimization Algorithms" https://arxiv.org/abs/1707.06347

    It works with ppo_loss.PPOLoss2. It should have same behavior as
    baselines.ppo2.
    """

    def __init__(self, algorithm: ActorCriticAlgorithm):
        assert isinstance(algorithm, ActorCriticAlgorithm)
        super().__init__(algorithm)

    def preprocess_experience(self, exp: Experience):
        """Compute advantages and put it into exp.info."""
        reward = exp.reward
        if self._algorithm._reward_shaping_fn is not None:
            reward = self._algorithm._reward_shaping_fn(reward)
        reward = self._algorithm.calc_training_reward(exp.reward, exp.info)
        advantages = value_ops.generalized_advantage_estimation(
            rewards=reward,
            values=exp.info.value,
            step_types=exp.step_type,
            discounts=exp.discount * self._algorithm._loss._gamma,
            td_lambda=self._algorithm._loss._lambda,
            time_major=False)
        advantages = tf.concat([
            advantages,
            tf.zeros(
                advantages.shape.as_list()[:-1] + [1], dtype=advantages.dtype)
        ],
                               axis=-1)
        returns = exp.info.value + advantages
        return exp._replace(info=PPOInfo(returns, advantages))

    def predict(self, time_step: ActionTimeStep, state=None):
        return self._algorithm.train_step(time_step, state)

    @property
    def predict_state_spec(self):
        return self._algorithm.train_state_spec


@gin.configurable
def create_ppo_algorithm(env, debug_summaries=False):
    """Create a simple PPOAlgorithm.

    Args:
        env (TFEnvironment): A TFEnvironment
        debug_summaries (bool): True if debug summaries should be created.
    """
    algorithm = create_ac_algorithm(env, debug_summaries=debug_summaries)
    return PPOAlgorithm(algorithm)
