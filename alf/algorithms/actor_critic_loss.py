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

from collections import namedtuple

import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.utils import common as tfa_common

from alf.algorithms.on_policy_algorithm import TrainingInfo
from alf.utils.losses import element_wise_squared_loss
from alf.utils import value_ops

ActorCriticLossInfo = namedtuple("ActorCriticLossInfo",
                                 ["pg_loss", "td_loss", "entropy_loss"])


def _normalize_advantages(advantages, axes=(0, ), variance_epsilon=1e-8):
    adv_mean, adv_var = tf.nn.moments(x=advantages, axes=axes, keepdims=True)
    normalized_advantages = (
        (advantages - adv_mean) / (tf.sqrt(adv_var) + variance_epsilon))
    return normalized_advantages


@gin.configurable
class ActorCriticLoss(object):
    def __init__(self,
                 action_spec,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 use_gae=False,
                 td_lambda=0.95,
                 use_td_lambda_return=True,
                 normalize_advantages=False,
                 entropy_regularization=None,
                 td_loss_weight=1.0,
                 debug_summaries=False):
        """Create a ActorCriticLoss object

        The total loss equals to 
        (policy_gradient_loss
         + td_loss_weight * td_loss
         - entropy_regularization * entropy)

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            gamma (float): A discount factor for future rewards.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            use_gae (bool): If True, uses generalized advantage estimation for
                computing per-timestep advantage. Else, just subtracts value
                predictions from empirical return.
            use_td_lambda_return (bool): Only effective if use_gae is True.
                If True, uses td_lambda_return for training value function.
                (td_lambda_return = gae_advantage + value_predictions)
            td_lambda (float): Lambda parameter for TD-lambda computation.
            entropy_regularization (float): Coefficient for entropy
                regularization loss term.
            td_loss_weight (float): the weigt for the loss of td error.
        """

        self._action_spec = action_spec
        self._td_loss_weight = td_loss_weight
        self._gamma = gamma
        self._td_error_loss_fn = td_error_loss_fn
        self._use_gae = use_gae
        self._lambda = td_lambda
        self._use_td_lambda_return = use_td_lambda_return
        self._normalize_advantages = normalize_advantages
        self._entropy_regularization = entropy_regularization
        self._debug_summaries = debug_summaries

    def __call__(self, training_info: TrainingInfo, value, final_time_step,
                 final_value):
        """Cacluate actor critic loss

        Except final_value, the first dimension of all the tensors is time
        dimension and the second dimesion is the batch dimension.

        Args:
            training_info (TrainingInfo): training_info collected by
                TrainingPolicy. All tensors in training_info are time-major
            value (tf.Tensor): the time-major tensor for the value at each time
                step
            final_value (tf.Tensor): the value at one step ahead.
        Returns:
            loss_info (LossInfo): with loss_info.extra being ActorCriticLossInfo
        """
        returns = value_ops.discounted_return(
            rewards=training_info.reward,
            values=value,
            step_types=training_info.step_type,
            discounts=training_info.discount * self._gamma,
            final_value=final_value,
            final_time_step=final_time_step)

        if not self._use_gae:
            advantages = returns - value
        else:
            advantages = value_ops.generalized_advantage_estimation(
                rewards=training_info.reward,
                values=value,
                step_types=training_info.step_type,
                discounts=training_info.discount * self._gamma,
                final_value=final_value,
                final_time_step=final_time_step,
                td_lambda=self._lambda)
            if self._use_td_lambda_return:
                returns = advantages + value

        if self._normalize_advantages:
            advantages = _normalize_advantages(advantages, axes=(0, 1))
        pg_loss = self._pg_loss(training_info, advantages)

        td_loss = self._td_error_loss_fn(tf.stop_gradient(returns), value)

        loss = pg_loss + self._td_loss_weight * td_loss

        entropy_loss = ()
        if self._entropy_regularization is not None:
            entropies = tfa_common.entropy(training_info.action_distribution,
                                           self._action_spec)
            entropy_loss = -entropies
            loss += self._entropy_regularization * entropy_loss

        if self._debug_summaries:
            with tf.name_scope('ActorCriticLoss'):
                tf.summary.scalar("values", tf.reduce_mean(value))
                tf.summary.scalar("returns", tf.reduce_mean(returns))
                tf.summary.scalar("advantages", tf.reduce_mean(advantages))

        return LossInfo(
            loss,
            ActorCriticLossInfo(
                td_loss=td_loss, pg_loss=pg_loss, entropy_loss=entropy_loss))

    def _pg_loss(self, training_info, advantages):
        action_log_prob = tfa_common.log_probability(
            training_info.action_distribution, training_info.action,
            self._action_spec)
        return -tf.stop_gradient(advantages) * action_log_prob
