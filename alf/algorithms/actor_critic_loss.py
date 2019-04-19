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
from tf_agents.utils import value_ops

from alf.utils.losses import element_wise_squared_loss

ActorCriticLossInfo = namedtuple("ActorCriticLossInfo",
                                 ["pg_loss", "td_loss", "entropy_loss"])


@gin.configurable
class ActorCriticLoss(object):
    def __init__(self,
                 action_spec,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 use_gae=False,
                 td_lambda=0.95,
                 entropy_regularization=None):

        self._action_spec = action_spec
        self._td_loss_weight = 1.0
        self._gamma = gamma
        self._td_error_loss_fn = td_error_loss_fn
        self._use_gae = use_gae
        self._lambda = td_lambda
        self._entropy_regularization = entropy_regularization

    def __call__(self, training_info, value, final_value):
        returns = value_ops.discounted_return(
            training_info.reward, training_info.discount, final_value)

        valid_masks = 1 - training_info.is_last

        action_log_prob = tfa_common.log_probability(
            training_info.action_distribution, training_info.action,
            self._action_spec)

        if not self._use_gae:
            advantages = returns - value
        else:
            advantages = value_ops.generalized_advantage_estimation(
                values=value,
                final_value=final_value,
                rewards=training_info.reward,
                discounts=training_info.discount,
                td_lambda=self._lambda)

        pg_loss = -tf.stop_gradient(advantages) * action_log_prob
        pg_loss = tf.reduce_mean(pg_loss * valid_masks)

        td_loss = self._td_error_loss_fn(tf.stop_gradient(returns), value)
        td_loss = tf.reduce_mean(td_loss * valid_masks)

        loss = pg_loss + self._td_loss_weight * td_loss

        entropy_loss = 0
        if self._entropy_regularization is not None:
            entropies = tfa_common.entropy(training_info.action_distribution,
                                           self._action_spec)
            entropy_loss = -tf.reduce_mean(input_tensor=entropies)
            loss += self._entropy_regularization * entropy_loss

        return LossInfo(
            loss,
            ActorCriticLossInfo(
                td_loss=td_loss, pg_loss=pg_loss, entropy_loss=entropy_loss))
