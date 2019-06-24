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
"""Loss for PPO algorithm."""

import gin
import tensorflow as tf

from tf_agents.utils import common as tfa_common
from tf_agents.specs import tensor_spec

from alf.algorithms.rl_algorithm import TrainingInfo
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.utils.losses import element_wise_squared_loss
from alf.utils import common


@gin.configurable
class PPOLoss(ActorCriticLoss):
    def __init__(self,
                 action_spec,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 use_gae=True,
                 td_lambda=0.95,
                 use_td_lambda_return=True,
                 normalize_advantages=True,
                 advantage_clip=None,
                 entropy_regularization=None,
                 td_loss_weight=1.0,
                 importance_ratio_clipping=0.2,
                 log_prob_clipping=0.0,
                 check_numerics=False,
                 debug_summaries=False):
        """Create a PPOLoss object

        Implement the simplified surrogate loss in equation (9) of "Proximal
        Policy Optimization Algorithms" https://arxiv.org/abs/1707.06347

        The total loss equals to 
        (policy_gradient_loss (L^{CLIP} in equation (9))
         + td_loss_weight * td_loss (L^{VF} in equation (9))
         - entropy_regularization * entropy)

        Note: There is a difference with baseline.ppo2 implementation. Here the
            advantage is recomputed after every update performed in
            OffPolicyDriver._update(), where in baseline.ppo2, the advantage is
            fixed within one epoch.
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
            normalize_advantages (bool): If True, normalize advantage to zero
                mean and unit variance within batch for caculating policy
                gradient.
            advantage_clip (float): If set, clip advantages to [-x, x]
            entropy_regularization (float): Coefficient for entropy
                regularization loss term.
            td_loss_weight (float): the weigt for the loss of td error.
            importance_ratio_clipping (float):  Epsilon in clipped, surrogate
                PPO objective. See the cited paper for more detail.
            log_prob_clipping (float): If >0, clipping log probs to the range
                (-log_prob_clipping, log_prob_clipping) to prevent inf / NaN
                values.
            check_numerics (bool):  If true, adds tf.debugging.check_numerics to
                help find NaN / Inf values. For debugging only.
        """

        super(PPOLoss, self).__init__(
            action_spec=action_spec,
            gamma=gamma,
            td_error_loss_fn=td_error_loss_fn,
            use_gae=use_gae,
            td_lambda=td_lambda,
            use_td_lambda_return=use_td_lambda_return,
            normalize_advantages=normalize_advantages,
            entropy_regularization=entropy_regularization,
            td_loss_weight=td_loss_weight,
            debug_summaries=debug_summaries)

        self._importance_ratio_clipping = importance_ratio_clipping
        self._log_prob_clipping = log_prob_clipping
        self._check_numerics = check_numerics

    def _pg_loss(self, training_info: TrainingInfo, advantages):
        current_policy_distribution = training_info.action_distribution

        sample_action_log_probs = tfa_common.log_probability(
            training_info.collect_action_distribution, training_info.action,
            self._action_spec)
        sample_action_log_probs = tf.stop_gradient(sample_action_log_probs)

        action_log_prob = tfa_common.log_probability(
            current_policy_distribution, training_info.action,
            self._action_spec)
        if self._log_prob_clipping > 0.0:
            action_log_prob = tf.clip_by_value(action_log_prob,
                                               -self._log_prob_clipping,
                                               self._log_prob_clipping)
        if self._check_numerics:
            action_log_prob = tf.debugging.check_numerics(
                action_log_prob, 'action_log_prob')

        # Prepare both clipped and unclipped importance ratios.
        importance_ratio = tf.exp(action_log_prob - sample_action_log_probs)
        if self._check_numerics:
            importance_ratio = tf.debugging.check_numerics(
                importance_ratio, 'importance_ratio')
        importance_ratio_clipped = tf.clip_by_value(
            importance_ratio, 1 - self._importance_ratio_clipping,
            1 + self._importance_ratio_clipping)

        # Pessimistically choose the maximum objective value for clipped and
        # unclipped importance ratios.
        pg_objective = -importance_ratio * advantages
        pg_objective_clipped = -importance_ratio_clipped * advantages
        policy_gradient_loss = tf.maximum(pg_objective, pg_objective_clipped)

        if self._debug_summaries and common.should_record_summaries():
            with tf.name_scope('PPOLoss'):
                if self._importance_ratio_clipping > 0.0:
                    clip_fraction = tf.reduce_mean(
                        input_tensor=tf.cast(
                            tf.greater(
                                tf.abs(importance_ratio - 1.0), self.
                                _importance_ratio_clipping), tf.float32))
                    tf.summary.scalar('clip_fraction', clip_fraction)

                tf.summary.histogram('action_log_prob', action_log_prob)
                tf.summary.histogram('action_log_prob_sample',
                                     sample_action_log_probs)
                tf.summary.histogram('importance_ratio', importance_ratio)
                tf.summary.scalar(
                    'importance_ratio_mean',
                    tf.reduce_mean(input_tensor=importance_ratio))
                tf.summary.histogram('importance_ratio_clipped',
                                     importance_ratio_clipped)
                tf.summary.histogram('pg_objective', pg_objective)
                tf.summary.histogram('pg_objective_clipped',
                                     pg_objective_clipped)

        if self._check_numerics:
            policy_gradient_loss = tf.debugging.check_numerics(
                policy_gradient_loss, 'policy_gradient_loss')

        return policy_gradient_loss
