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

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.utils import common as tfa_common
from tf_agents.specs import tensor_spec

from alf.algorithms.rl_algorithm import TrainingInfo
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.actor_critic_loss import _normalize_advantages
from alf.utils.losses import element_wise_squared_loss
from alf.utils import common
from alf.utils import value_ops


@gin.configurable
class PPOLoss(ActorCriticLoss):
    """PPO loss."""

    def __init__(self,
                 action_spec,
                 gamma=0.99,
                 td_error_loss_fn=element_wise_squared_loss,
                 td_lambda=0.95,
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

        This loss works with PPOAlgorithm. The advantages and returns are
        pre-computed by PPOAlgorithm.preprocess(). One known difference with
        baselines.ppo2 is that value estimation is not clipped here, while
        baselines.ppo2 also clipped value if it is deviate from returns too
        much.

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            gamma (float): A discount factor for future rewards.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
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
            use_gae=True,
            td_lambda=td_lambda,
            use_td_lambda_return=True,
            normalize_advantages=normalize_advantages,
            entropy_regularization=entropy_regularization,
            td_loss_weight=td_loss_weight,
            debug_summaries=debug_summaries)

        self._importance_ratio_clipping = importance_ratio_clipping
        self._log_prob_clipping = log_prob_clipping
        self._check_numerics = check_numerics

    def _pg_loss(self, training_info: TrainingInfo, advantages):
        scope = tf.name_scope(self.__class__.__name__)
        importance_ratio, importance_ratio_clipped = value_ops.action_importance_ratio(
            action_distribution=training_info.action_distribution,
            collect_action_distribution=training_info.
            collect_action_distribution,
            action=training_info.action,
            action_spec=self._action_spec,
            clipping_mode='double_sided',
            scope=scope,
            importance_ratio_clipping=self._importance_ratio_clipping,
            log_prob_clipping=self._log_prob_clipping,
            check_numerics=self._check_numerics,
            debug_summaries=self._debug_summaries)
        # Pessimistically choose the maximum objective value for clipped and
        # unclipped importance ratios.
        pg_objective = -importance_ratio * advantages
        pg_objective_clipped = -importance_ratio_clipped * advantages
        policy_gradient_loss = tf.maximum(pg_objective, pg_objective_clipped)

        if self._debug_summaries and common.should_record_summaries():
            with scope:
                tf.summary.histogram('pg_objective', pg_objective)
                tf.summary.histogram('pg_objective_clipped',
                                     pg_objective_clipped)

        if self._check_numerics:
            policy_gradient_loss = tf.debugging.check_numerics(
                policy_gradient_loss, 'policy_gradient_loss')

        return policy_gradient_loss

    def _calc_returns_and_advantages(self, training_info: TrainingInfo, value):
        advantages = training_info.collect_info.advantages
        returns = training_info.collect_info.returns
        return returns, advantages
