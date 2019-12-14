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

import tensorflow as tf
import gin.tf

from alf.algorithms.rl_algorithm import TrainingInfo, LossInfo
from alf.utils import common, losses, value_ops


@gin.configurable
class OneStepTDLoss(object):
    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=losses.element_wise_squared_loss,
                 debug_summaries=False):
        """
        Args:
            gamma (float): A discount factor for future rewards.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            debug_summaries (bool): True if debug summaries should be created
        """
        self._gamma = gamma
        self._td_error_loss_fn = td_error_loss_fn
        self._debug_summaries = debug_summaries

    def __call__(self, training_info: TrainingInfo, value, target_value):
        returns = value_ops.one_step_discounted_return(
            rewards=training_info.reward,
            values=target_value,
            discounts=training_info.discount * self._gamma)
        returns = common.tensor_extend(returns, value[-1])
        if self._debug_summaries:
            with tf.name_scope('OneStepTDLoss'):
                tf.summary.scalar("values", tf.reduce_mean(value))
                tf.summary.scalar("returns", tf.reduce_mean(returns))
        loss = self._td_error_loss_fn(tf.stop_gradient(returns), value)
        return LossInfo(loss=loss, extra=loss)
