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

from tf_agents.utils import common as tfa_common


def calc_nstep_return(reward: tf.Tensor, value: tf.Tensor, discount: tf.Tensor,
                      step_type: tf.Tensor, nstep: int):
    raise NotImplementedError()


def element_wise_huber_loss(x, y):
    """Elementwise Huber loss
    """
    return tf.compat.v1.losses.huber_loss(
        x, y, reduction=tf.compat.v1.losses.Reduction.NONE)


def calc_q_loss(q_values: tf.Tensor, action: tf.Tensor,
                target_q_value: tf.Tensor):
    """Calculate loss for Q learning
    
    The loss is element_wise_huber_loss(q_values[action], target_q_value)
    Args:
        q_values (tf.Tensor): shape=(..., NA), Q values for all possible actions
        action (tf.Tensor): shape=(...), action from the rollout
        target_q_values (tf.Tensor), shape=(...), the target Q value
    Returns:
        loss (tf.Tensor): shape=(...)
    """
    multi_dim_actions = len(action.shape) > 0
    q_value = tfa_common.index_with_actions(
        q_values,
        tf.cast(action, dtype=tf.int32),
        multi_dim_actions=multi_dim_actions)
    return element_wise_huber_loss(q_value, target_q_value)
