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

import gin.tf
import tensorflow as tf

# implement the respective reward functions for desired environments here


@gin.configurable
def reward_function_for_pendulum(obs, action):
    """Function for computing reward for gym Pendulum environment. It takes
        as input:
        (1) observation (tf.Tensor of shape [batch_size, observation_dim])
        (2) action (tf.Tensor of shape [batch_size, num_actions])
        and returns a reward Tensor of shape [batch_size].
    """

    def _observation_cost(obs):
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
