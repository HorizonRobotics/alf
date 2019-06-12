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

from tf_agents.trajectories.time_step import StepType


def shift_back(a, last):
    """Pop `a[0]` and append `last` to `a`
    """
    return tf.concat([a[1:], tf.expand_dims(last, 0)], axis=0)


def discounted_return(rewards,
                      values,
                      step_types,
                      discounts,
                      final_value,
                      final_time_step,
                      time_major=True):
    """Computes discounted return.

    The difference between this function and the one tf_agents.utils.value_ops
    is that the accumulated_discounted_reward is replaced by value for is_last
    steps in this function.

    ```
    Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'} + gamma^(T-t+1)*final_value.
    ```

    Define abbreviations:
    (B) batch size representing number of trajectories
    (T) number of steps per trajectory

    Args:
        rewards (Tensor): shape is [T, B] (or [T]) representing rewards.
        values (Tensor): shape is [T,B] (or [T]) representing values.
        step_types (Tensor): shape is [T,B] (or [T]) representing step types.
        discounts (Tensor): shape is [T, B] (or [T]) representing discounts.
        final_value (Tensor): shape is [B] (or [1]) representing value estimate
            at t=T.
        final_time_step (TimeStep): time_step at t=T
        time_major (bool): Whether input tensors are time major.
            False means input tensors have shape [B, T].

    Returns:
        A tensor with shape [T, B] (or [T]) representing the discounted returns.
        Shape is [B, T] when time_major is false.
    """

    if not time_major:
        discounts = tf.transpose(a=discounts)
        rewards = tf.transpose(a=rewards)
        values = tf.transpose(a=values)
        step_types = tf.transpose(a=step_types)

    discounts = shift_back(discounts, final_time_step.discount)
    rewards = shift_back(rewards, final_time_step.reward)

    is_lasts = tf.cast(tf.equal(step_types, StepType.LAST), tf.float32)

    def discounted_return_fn(acc_discounted_reward, args):
        (reward, value, is_last, discount) = args
        acc_discounted_value = acc_discounted_reward * discount + reward
        return is_last * value + (1 - is_last) * acc_discounted_value

    returns = tf.scan(
        fn=discounted_return_fn,
        elems=(rewards, values, is_lasts, discounts),
        reverse=True,
        initializer=final_value,
        back_prop=False)

    if not time_major:
        returns = tf.transpose(a=returns)

    return tf.stop_gradient(returns)


def one_step_discounted_return(rewards, values, step_types, discounts,
                               final_value, final_time_step):
    """Calculate the one step discounted return.
    
    return = next_reward + next_discount * next_value
    Note: Input tensors must be time major
    Args:
        rewards (Tensor): shape is [T, B] (or [T]) representing rewards.
        values (Tensor): shape is [T,B] (or [T]) representing values.
        step_types (Tensor): shape is [T,B] (or [T]) representing step types.
        discounts (Tensor): shape is [T, B] (or [T]) representing discounts.
        final_value (Tensor): shape is [B] (or [1]) representing value estimate
            at t=T.
        final_time_step (TimeStep): time_step at t=T

    Returns:
        A tensor with shape [T, B] (or [T]) representing the discounted returns.
    """

    discounts = shift_back(discounts, final_time_step.discount)
    rewards = shift_back(rewards, final_time_step.reward)
    values = shift_back(values, final_value)

    is_lasts = tf.cast(tf.equal(step_types, StepType.LAST), tf.float32)
    returns = rewards + (1 - is_lasts) * discounts * values

    return returns


def generalized_advantage_estimation(rewards,
                                     values,
                                     step_types,
                                     discounts,
                                     final_value,
                                     final_time_step,
                                     td_lambda=1.0,
                                     time_major=True):
    """Computes generalized advantage estimation (GAE).

    For theory, see
    "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    by John Schulman, Philipp Moritz et al.
    See https://arxiv.org/abs/1506.02438 for full paper.

    The difference between this function and the one tf_agents.utils.value_ops
    is that the accumulated_td is reset to 0 for is_last steps in this function.

    Define abbreviations:
        (B) batch size representing number of trajectories
        (T) number of steps per trajectory

    Args:
        rewards (Tensor): shape is [T, B] (or [T]) representing rewards.
        values (Tensor): shape is [T,B] (or [T]) representing values.
        step_types (Tensor): shape is [T,B] (or [T]) representing step types.
        discounts (Tensor): shape is [T, B] (or [T]) representing discounts.
        final_value (Tensor): shape is [B] (or [1]) representing value estimate
            at t=T.
        final_time_step (TimeStep): time_step at t=T
        td_lambda (float): A scalar between [0, 1]. It's used for variance
            reduction in temporal difference.
        time_major (bool): Whether input tensors are time major.
            False means input tensors have shape [B, T].

    Returns:
        A tensor with shape [T, B] representing advantages. Shape is [B, T] when
        time_major is false.
    """

    if not time_major:
        discounts = tf.transpose(a=discounts)
        rewards = tf.transpose(a=rewards)
        values = tf.transpose(a=values)
        step_types = tf.transpose(a=step_types)

    rewards = shift_back(rewards, final_time_step.reward)
    next_values = shift_back(values, final_value)
    discounts = shift_back(discounts, final_time_step.discount)

    is_lasts = tf.cast(tf.equal(step_types, StepType.LAST), tf.float32)

    delta = rewards + discounts * next_values - values
    weighted_discounts = discounts * td_lambda

    def weighted_cumulative_td_fn(accumulated_td, weights_td_is_last):
        weighted_discount, td, is_last = weights_td_is_last
        return (1 - is_last) * (td + weighted_discount * accumulated_td)

    advantages = tf.scan(
        fn=weighted_cumulative_td_fn,
        elems=(weighted_discounts, delta, is_lasts),
        initializer=tf.zeros_like(final_value),
        reverse=True,
        back_prop=False)

    if not time_major:
        advantages = tf.transpose(a=advantages)

    return tf.stop_gradient(advantages)
