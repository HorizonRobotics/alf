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


def discounted_return(
        rewards,
        values,
        is_lasts,
        discounts,
        time_major=True):
    if not time_major:
        with tf.name_scope("to_time_major_tensors"):
            discounts = tf.transpose(a=discounts)
            rewards = tf.transpose(a=rewards)
            values = tf.transpose(a=values)
            is_lasts = tf.transpose(a=is_lasts)
    """Computes discounted return.

    ```
    final_value = 0 for terminate state else v
    Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'} + gamma^(T-t+1)*final_value.
    ```

    Define abbreviations:
    (B) batch size representing number of trajectories
    (T) number of steps per trajectory

    Args:
      rewards: Tensor with shape [T, B] (or [T]) representing rewards.
      values: Tensor with shape [T,B] (or [T]) representing values.
      is_lasts: Tensor with shape [T,B] (or [T]) indicating whether is last value 
      discounts: Tensor with shape [T, B] (or [T]) representing discounts.
      time_major: A boolean indicating whether input tensors are time major. False
        means input tensors have shape [B, T].

    Returns:
        A tensor with shape [T, B] (or [T]) representing the discounted returns.
        Shape is [B, T] when time_major is false.
    """

    is_terminated = tf.cast(tf.math.equal(discounts, 0.0), tf.float32)
    # For Not TimeLimit environments such as CartPole-v0, TimeStep.discount = 0.0
    # when TimeStep.step_type== StepType.LAST, its value should be zero.
    # And for TimeLimit environments such as Pendulum, TimeStep.discount !=0
    # when TimeStep.step_type == StepType.LAST, its estimated value should stay
    values = values * (1. - is_terminated)

    final_value = values[-1]

    def discounted_return_fn(acc_discounted_reward, args):
        (reward, value, is_last, discount) = args
        acc_discounted_value = acc_discounted_reward * discount + reward
        return is_last * value + (1 - is_last) * acc_discounted_value

    returns = tf.scan(
        fn=discounted_return_fn,
        elems=(rewards[:-1],
               values[:-1],
               tf.cast(is_lasts, tf.float32)[:-1],
               discounts[:-1]),
        reverse=True,
        initializer=final_value,
        back_prop=False)
    if len(returns.shape) != len(final_value.shape):
        final_value = tf.reshape(final_value, (1, -1))

    returns = tf.concat((returns, final_value), axis=0)

    if not time_major:
        with tf.name_scope("to_batch_major_tensors"):
            returns = tf.transpose(a=returns)

    return tf.stop_gradient(returns)
