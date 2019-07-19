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
"""Various functions related to calculating values."""

import tensorflow as tf

from tf_agents.trajectories.time_step import StepType
from tf_agents.utils import common as tfa_common

from alf.utils import common


def action_importance_ratio(action_distribution, collect_action_distribution,
                            action, action_spec, clipping_mode, scope,
                            importance_ratio_clipping, log_prob_clipping,
                            check_numerics, debug_summaries):
    """ ratio for importance sampling, used in PPO loss and vtrace loss.

        Caller has to save tf.name_scope() and pass scope to this function.

        Args:
            action_distribution (nested tf.distribution): Distribution over
                actions under target policy.
            collect_action_distribution (nested tf.distribution): distribution
                over actions from behavior policy, used to sample actions for
                the rollout.
            action (nested tf.distribution): possibly batched action tuple
                taken during rollout.
            action_spec (nested BoundedTensorSpec): representing the actions.
            clipping_mode (str): mode for clipping the importance ratio.
                'double_sided': clips the range of importance ratio into
                    [1-importance_ratio_clipping, 1+importance_ratio_clipping],
                    which is used by PPOLoss.
                'capping': clips the range of importance ratio into
                    min(1+importance_ratio_clipping, importance_ratio),
                    which is used by VTraceLoss, where c_bar or rho_bar =
                    1+importance_ratio_clipping.
            scope (name scope manager): returned by tf.name_scope(), set
                outside.
            importance_ratio_clipping (float):  Epsilon in clipped, surrogate
                PPO objective. See the cited paper for more detail.
            log_prob_clipping (float): If >0, clipping log probs to the range
                (-log_prob_clipping, log_prob_clipping) to prevent inf / NaN
                values.
            check_numerics (bool):  If true, adds tf.debugging.check_numerics to
                help find NaN / Inf values. For debugging only.
            debug_summaries (bool): If true, output summary metrics to tf.

        Returns:
            importance_ratio (Tensor), importance_ratio_clipped (Tensor).
    """
    current_policy_distribution = action_distribution

    sample_action_log_probs = tfa_common.log_probability(
        collect_action_distribution, action, action_spec)
    sample_action_log_probs = tf.stop_gradient(sample_action_log_probs)

    action_log_prob = tfa_common.log_probability(current_policy_distribution,
                                                 action, action_spec)
    if log_prob_clipping > 0.0:
        action_log_prob = tf.clip_by_value(action_log_prob, -log_prob_clipping,
                                           log_prob_clipping)
    if check_numerics:
        action_log_prob = tf.debugging.check_numerics(action_log_prob,
                                                      'action_log_prob')

    # Prepare both clipped and unclipped importance ratios.
    importance_ratio = tf.exp(action_log_prob - sample_action_log_probs)
    if check_numerics:
        importance_ratio = tf.debugging.check_numerics(importance_ratio,
                                                       'importance_ratio')

    if clipping_mode == 'double_sided':
        importance_ratio_clipped = tf.clip_by_value(
            importance_ratio, 1 - importance_ratio_clipping,
            1 + importance_ratio_clipping)
    elif clipping_mode == 'capping':
        importance_ratio_clipped = tf.minimum(importance_ratio,
                                              1 + importance_ratio_clipping)
    else:
        raise Exception('Unsupported clipping mode: ' + clipping_mode)

    if debug_summaries and common.should_record_summaries():
        with scope:
            if importance_ratio_clipping > 0.0:
                clip_fraction = tf.reduce_mean(
                    input_tensor=tf.cast(
                        tf.greater(
                            tf.abs(importance_ratio - 1.0),
                            importance_ratio_clipping), tf.float32))
                tf.summary.scalar('clip_fraction', clip_fraction)

            tf.summary.histogram('action_log_prob', action_log_prob)
            tf.summary.histogram('action_log_prob_sample',
                                 sample_action_log_probs)
            tf.summary.histogram('importance_ratio', importance_ratio)
            tf.summary.scalar('importance_ratio_mean',
                              tf.reduce_mean(input_tensor=importance_ratio))
            tf.summary.histogram('importance_ratio_clipped',
                                 importance_ratio_clipped)

    return importance_ratio, importance_ratio_clipped


def discounted_return(rewards, values, step_types, discounts, time_major=True):
    """Computes discounted return for the first T-1 steps.

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
        time_major (bool): Whether input tensors are time major.
            False means input tensors have shape [B, T].

    Returns:
        A tensor with shape [T-1, B] (or [T-1]) representing the discounted
        returns. Shape is [B, T-1] when time_major is false.
    """

    if not time_major:
        discounts = tf.transpose(a=discounts)
        rewards = tf.transpose(a=rewards)
        values = tf.transpose(a=values)
        step_types = tf.transpose(a=step_types)

    discounts = discounts[1:]
    rewards = rewards[1:]
    final_value = values[-1]
    values = values[:-1]

    step_types = step_types[:-1]
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


def one_step_discounted_return(rewards, values, step_types, discounts):
    """Calculate the one step discounted return  for the first T-1 steps.

    return = next_reward + next_discount * next_value
    Note: Input tensors must be time major
    Args:
        rewards (Tensor): shape is [T, B] (or [T]) representing rewards.
        values (Tensor): shape is [T,B] (or [T]) representing values.
        step_types (Tensor): shape is [T,B] (or [T]) representing step types.
        discounts (Tensor): shape is [T, B] (or [T]) representing discounts.
    Returns:
        A tensor with shape [T-1, B] (or [T-1]) representing the discounted
        returns.
    """

    discounts = discounts[1:]
    rewards = rewards[1:]
    values = values[1:]
    step_types = step_types[:-1]

    is_lasts = tf.cast(tf.equal(step_types, StepType.LAST), tf.float32)
    returns = rewards + (1 - is_lasts) * discounts * values

    return returns


def generalized_advantage_estimation(rewards,
                                     values,
                                     step_types,
                                     discounts,
                                     td_lambda=1.0,
                                     time_major=True):
    """Computes generalized advantage estimation (GAE) for the first T-1 steps.

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
        td_lambda (float): A scalar between [0, 1]. It's used for variance
            reduction in temporal difference.
        time_major (bool): Whether input tensors are time major.
            False means input tensors have shape [B, T].

    Returns:
        A tensor with shape [T-1, B] representing advantages. Shape is [B, T-1]
        when time_major is false.
    """

    if not time_major:
        discounts = tf.transpose(a=discounts)
        rewards = tf.transpose(a=rewards)
        values = tf.transpose(a=values)
        step_types = tf.transpose(a=step_types)

    rewards = rewards[1:]
    next_values = values[1:]
    final_value = values[-1]
    values = values[:-1]
    discounts = discounts[1:]
    step_types = step_types[:-1]
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
