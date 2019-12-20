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
"""An algorithm for adjusting entropy regularization strength."""
from absl import logging
import gin
import numpy as np
import tensorflow as tf

from tf_agents.trajectories.time_step import StepType

from alf.algorithms.algorithm import Algorithm, AlgorithmStep
from alf.data_structures import namedtuple, LossInfo
from alf.utils import dist_utils
from alf.utils.averager import ScalarWindowAverager
from alf.utils.common import run_if, should_record_summaries
from alf.utils.dist_utils import calc_default_target_entropy
from alf.utils.dist_utils import calc_default_max_entropy

EntropyTargetLossInfo = namedtuple("EntropyTargetLossInfo", ["neg_entropy"])
EntropyTargetInfo = namedtuple("EntropyTargetInfo", ["step_type", "loss"])


@gin.configurable
class EntropyTargetAlgorithm(Algorithm):
    """Algorithm for adjust entropy regularization.

    It tries to adjust the entropy regularization (i.e. alpha) so that the
    the entropy is not smaller than `target_entropy`.

    The algorithm has three stages:
    0. init stage. This is an optional stage. If the initial entropy is already
       below `max_entropy`, then this stage is skipped. Otherwise, the alpha will
       be slowly decreased so that the entropy will land at `max_entropy` to
       trigger the next `free stage`. Basically, this stage let the user to choose
       an arbitrary large init alpha without considering every specific case.
    1. free stage. During this stage, the alpha is not changed. It transitions
       to adjust_stage once entropy drops below `target_entropy`.
    2. adjust stage. During this stage, log_alpha is adjusted using this formula:
       ((below + 0.5 * above) * decreasing - (above + 0.5 * below) * increasing) * update_rate
       Note that log_alpha will always be decreased if entropy is increasing
       even when the entropy is below the target entropy. This is to prevent
       overshooting log_alpha to a too big value. Same reason for always
       increasing log_alpha even when the entropy is above the target entropy.
       `update_rate` is initialized to `fast_update_rate` and is reduced by a
       factor of 0.9 whenever the entropy crosses `target_entropy`. `udpate_rate`
       is reset to `fast_update_rate` if entropy drops too much below
       `target_entropy` (i.e., fast_stage_thresh in the code, which is the half
       of `target_entropy` if it is positive, and twice of `target_entropy` if
       it is negative.
    """

    def __init__(self,
                 action_spec,
                 initial_alpha=0.1,
                 max_entropy=None,
                 target_entropy=None,
                 very_slow_update_rate=0.001,
                 slow_update_rate=0.01,
                 fast_update_rate=np.log(2),
                 min_alpha=1e-4,
                 average_window=2,
                 debug_summaries=False):
        """Create an EntropyTargetAlgorithm

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            initial_alpha (float): initial value for alpha; make sure that it's
                large enough for initial meaningful exploration
            max_entropy (float): the upper bound of the entropy. If not provided,
                a default value proportional to the action dimension is used.
            target_entropy (float): the lower bound of the entropy. If not
                provided, a default value proportional to the action dimension
                is used. This value should be less or equal than `max_entropy`.
            very_slow_update_rate (float): a tiny update rate for log_alpha; used
                in stage 0
            slow_update_rate (float): minimal update rate for log_alpha; used in
                stage 2
            fast_update_rate (float): maximum update rate for log_alpha; used in
                state 2
            min_alpha (float): the minimal value of alpha. If <=0, exp(-100) is
                used.
            average_window (int): window size for averaging past entropies.
            optimizer (tf.optimizers.Optimizer): The optimizer for training. If
                not provided, will use the same optimizer of the parent
                algorithm.
            debug_summaries (bool): True if debug summaries should be created.
        """
        super().__init__(
            debug_summaries=debug_summaries, name="EntropyTargetAlgorithm")

        self._log_alpha = tf.Variable(
            name='log_alpha',
            initial_value=np.log(initial_alpha),
            dtype=tf.float32,
            trainable=False)
        self._stage = tf.Variable(
            name='stage', initial_value=-2, dtype=tf.int32, trainable=False)
        self._avg_entropy = ScalarWindowAverager(average_window)
        self._update_rate = tf.Variable(
            name='update_rate',
            initial_value=fast_update_rate,
            dtype=tf.float32,
            trainable=False)
        self._action_spec = action_spec
        self._min_log_alpha = -100.
        if min_alpha >= 0.:
            self._min_log_alpha = np.log(min_alpha)

        flat_action_spec = tf.nest.flatten(self._action_spec)
        if target_entropy is None:
            target_entropy = np.sum(
                list(map(calc_default_target_entropy, flat_action_spec)))
        if max_entropy is None:
            max_entropy = np.sum(
                list(map(calc_default_max_entropy, flat_action_spec)))
        assert target_entropy <= max_entropy, \
            ("Target entropy %s should be less or equal than max entropy %s!"
             % (target_entropy, max_entropy))
        if target_entropy > 0:
            self._fast_stage_thresh = 0.5 * target_entropy
        else:
            self._fast_stage_thresh = 2.0 * target_entropy
        self._target_entropy = target_entropy
        self._max_entropy = max_entropy
        self._very_slow_update_rate = very_slow_update_rate
        self._slow_update_rate = slow_update_rate
        self._fast_update_rate = fast_update_rate
        logging.info("target_entropy=%s" % target_entropy)
        logging.info("max_entropy=%s" % max_entropy)

    def train_step(self, distribution, step_type):
        """Train step.

        Args:
            distribution (nested Distribution): action distribution from the
                policy.
            step_type (StepType): the step type for the distributions.
        Returns:
            AlgorithmStep. `info` field is LossInfo, other fields are empty.
        """
        entropy, entropy_for_gradient = dist_utils.entropy_with_fallback(
            distribution, self._action_spec)
        return AlgorithmStep(
            outputs=(),
            state=(),
            info=EntropyTargetInfo(
                step_type=step_type,
                loss=LossInfo(
                    loss=-entropy_for_gradient,
                    extra=EntropyTargetLossInfo(neg_entropy=-entropy))))

    def calc_loss(self, training_info: EntropyTargetInfo, valid_mask=None):
        loss_info = training_info.loss
        mask = tf.cast(training_info.step_type != StepType.LAST, tf.float32)
        if valid_mask:
            mask = mask * tf.cast(valid_mask, tf.float32)
        entropy = -loss_info.extra.neg_entropy * mask
        num = tf.reduce_sum(mask)
        not_empty = num > 0
        num = tf.maximum(num, 1)
        entropy2 = tf.reduce_sum(tf.square(entropy)) / num
        entropy = tf.reduce_sum(entropy) / num
        entropy_std = tf.sqrt(tf.maximum(0.0, entropy2 - entropy * entropy))

        run_if(not_empty, lambda: self.adjust_alpha(entropy))

        def _summarize():
            with self.name_scope:
                tf.summary.scalar("entropy_std", entropy_std)

        if self._debug_summaries:
            run_if(
                tf.logical_and(not_empty, should_record_summaries()),
                _summarize)

        alpha = tf.exp(self._log_alpha)
        return loss_info._replace(loss=loss_info.loss * alpha)

    def adjust_alpha(self, entropy):
        """Adjust alpha according to the current entropy.

        Args:
            entropy (scalar Tensor). the current entropy.
        Returns:
            adjusted entropy regularization
        """
        prev_avg_entropy = self._avg_entropy.get()
        avg_entropy = self._avg_entropy.average(entropy)

        def _init():
            below = avg_entropy < self._max_entropy
            increasing = tf.cast(avg_entropy > prev_avg_entropy, tf.float32)
            # -1 * increasing + 0.5 * (1 - increasing)
            update_rate = (
                0.5 - 1.5 * increasing) * self._very_slow_update_rate
            self._stage.assign_add(tf.cast(below, tf.int32))
            self._log_alpha.assign(
                tf.maximum(self._log_alpha + update_rate,
                           np.float32(self._min_log_alpha)))

        def _free():
            crossing = avg_entropy < self._target_entropy
            self._stage.assign_add(tf.cast(crossing, tf.int32))

        def _adjust():
            previous_above = tf.cast(self._stage, tf.bool)
            above = avg_entropy > self._target_entropy
            self._stage.assign(tf.cast(above, tf.int32))
            crossing = above != previous_above
            update_rate = self._update_rate
            update_rate = tf.where(crossing, 0.9 * update_rate, update_rate)
            update_rate = tf.maximum(update_rate, self._slow_update_rate)
            update_rate = tf.where(entropy < self._fast_stage_thresh,
                                   np.float32(self._fast_update_rate),
                                   update_rate)
            self._update_rate.assign(update_rate)
            above = tf.cast(above, tf.float32)
            below = 1 - above
            increasing = tf.cast(avg_entropy > prev_avg_entropy, tf.float32)
            decreasing = 1 - increasing
            log_alpha = self._log_alpha + (
                (below + 0.5 * above) * decreasing -
                (above + 0.5 * below) * increasing) * update_rate
            log_alpha = tf.maximum(log_alpha, np.float32(self._min_log_alpha))
            self._log_alpha.assign(log_alpha)

        run_if(self._stage == -2, _init)
        run_if(self._stage == -1, _free)
        run_if(self._stage >= 0, _adjust)
        alpha = tf.exp(self._log_alpha)

        def _summarize():
            with self.name_scope:
                tf.summary.scalar("alpha", alpha)
                tf.summary.scalar("avg_entropy", avg_entropy)
                tf.summary.scalar("stage", self._stage)
                tf.summary.scalar("update_rate", self._update_rate)

        if self._debug_summaries:
            run_if(should_record_summaries(), _summarize)

        return alpha
