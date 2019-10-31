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

from absl import logging
import gin
import numpy as np
import tensorflow as tf

from tf_agents.trajectories.time_step import StepType

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.utils import dist_utils
from alf.utils.averager import ScalarWindowAverager
from alf.utils.common import namedtuple, run_if, should_record_summaries
from alf.utils.dist_utils import calc_default_target_entropy

EntropyTargetLossInfo = namedtuple("EntropyTargetLossInfo", ["entropy_loss"])
EntropyTargetInfo = namedtuple("EntropyTargetInfo", ["step_type", "loss"])


@gin.configurable
class EntropyTargetAlgorithm(Algorithm):
    """Algorithm for adjust entropy regularization.

    It tries to adjust the entropy regularization (i.e. alpha) so that the
    the entropy is not smaller than `target_entropy`.

    The algorithm has two stages:
    1. init stage. During this stage, the alpha is not changed. It transitions
       to fast_stage once entropy drops below `target_entropy`.
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
                 initial_alpha=0.01,
                 target_entropy=None,
                 slow_update_rate=0.01,
                 fast_update_rate=np.log(2),
                 min_alpha=1e-4,
                 debug_summaries=False):
        """Create an EntropyTargetAlgorithm

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            initial_log_alpha (float): initial value for alpha.
            target_entropy (float): the lower bound of the entropy. If not
                provided, a default value proportional to the action dimension
                is used.
            slow_update_rate (float):
            fast_update_rate (float):
            min_alpha (float): the minimal value of alpha. If <=0, exp(-100) is
                used.
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
            name='stage', initial_value=-1, dtype=tf.int32, trainable=False)
        self._avg_entropy = ScalarWindowAverager(2)
        self._update_rate = tf.Variable(
            name='update_rate',
            initial_value=fast_update_rate,
            dtype=tf.float32,
            trainable=False)
        self._action_spec = action_spec
        self._min_log_alpha = -100.
        if min_alpha >= 0.:
            self._min_log_alpha = np.log(min_alpha)

        if target_entropy is None:
            flat_action_spec = tf.nest.flatten(self._action_spec)
            target_entropy = np.sum(
                list(map(calc_default_target_entropy, flat_action_spec)))
        if target_entropy > 0:
            self._fast_stage_thresh = 0.5 * target_entropy
        else:
            self._fast_stage_thresh = 2.0 * target_entropy
        self._target_entropy = target_entropy
        self._slow_update_rate = slow_update_rate
        self._fast_update_rate = fast_update_rate
        logging.info("target_entropy=%s" % target_entropy)

    def train_step(self, distribution, step_type):
        """Train step.

        Args:
            distribution (nested Distribution): action distribution from the
                policy.
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
                    extra=EntropyTargetLossInfo(entropy_loss=-entropy))))

    def calc_loss(self, training_info: EntropyTargetInfo):
        loss_info = training_info.loss
        mask = tf.cast(training_info.step_type != StepType.LAST, tf.float32)
        entropy = -loss_info.extra.entropy_loss * mask
        num = tf.reduce_sum(mask)
        entropy2 = tf.reduce_sum(tf.square(entropy)) / num
        entropy = tf.reduce_sum(entropy) / num
        entropy_std = tf.sqrt(tf.maximum(0.0, entropy2 - entropy * entropy))
        prev_avg_entropy = self._avg_entropy.get()
        avg_entropy = self._avg_entropy.average(entropy)

        def _init():
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

        run_if(self._stage == -1, _init)
        run_if(self._stage >= 0, _adjust)
        alpha = tf.exp(self._log_alpha)

        def _summarize():
            with self.name_scope:
                tf.summary.scalar("alpha", alpha)
                tf.summary.scalar("entropy_std", entropy_std)
                tf.summary.scalar("avg_entropy", avg_entropy)
                tf.summary.scalar("stage", self._stage)
                tf.summary.scalar("update_rate", self._update_rate)

        if self._debug_summaries:
            run_if(should_record_summaries(), _summarize)

        return loss_info._replace(loss=loss_info.loss * alpha)
