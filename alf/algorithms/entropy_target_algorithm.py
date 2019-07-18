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

from collections import namedtuple

import gin
import numpy as np
import tensorflow as tf

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.specs import tensor_spec
from tf_agents.utils import common as tfa_common
from alf.algorithms.algorithm import Algorithm, AlgorithmStep

from alf.utils import common, dist_utils

EntropyTargetLossInfo = namedtuple("EntropyTargetLossInfo",
                                   ["alpha_loss", "entropy_loss"])


@gin.configurable
class EntropyTargetAlgorithm(Algorithm):
    """Algorithm for adjust entropy regularization.

    It tries to adjust the entropy regularization (i.e. alpha) so that the
    the entropy is not smaller than `target_entropy`.

    It's described in section 5 of the following paper:
    Haarnoja et al "Soft Actor-Critic Algorithms and Applications" arXiv:1812.05905v2
    """

    def __init__(self,
                 action_spec,
                 initial_log_alpha=-3.0,
                 target_entropy=None,
                 optimizer=None,
                 debug_summaries=False):
        """Create an EntropyTargetAlgorithm

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            initial_log_alpha (float): initial value for log(alpha). The default
                value corresponds to alpha ~ 0.05.
            target_entropy (float): the lower bound of the entropy. If not
                provided, a default value proportional to the action dimension
                is used.
            optimizer (tf.optimizers.Optimizer): The optimizer for training. If
                not provided, will use the same optimizer of the parent
                algorithm.
            debug_summaries (bool): True if debug summaries should be created.
        """
        log_alpha = tfa_common.create_variable(
            name='log_alpha',
            initial_value=initial_log_alpha,
            dtype=tf.float32,
            trainable=True)
        super().__init__(
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name="EntropyTargetAlgorithm")

        self._log_alpha = log_alpha
        self._action_spec = action_spec

        def _calc_default_target_entropy(spec):
            dims = np.product(spec.shape.as_list())
            if tensor_spec.is_continuous(spec):
                e = -1
            else:
                min_prob = 0.01
                p = min_prob
                q = 1 - p
                e = -p * np.log(p) - q * np.log(q)
            return e * dims

        if target_entropy is None:
            flat_action_spec = tf.nest.flatten(self._action_spec)
            target_entropy = np.sum(
                list(map(_calc_default_target_entropy, flat_action_spec)))
        self._target_entropy = target_entropy

    def train_step(self, distribution):
        """Train step.

        Args:
            distribution (nested Distribution): action distribution from the
                policy.
        Returns:
            AlgorithmStep. `info` field is LossInfo, other fields are empty.
        """
        entropy, entropy_for_gradient = dist_utils.entropy_with_fallback(
            distribution, self._action_spec)
        alpha_loss = self._log_alpha * tf.stop_gradient(entropy -
                                                        self._target_entropy)
        alpha = tf.stop_gradient(tf.exp(self._log_alpha))
        loss = alpha_loss
        entropy_loss = -entropy

        # Joint loss for optimizing alpha and entropy. The effect of alpha_loss
        # is to increase alpha when entropy is lower than target and decrease
        # alpha when entropy is larger than target. alpha * entropy_for_gradient
        # is to encourage higher action entropy.
        loss -= alpha * entropy_for_gradient

        return AlgorithmStep(
            outputs=(),
            state=(),
            info=LossInfo(
                loss,
                extra=EntropyTargetLossInfo(
                    alpha_loss=alpha_loss, entropy_loss=entropy_loss)))

    def calc_loss(self, training_info):
        if self._debug_summaries and common.should_record_summaries():
            tf.summary.scalar("EntropyTargetAlgorithm/alpha",
                              tf.exp(self._log_alpha))
        return training_info
