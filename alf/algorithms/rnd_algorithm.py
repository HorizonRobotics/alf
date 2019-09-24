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

import gin.tf
import tensorflow as tf

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks.network import Network

from alf.algorithms.algorithm import Algorithm, AlgorithmStep
from alf.utils.adaptive_normalizer import ScalarAdaptiveNormalizer


@gin.configurable
class RNDAlgorithm(Algorithm):
    """Exploration by Random Network Distillation, Burda et al. 2019.

    This module generates the intrinsic reward based on the prediction errors of
    randomly generated state embeddings.

    Suppose we have a fixed randomly initialized target network g: s -> e_t and
    a trainable predictor network h: s -> e_p, then the intrinsic reward is

    r = |e_t - e_p|^2

    The reward is expected to be higher for novel states.
    """

    def __init__(self,
                 target_net: Network,
                 predictor_net: Network,
                 reward_adapt_speed=8.0,
                 name="RNDAlgorithm"):

        super(RNDAlgorithm, self).__init__(train_state_spec=(), name=name)
        self._target_net = target_net  # fixed
        self._predictor_net = predictor_net  # trainable
        self._reward_normalizer = ScalarAdaptiveNormalizer(
            speed=reward_adapt_speed)

    def train_step(self, inputs, state):
        """
        Args:
            inputs (tuple): observation and previous action
        Returns:
            TrainStep:
                outputs: intrinsic reward
                state:
                info:
        """
        observation, _ = inputs

        pred_embedding, _ = self._predictor_net(observation)
        target_embedding, _ = self._target_net(observation)

        loss = 0.5 * tf.reduce_mean(
            tf.square(pred_embedding - tf.stop_gradient(target_embedding)),
            axis=-1)

        intrinsic_reward = tf.stop_gradient(loss)
        intrinsic_reward = self._reward_normalizer.normalize(intrinsic_reward)

        return AlgorithmStep(
            outputs=intrinsic_reward,
            state=(),
            info=LossInfo(loss=loss, extra=dict(pred_loss=loss)))
