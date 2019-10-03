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

import gin.tf
import tensorflow as tf

from tf_agents.networks.network import Network

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.utils.adaptive_normalizer import ScalarAdaptiveNormalizer
from alf.utils.adaptive_normalizer import AdaptiveNormalizer

RNDInfo = namedtuple("RNDInfo", ["reward", "loss"])


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
                 observation_adapt_speed=None,
                 observation_spec=None,
                 name="RNDAlgorithm"):
        """
        Args:
            target_net (Network): the random fixed network that generates target
                state embeddings to be fitted
            predictor_net (Network): the trainable network that predicts target
                embeddings. If fully trained given enough data, predictor_net
                will become target_net eventually.
            reward_adapt_speed (float): speed for adaptively normalizing intrinsic
                rewards
            observation_adapt_speed (float): speed for adaptively normalizing
                observations. Only useful if `observation_spec` is not None.
            observation_spec (TensorSpec): the observation tensor spec; used
                for creating an adaptive observation normalizer
            name (str):
        """
        super(RNDAlgorithm, self).__init__(train_state_spec=(), name=name)
        self._target_net = target_net  # fixed
        self._predictor_net = predictor_net  # trainable
        self._reward_normalizer = ScalarAdaptiveNormalizer(
            speed=reward_adapt_speed)
        # The paper suggests to also normalize observations, because the
        # original observation subspace might be small and the target network will
        # yield random embeddings that are indistinguishable
        self._observation_normalizer = None
        if observation_adapt_speed is not None:
            assert observation_spec is not None, \
                "Observation normalizer requires its input tensor spec!"
            self._observation_normalizer = AdaptiveNormalizer(
                tensor_spec=observation_spec, speed=observation_adapt_speed)

    def train_step(self, inputs, state):
        """
        Args:
            inputs (tuple): observation and previous action
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: emplty tuple ()
                info: RNDInfo
        """
        observation, _ = inputs
        if self._observation_normalizer is not None:
            observation = self._observation_normalizer.normalize(observation)

        pred_embedding, _ = self._predictor_net(observation)
        target_embedding, _ = self._target_net(observation)

        loss = 0.5 * tf.reduce_mean(
            tf.square(pred_embedding - tf.stop_gradient(target_embedding)),
            axis=-1)

        intrinsic_reward = tf.stop_gradient(loss)
        intrinsic_reward = self._reward_normalizer.normalize(intrinsic_reward)

        return AlgorithmStep(
            outputs=(),
            state=(),
            info=RNDInfo(reward=intrinsic_reward, loss=LossInfo(loss=loss)))

    def calc_loss(self, info: RNDInfo):
        return LossInfo(scalar_loss=tf.reduce_mean(info.loss.loss))
