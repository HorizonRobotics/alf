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
from alf.data_structures import ActionTimeStep
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.utils.normalizers import AdaptiveNormalizer
from alf.algorithms.icm_algorithm import ICMInfo


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
                 encoder_net: Network = None,
                 reward_adapt_speed=None,
                 observation_adapt_speed=None,
                 observation_spec=None,
                 learning_rate=None,
                 clip_value=-1.0,
                 stacked_frames=True,
                 name="RNDAlgorithm"):
        """
        Args:
            encoder_net (Network): a shared network that encodes imgs to
                embeddings before being input to `target_net` or `predictor_net`;
                its parameters are not trainable
            target_net (Network): the random fixed network that generates target
                state embeddings to be fitted
            predictor_net (Network): the trainable network that predicts target
                embeddings. If fully trained given enough data, predictor_net
                will become target_net eventually.
            reward_adapt_speed (float): speed for adaptively normalizing intrinsic
                rewards; if None, no normalizer is used
            observation_adapt_speed (float): speed for adaptively normalizing
                observations. Only useful if `observation_spec` is not None.
            observation_spec (TensorSpec): the observation tensor spec; used
                for creating an adaptive observation normalizer
            learning_rate (float): the learning rate for prediction cost; if None,
                a global learning rate will be used
            clip_value (float): if positive, the rewards will be clipped to
                [-clip_value, clip_value]; only used for reward normalization
            stacked_frames (bool): a boolean flag indicating whether the input
                observation has stacked frames. If True, then we only keep the
                last frame for RND to make predictions on, as suggested by the
                original paper Burda et al. 2019. For Atari games, this flag is
                usually True (`frame_stacking==4`).
            name (str):
        """
        optimizer = None
        if learning_rate is not None:
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        super(RNDAlgorithm, self).__init__(
            train_state_spec=(), optimizer=optimizer, name=name)
        self._encoder_net = encoder_net
        self._target_net = target_net  # fixed
        self._predictor_net = predictor_net  # trainable
        if reward_adapt_speed is not None:
            self._reward_normalizer = ScalarAdaptiveNormalizer(
                speed=reward_adapt_speed)
            self._reward_clip_value = clip_value
        else:
            self._reward_normalizer = None

        self._stacked_frames = stacked_frames
        if stacked_frames and (observation_spec is not None):
            # Assuming stacking in the last dim, we only keep the last frame.
            shape = observation_spec.shape
            new_shape = shape[:-1] + (1, )
            observation_spec = tf.TensorSpec(
                shape=new_shape, dtype=observation_spec.dtype)

        # The paper suggests to also normalize observations, because the
        # original observation subspace might be small and the target network will
        # yield random embeddings that are indistinguishable
        self._observation_normalizer = None
        if observation_adapt_speed is not None:
            assert observation_spec is not None, \
                "Observation normalizer requires its input tensor spec!"
            self._observation_normalizer = AdaptiveNormalizer(
                tensor_spec=observation_spec, speed=observation_adapt_speed)

    def train_step(self,
                   time_step: ActionTimeStep,
                   state,
                   calc_intrinsic_reward=True):
        """
        Args:
            time_step (ActionTimeStep): input time_step data
            state (tuple):  empty tuple ()
            calc_intrinsic_reward (bool): if False, only return the losses
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: empty tuple ()
                info: ICMInfo
        """
        observation = time_step.observation

        if self._stacked_frames:
            # Assuming stacking in the last dim, we only keep the last frame.
            observation = observation[..., -1:]

        if self._observation_normalizer is not None:
            observation = self._observation_normalizer.normalize(observation)

        if self._encoder_net is not None:
            observation = tf.stop_gradient(self._encoder_net(observation)[0])

        pred_embedding, _ = self._predictor_net(observation)
        target_embedding, _ = self._target_net(observation)

        loss = tf.reduce_sum(
            tf.square(pred_embedding - tf.stop_gradient(target_embedding)),
            axis=-1)

        intrinsic_reward = ()
        if calc_intrinsic_reward:
            intrinsic_reward = tf.stop_gradient(loss)
            if self._reward_normalizer:
                intrinsic_reward = self._reward_normalizer.normalize(
                    intrinsic_reward, clip_value=self._reward_clip_value)

        return AlgorithmStep(
            outputs=(),
            state=(),
            info=ICMInfo(reward=intrinsic_reward, loss=LossInfo(loss=loss)))

    def calc_loss(self, info: ICMInfo):
        return LossInfo(scalar_loss=tf.reduce_mean(info.loss.loss))
