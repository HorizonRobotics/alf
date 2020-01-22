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
from collections import namedtuple

import gin.tf
import tensorflow as tf

from tf_agents.networks.network import Network
import tf_agents.specs.tensor_spec as tensor_spec

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.utils.encoding_network import EncodingNetwork
from alf.data_structures import StepType

DIAYNInfo = namedtuple("DIAYNInfo", ["reward", "loss"])


@gin.configurable
class DIAYNAlgorithm(Algorithm):
    """Diversity is All You Need Module

    This module learns a set of skill-conditional policies in an unsupervised
    way. See Eysenbach et al "Diversity is All You Need: Learning Diverse Skills
    without a Reward Function" for more details.
    """

    def __init__(self,
                 num_of_skills,
                 feature_spec,
                 hidden_size=256,
                 reward_normalizer=None,
                 encoding_net: Network = None,
                 discriminator_net: Network = None,
                 name="DIAYNAlgorithm"):
        """Create a DIAYNAlgorithm.

        Args:
            num_of_skills (int): number of skills
            hidden_size (int|tuple): size of hidden layer(s)
            reward_normalizer (AdaptiveNormalizer): normalizer for the reward
            encoding_net (Network): network for encoding observation into a
                latent feature specified by feature_spec. Its input is the same
                as the input of this algorithm.
            discriminator_net (Network): network for predicting the skill labels
                based on the observation.
        """
        skill_spec = tf.TensorSpec((num_of_skills, ))
        super().__init__(train_state_spec=skill_spec, name=name)

        flat_feature_spec = tf.nest.flatten(feature_spec)
        assert len(flat_feature_spec
                   ) == 1, "DIAYNAlgorithm doesn't support nested feature_spec"

        self._num_skills = num_of_skills

        self._encoding_net = encoding_net

        if isinstance(hidden_size, int):
            hidden_size = (hidden_size, )

        if discriminator_net is None:
            discriminator_net = EncodingNetwork(
                name="discriminator_net",
                input_tensor_spec=feature_spec,
                fc_layer_params=hidden_size,
                last_layer_size=self._num_skills,
                last_kernel_initializer=tf.initializers.Zeros())

        self._discriminator_net = discriminator_net
        self._reward_normalizer = reward_normalizer

    def train_step(self, inputs, state, calc_intrinsic_reward=True):
        """
        Args:
            inputs (tuple):  skill-augmened observation and step_type
            state (Tensor): state for DIAYN (previous skill)
            calc_intrinsic_reward (bool): if False, only return the losses
        Returns:
            TrainStep:
                outputs: empty tuple ()
                state: skill
                info (DIAYNInfo):
        """
        observations_aug, step_type = inputs
        observation, skill = observations_aug
        prev_skill = state
        batch_size = tf.shape(observation)[0]

        if self._encoding_net is not None:
            feature, _ = self._encoding_net(observation)

        skill_pred, _ = self._discriminator_net(inputs=feature)

        skill_discriminate_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=prev_skill, logits=skill_pred)

        valid_masks = tf.cast(
            tf.not_equal(step_type, StepType.FIRST), tf.float32)
        valid_masks = tf.reshape(valid_masks, [batch_size])

        skill_discriminate_loss = skill_discriminate_loss * valid_masks

        intrinsic_reward = ()

        if calc_intrinsic_reward:
            # use negative cross-entropy as reward
            # neglect neg-prior term as it is constant
            intrinsic_reward = tf.stop_gradient(-skill_discriminate_loss)
            if self._reward_normalizer is not None:
                intrinsic_reward = self._reward_normalizer.normalize(
                    intrinsic_reward)

        return AlgorithmStep(
            outputs=(),
            state=skill,
            info=DIAYNInfo(
                reward=intrinsic_reward,
                loss=LossInfo(
                    loss=skill_discriminate_loss,
                    extra=dict(
                        skill_discriminate_loss=skill_discriminate_loss))))

    def calc_loss(self, info: DIAYNInfo):
        loss = tf.nest.map_structure(tf.reduce_mean, info.loss)
        return LossInfo(scalar_loss=loss.loss, extra=loss.extra)
