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

import torch

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import TimeStep, AlgStep, LossInfo
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops
from alf.utils.normalizers import ScalarAdaptiveNormalizer
from alf.utils.normalizers import AdaptiveNormalizer


@alf.configurable
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
                 target_net: EncodingNetwork,
                 predictor_net: EncodingNetwork,
                 encoder_net: EncodingNetwork = None,
                 reward_adapt_speed=None,
                 observation_adapt_speed=None,
                 observation_spec=None,
                 optimizer=None,
                 clip_value=-1.0,
                 keep_stacked_frames=1,
                 name="RNDAlgorithm"):
        """
        Args:
            encoder_net (EncodingNetwork): a shared network that encodes
                observation to embeddings before being input to `target_net` or
                `predictor_net`; its parameters are not trainable.
            target_net (EncodingNetwork): the random fixed network that generates
                target state embeddings to be fitted.
            predictor_net (EncodingNetwork): the trainable network that predicts
                target embeddings. If fully trained given enough data,
                `predictor_net` will become target_net eventually.
            reward_adapt_speed (float): speed for adaptively normalizing intrinsic
                rewards; if None, no normalizer is used.
            observation_adapt_speed (float): speed for adaptively normalizing
                observations. Only useful if `observation_spec` is not None.
            observation_spec (TensorSpec): the observation tensor spec; used
                for creating an adaptive observation normalizer.
            optimizer (torch.optim.Optimizer): The optimizer for training
            clip_value (float): if positive, the rewards will be clipped to
                [-clip_value, clip_value]; only used for reward normalization.
            keep_stacked_frames (int): a non-negative integer indicating how many
                stacked frames we want to keep as the observation. If >0, we only
                keep the last so many frames for RND to make predictions on,
                as suggested by the original paper Burda et al. 2019. For Atari
                games, this argument is usually 1 (with `frame_stacking==4`). If
                it's 0, the observation is unchanged. For other games, the user
                is responsible for setting this value correctly depending on
                how many channels an observation has at each time step.
            name (str):
        """
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

        self._keep_stacked_frames = keep_stacked_frames
        if keep_stacked_frames > 0 and (observation_spec is not None):
            # Assuming stacking in the first dim, we only keep the last frames.
            shape = observation_spec.shape
            assert keep_stacked_frames <= shape[0]
            new_shape = (keep_stacked_frames, ) + tuple(shape[1:])
            observation_spec = TensorSpec(
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

    def _step(self, time_step: TimeStep, state, calc_rewards=True):
        """
        Args:
            time_step (TimeStep): input time_step data
            state (tuple):  empty tuple ()
            calc_rewards (bool): whether calculate rewards

        Returns:
            AlgStep:
                output: empty tuple ()
                state: empty tuple ()
                info: ICMInfo
        """
        observation = time_step.observation

        if self._keep_stacked_frames > 0:
            # Assuming stacking in the first dim, we only keep the last frames.
            observation = observation[:, -self._keep_stacked_frames:, ...]

        if self._observation_normalizer is not None:
            observation = self._observation_normalizer.normalize(observation)

        if self._encoder_net is not None:
            with torch.no_grad():
                observation, _ = self._encoder_net(observation)

        pred_embedding, _ = self._predictor_net(observation)
        with torch.no_grad():
            target_embedding, _ = self._target_net(observation)

        loss = torch.sum(
            math_ops.square(pred_embedding - target_embedding), dim=-1)

        intrinsic_reward = ()
        if calc_rewards:
            intrinsic_reward = loss.detach()
            if self._reward_normalizer:
                intrinsic_reward = self._reward_normalizer.normalize(
                    intrinsic_reward, clip_value=self._reward_clip_value)

        return AlgStep(output=intrinsic_reward, info=loss)

    def predict_step(self, inputs: TimeStep, state):
        return self._step(inputs, state)

    def rollout_step(self, inputs: TimeStep, state):
        return self._step(inputs, state)

    def train_step(self, inputs: TimeStep, state, rollout_info=None):
        return self._step(inputs, state, calc_rewards=False)

    def calc_loss(self, info):
        return LossInfo(scalar_loss=info.mean())
