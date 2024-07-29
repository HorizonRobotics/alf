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
from alf.data_structures import TimeStep, namedtuple, AlgStep, LossInfo, StepType
from alf.networks import EncodingNetwork
from alf.nest.utils import NestConcat
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops
from alf.utils.normalizers import ScalarAdaptiveNormalizer, AdaptiveNormalizer

ICMInfo = namedtuple("ICMInfo", ["step_type", "forward_loss", "inverse_loss"])


@alf.configurable
class ICMAlgorithm(Algorithm):
    """Intrinsic Curiosity Module

    This module generate the intrinsic reward based on predition error of
    observation.

    See Pathak et al "Curiosity-driven Exploration by Self-supervised Prediction"
    """

    def __init__(self,
                 action_spec,
                 observation_spec=None,
                 hidden_size=256,
                 reward_adapt_speed=8.0,
                 encoding_net: EncodingNetwork = None,
                 forward_net: EncodingNetwork = None,
                 inverse_net: EncodingNetwork = None,
                 activation=torch.relu_,
                 optimizer=None,
                 name="ICMAlgorithm"):
        """Create an ICMAlgorithm.

        Args
            action_spec (nested TensorSpec): agent's action spec
            observation_spec (nested TensorSpec): agent's observation spec. If
                not None, then a normalizer will be used to normalize the
                observation.
            hidden_size (int or tuple[int]): size of hidden layer(s)
            reward_adapt_speed (float): how fast to adapt the reward normalizer.
                rouphly speaking, the statistics for the normalization is
                calculated mostly based on the most recent T/speed samples,
                where T is the total number of samples.
            encoding_net (Network): network for encoding observation into a
                latent feature. Its input is same as the input of this algorithm.
            forward_net (Network): network for predicting next feature based on
                previous feature and action. It should accept input with spec
                [feature_spec, encoded_action_spec] and output a tensor of shape
                feature_spec. For discrete action, encoded_action is an one-hot
                representation of the action. For continuous action, encoded
                action is same as the original action.
            inverse_net (Network): network for predicting previous action given
                the previous feature and current feature. It should accept input
                with spec [feature_spec, feature_spec] and output tensor of
                shape (num_actions,).
            activation (torch.nn.functional): activation used for constructing
                any of the forward net and inverse net, if not provided.
            optimizer (torch.optim.Optimizer): The optimizer for training
            name (str):
        """
        if encoding_net is not None:
            feature_spec = encoding_net.output_spec
        else:
            feature_spec = observation_spec

        super(ICMAlgorithm, self).__init__(
            train_state_spec=feature_spec,
            predict_state_spec=(),
            optimizer=optimizer,
            name=name)

        flat_action_spec = alf.nest.flatten(action_spec)
        assert len(
            flat_action_spec) == 1, "ICM doesn't support nested action_spec"

        flat_feature_spec = alf.nest.flatten(feature_spec)
        assert len(
            flat_feature_spec) == 1, "ICM doesn't support nested feature_spec"

        action_spec = flat_action_spec[0]

        if action_spec.is_discrete:
            self._num_actions = int(action_spec.maximum - action_spec.minimum +
                                    1)
        else:
            self._num_actions = action_spec.shape[-1]

        self._action_spec = action_spec
        self._observation_normalizer = None
        if observation_spec is not None:
            self._observation_normalizer = AdaptiveNormalizer(
                tensor_spec=observation_spec)

        feature_dim = flat_feature_spec[0].shape[-1]

        self._encoding_net = encoding_net

        if isinstance(hidden_size, int):
            hidden_size = (hidden_size, )

        if forward_net is None:
            encoded_action_spec = TensorSpec((self._num_actions, ),
                                             dtype=torch.float32)
            forward_net = EncodingNetwork(
                name="forward_net",
                input_tensor_spec=[feature_spec, encoded_action_spec],
                preprocessing_combiner=NestConcat(),
                fc_layer_params=hidden_size,
                activation=activation,
                last_layer_size=feature_dim,
                last_activation=math_ops.identity)

        self._forward_net = forward_net

        if inverse_net is None:
            inverse_net = EncodingNetwork(
                name="inverse_net",
                input_tensor_spec=[feature_spec, feature_spec],
                preprocessing_combiner=NestConcat(),
                fc_layer_params=hidden_size,
                activation=activation,
                last_layer_size=self._num_actions,
                last_activation=math_ops.identity,
                last_kernel_initializer=torch.nn.init.zeros_)

        self._inverse_net = inverse_net

        self._reward_normalizer = ScalarAdaptiveNormalizer(
            speed=reward_adapt_speed)

    def _encode_action(self, action):
        if self._action_spec.is_discrete:
            return torch.nn.functional.one_hot(action, self._num_actions).to(
                torch.float32)
        else:
            return action

    def _step(self, time_step: TimeStep, state, calc_rewards=True):
        """This step is for both `rollout_step` and `train_step`.

        Args:
            time_step (TimeStep): input time_step data for ICM
            state (Tensor): state for ICM (previous observation)
            calc_rewards (bool): whether calculate rewards

        Returns:
            AlgStep:
                output: empty tuple ()
                state: observation
                info (ICMInfo):
        """
        feature = time_step.observation
        prev_action = time_step.prev_action.detach()

        # normalize observation for easier prediction
        if self._observation_normalizer is not None:
            feature = self._observation_normalizer.normalize(feature)

        if self._encoding_net is not None:
            feature, _ = self._encoding_net(feature)
        prev_feature = state

        forward_pred, _ = self._forward_net(
            [prev_feature.detach(),
             self._encode_action(prev_action)])
        # nn.MSELoss doesn't support reducing along a dim
        forward_loss = 0.5 * torch.mean(
            math_ops.square(forward_pred - feature.detach()), dim=-1)

        action_pred, _ = self._inverse_net([prev_feature, feature])

        if self._action_spec.is_discrete:
            inverse_loss = torch.nn.CrossEntropyLoss(reduction='none')(
                input=action_pred, target=prev_action.to(torch.int64))
        else:
            # nn.MSELoss doesn't support reducing along a dim
            inverse_loss = 0.5 * torch.mean(
                math_ops.square(action_pred - prev_action), dim=-1)

        intrinsic_reward = ()
        if calc_rewards:
            intrinsic_reward = forward_loss.detach()
            intrinsic_reward = self._reward_normalizer.normalize(
                intrinsic_reward)

        return AlgStep(
            output=intrinsic_reward,
            state=feature,
            info=ICMInfo(
                step_type=time_step.step_type,
                forward_loss=forward_loss,
                inverse_loss=inverse_loss))

    def predict_step(self, inputs: TimeStep, state):
        return self._step(inputs, state)

    def rollout_step(self, inputs: TimeStep, state):
        return self._step(inputs, state)

    def train_step(self, inputs: TimeStep, state, rollout_info=None):
        return self._step(inputs, state, calc_rewards=False)

    def calc_loss(self, info: ICMInfo):
        mask = (info.step_type != StepType.FIRST).to(torch.float32)
        forward_loss = (info.forward_loss * mask).mean()
        inverse_loss = (info.inverse_loss * mask).mean()
        return LossInfo(
            scalar_loss=forward_loss + inverse_loss,
            extra=dict(forward_loss=forward_loss, inverse_loss=inverse_loss))
