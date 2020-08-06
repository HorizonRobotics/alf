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

import abc
from functools import partial
import gin
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import alf
from alf.data_structures import LossInfo, namedtuple
from alf.networks import EncodingNetwork, StableNormalProjectionNetwork, CategoricalProjectionNetwork
from alf.utils import dist_utils, tensor_utils, summary_utils
from alf.utils.losses import element_wise_squared_loss

ModelOutput = namedtuple(
    'ModelOutput',
    [
        'value',  # [B], value for the player 0
        'reward',  # [B], reward for the player 0
        'game_over',  # [B], whether the game is over
        'actions',  # [B, K, ...]
        'action_probs',  # [B, K], prob of 0 indicate invalid action
        'state',  # [B, ...]

        # used by calc_loss
        'action_distribution',
        # used by calc_loss
        'game_over_logit'
    ])

ModelTarget = namedtuple(
    'ModelTarget',
    [
        # reward the for taken previous action and the next unoll_steps actions
        # [B, unroll_steps + 1]
        'reward',

        # the candidate actions of the search policy
        # [B, unroll_steps + 1, num_candidate_actions, ...]
        'action',

        # action probability from the search policy
        # [B, unroll_steps + 1, num_candidate_actions]
        'action_probability',

        # whether game is over
        # [B, unroll_steps + 1]
        'game_over',

        # value target
        # [B, unroll_steps + 1]
        'value',
    ])


class MCTSModel(nn.Module, metaclass=abc.ABCMeta):
    """The interface for the model used by MCTSAlgorithm."""

    def __init__(self, debug_summaries=False, name="MCTSModel"):
        super().__init__()
        self._debug_summaries = debug_summaries
        self._name = name

    @abc.abstractmethod
    def initial_inference(self, observation) -> ModelOutput:
        """Generate the initial prediction given observation.

        Returns:
            ModelOutput: the prediction
        """

    @abc.abstractmethod
    def recurrent_inference(self, state, action) -> ModelOutput:
        """Generate prediction given state and action.

        Args:
            state (Tensor): the latent state of the model. The state should be from
                previous call of ``initial_inference`` or ``recurrent_inference``.
            action (Tensor): the imagined action
        Returns:
            ModelOutput: the prediction
        """

    def calc_loss(self, model_output: ModelOutput,
                  target: ModelTarget) -> LossInfo:
        """Calculate the loss.

        The shapes of the tensors in model_output are [B, unroll_steps+1, ...]
        Returns:
            LossInfo
        """
        value_loss = element_wise_squared_loss(target.value,
                                               model_output.value)
        reward_loss = element_wise_squared_loss(target.reward,
                                                model_output.reward)

        # target_action.shape is [B, unroll_steps+1, num_candidate]
        # log_prob needs sample shape in the beginning
        action = target.action.permute(2, 0, 1,
                                       *list(range(3, target.action.ndim)))
        action_log_probs = model_output.action_distribution.log_prob(action)
        action_log_probs = action_log_probs.permute(1, 2, 0)
        policy_loss = -(target.action_probability * action_log_probs).sum(
            dim=2)
        game_over_loss = F.binary_cross_entropy_with_logits(
            input=model_output.game_over_logit,
            target=target.game_over.to(torch.float),
            reduction='none')

        num_unroll_steps = target.value.shape[1] - 1

        loss_scale = torch.ones((num_unroll_steps + 1, )) / num_unroll_steps
        loss_scale[0] = 1.0

        unscaled_game_over_loss = game_over_loss

        value_loss = (loss_scale * value_loss).sum(dim=1)
        reward_loss = (loss_scale * reward_loss).sum(dim=1)
        policy_loss = (loss_scale * policy_loss).sum(dim=1)
        game_over_loss = (loss_scale * game_over_loss).sum(dim=1)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                p = target.game_over.to(torch.float32).mean()
                alf.summary.scalar("game_over", p)
                p = torch.tensor([p, 1 - p]) + 1e-30
                h_game_over = -(p * p.log()).sum()
                alf.summary.scalar(
                    "explained_variance_of_value",
                    tensor_utils.explained_variance(model_output.value,
                                                    target.value))
                alf.summary.scalar(
                    "explained_variance_of_reward",
                    tensor_utils.explained_variance(model_output.reward,
                                                    target.reward))
                alf.summary.scalar(
                    "explained_entropy_of_game_over",
                    1. - unscaled_game_over_loss.mean() / h_game_over)
                summary_utils.add_mean_hist_summary("target_value",
                                                    target.value)
                summary_utils.add_mean_hist_summary("value",
                                                    model_output.value)
                summary_utils.add_mean_hist_summary(
                    "td_error", target.value - model_output.value)

        return LossInfo(
            loss=value_loss + reward_loss + policy_loss + game_over_loss,
            extra=dict(
                value=value_loss,
                reward=reward_loss,
                policy=policy_loss,
                game_over=game_over_loss))


def get_unique_num_actions(action_spec):
    unique_num_actions = np.unique(action_spec.maximum - action_spec.minimum +
                                   1)
    if len(unique_num_actions) > 1 or np.any(unique_num_actions <= 0):
        raise ValueError(
            'Bounds on discrete actions must be the same for all '
            'dimensions and have at least 1 action. Projection '
            'Network requires num_actions to be equal across '
            'action dimensions. Implement a more general '
            'categorical projection if you need more flexibility.')
    return int(unique_num_actions[0])


def create_simple_dynamics_net(input_tensor_spec):
    action_spec = input_tensor_spec[1]
    preproc = None
    if not action_spec.is_continuous:
        preproc = nn.Sequential(
            alf.layers.OneHot(num_classes=get_unique_num_actions(action_spec)),
            alf.layers.Reshape([-1]))
    return EncodingNetwork(
        input_tensor_spec,
        input_preprocessors=(None, preproc),
        preprocessing_combiner=alf.nest.utils.NestConcat(),
        fc_layer_params=(256, 256),
        last_layer_size=input_tensor_spec[0].numel,
        last_activation=alf.math.identity)


def create_simple_prediction_net(input_tensor_spec):
    return EncodingNetwork(input_tensor_spec, fc_layer_params=(256, ))


@gin.configurable
class SimpleMCTSModel(MCTSModel):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 num_sampled_actions=None,
                 dynamics_net_ctor=create_simple_dynamics_net,
                 prediction_net_ctor=create_simple_prediction_net,
                 initial_game_over_bias=0.0,
                 game_over_logit_thresh=1.0,
                 debug_summaries=False,
                 name="SimpleMCTSModel"):
        """
        Args:
            initial_game_over_bias (float): initial bias for predicting the
                logit of game_over. Sugguest to use ``log(game_over_prob/(1 - game_over_prob))``
            game_over_logit_thresh (float): the threshold of treating the
                state as game over if the logit for game is greater than this.
        """
        super().__init__(debug_summaries=debug_summaries, name=name)
        self._num_sampled_actions = num_sampled_actions
        self._dynamics_net = dynamics_net_ctor(
            input_tensor_spec=(observation_spec, action_spec))
        self._prediction_net = prediction_net_ctor(
            input_tensor_spec=self._dynamics_net.output_spec)
        dim = self._prediction_net.output_spec.shape[0]
        self._value_layer = alf.layers.FC(
            dim, 1, kernel_initializer=torch.nn.init.zeros_)
        self._reward_layer = alf.layers.FC(
            dim, 1, kernel_initializer=torch.nn.init.zeros_)
        if action_spec.is_continuous or action_spec.numel > 1:
            assert num_sampled_actions is not None, (
                "num_sampled_actions needs "
                "to be provided for continuous actions or multi-dimensional "
                "discrete actions: action_spec=" % action_spec)
            self._action_net = StableNormalProjectionNetwork(
                input_size=dim,
                action_spec=action_spec,
                state_dependent_std=True,
                scale_distribution=True,
                dist_squashing_transform=dist_utils.Softsign())
        else:
            self._action_net = CategoricalProjectionNetwork(
                input_size=dim,
                action_spec=action_spec,
                logits_init_output_factor=1e-10)
            if num_sampled_actions is None:
                num_actions = action_spec.maximum - action_spec.minimum + 1
                self._actions = torch.arange(
                    num_actions, dtype=torch.int64).unsqueeze(0)

        self._game_over_logit_thresh = 1.0
        self._game_over_layer = alf.layers.FC(
            dim,
            1,
            kernel_initializer=torch.nn.init.zeros_,
            bias_init_value=initial_game_over_bias)

    def initial_inference(self, observation):
        state = self._normalize_state(observation)
        return self.prediction_model(state)

    def recurrent_inference(self, state, action):
        new_state = self._dynamics_net((state, action))[0]
        new_state = self._normalize_state(new_state)
        return self.prediction_model(new_state)

    def _normalize_state(self, state):
        # normalize state to [0, 1] as suggested in Appendix G
        min = state.min(dim=1, keepdims=True)[0]
        max = state.max(dim=1, keepdims=True)[0]
        return (state - min) / (max - min + 1e-10)

    def prediction_model(self, state):
        # TODO: transform reward/value and use softmax to estimate the value and
        # reward as in appendix F.
        x = self._prediction_net(state)[0]
        value = self._value_layer(x).squeeze(1)
        reward = self._reward_layer(x).squeeze(1)
        action_distribution = self._action_net(x)[0]
        game_over_logit = self._game_over_layer(x).squeeze(1)
        game_over = game_over_logit > self._game_over_logit_thresh
        if self._num_sampled_actions is not None:
            # [num_sampled_actions, B, ...]
            actions = action_distribution.rsample(
                (self._num_sampled_actions, ))
            # [B, num_sampled_actions]
            log_probs = action_distribution.log_prob(actions).transpose(0, 1)
            action_probs = F.softmax(log_probs, dim=1)
            actions = actions.transpose(0, 1)
        else:
            actions = self._actions.expand(state.shape[0], -1)
            action_probs = action_distribution.probs

        return ModelOutput(
            value=value,
            reward=reward,
            game_over=game_over,
            actions=actions,
            action_probs=action_probs,
            state=state,
            action_distribution=action_distribution,
            game_over_logit=game_over_logit)
