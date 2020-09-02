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
import torch.distributions as td

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
        'actions',  # [B, K, ...], () for all possible actions. If None, means all available discrete actions
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

        # action policy from the search policy
        # [B, unroll_steps + 1, num_candidate_actions]
        'action_policy',

        # whether game is over
        # [B, unroll_steps + 1]
        'game_over',

        # value target
        # [B, unroll_steps + 1]
        'value',
    ])


def _entropy(events):
    p = events.to(torch.float32).mean()
    p = torch.tensor([p, 1 - p])
    return -(p * (p + 1e-30).log()).sum(), p[0]


class MCTSModel(nn.Module, metaclass=abc.ABCMeta):
    """The interface for the model used by MCTSAlgorithm."""

    def __init__(self,
                 train_reward_function,
                 train_game_over_function,
                 debug_summaries=False,
                 name="MCTSModel"):
        super().__init__()
        self._debug_summaries = debug_summaries
        self._name = name
        self._train_reward_function = train_reward_function
        self._train_game_over_function = train_game_over_function

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
        num_unroll_steps = target.value.shape[1] - 1
        loss_scale = torch.ones((num_unroll_steps + 1, )) / num_unroll_steps
        loss_scale[0] = 1.0

        value_loss = element_wise_squared_loss(target.value,
                                               model_output.value)
        value_loss = (loss_scale * value_loss).sum(dim=1)
        loss = value_loss

        reward_loss = ()
        if self._train_reward_function:
            reward_loss = element_wise_squared_loss(target.reward,
                                                    model_output.reward)
            reward_loss = (loss_scale * reward_loss).sum(dim=1)
            loss = loss + reward_loss

        # target_action.shape is [B, unroll_steps+1, num_candidate]
        # log_prob needs sample shape in the beginning
        if isinstance(target.action, tuple) and target.action == ():
            # This condition is only possible for Categorical distribution
            assert isinstance(model_output.action_distribution, td.Categorical)
            policy_loss = -(target.action_policy *
                            model_output.action_distribution.logits).sum(dim=2)
        else:
            action = target.action.permute(2, 0, 1,
                                           *list(range(3, target.action.ndim)))
            action_log_probs = model_output.action_distribution.log_prob(
                action)
            action_log_probs = action_log_probs.permute(1, 2, 0)
            policy_loss = -(target.action_policy * action_log_probs).sum(dim=2)

        game_over_loss = ()
        if self._train_game_over_function:
            game_over_loss = F.binary_cross_entropy_with_logits(
                input=model_output.game_over_logit,
                target=target.game_over.to(torch.float),
                reduction='none')
            # no need to train policy after game over.
            policy_loss = policy_loss * (~target.game_over).to(torch.float32)
            unscaled_game_over_loss = game_over_loss
            game_over_loss = (loss_scale * game_over_loss).sum(dim=1)
            loss = loss + game_over_loss

        policy_loss = (loss_scale * policy_loss).sum(dim=1)
        loss = loss + policy_loss

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar(
                    "explained_variance_of_value0",
                    tensor_utils.explained_variance(model_output.value[:, 0],
                                                    target.value[:, 0]))
                alf.summary.scalar(
                    "explained_variance_of_value1",
                    tensor_utils.explained_variance(model_output.value[:, 1:],
                                                    target.value[:, 1:]))
                if self._train_reward_function:
                    alf.summary.scalar(
                        "explained_variance_of_reward0",
                        tensor_utils.explained_variance(
                            model_output.reward[:, 0], target.reward[:, 0]))
                    alf.summary.scalar(
                        "explained_variance_of_reward1",
                        tensor_utils.explained_variance(
                            model_output.reward[:, 1:], target.reward[:, 1:]))

                if self._train_game_over_function:
                    h0, p0 = _entropy(target.game_over[:, 0])
                    alf.summary.scalar("game_over0", p0)
                    h1, p1 = _entropy(target.game_over[:, 1:])
                    alf.summary.scalar("game_over1", p1)

                    alf.summary.scalar(
                        "explained_entropy_of_game_over0",
                        torch.where(
                            h0 == 0, h0,
                            1. - unscaled_game_over_loss[:, 0].mean() /
                            (h0 + 1e-30)))
                    alf.summary.scalar(
                        "explained_entropy_of_game_over1",
                        torch.where(
                            h1 == 0, h1,
                            1. - unscaled_game_over_loss[:, 0].mean() /
                            (h1 + 1e-30)))
                summary_utils.add_mean_hist_summary("target_value",
                                                    target.value)
                summary_utils.add_mean_hist_summary("value",
                                                    model_output.value)
                summary_utils.add_mean_hist_summary(
                    "td_error", target.value - model_output.value)

        return LossInfo(
            loss=loss,
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


@gin.configurable
class SimplePredictionNet(alf.networks.Network):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 trunk_net_ctor,
                 initial_game_over_bias=0.0):
        """
        Args:
            observation_spec (TensorSpec): describing the observation.
            action_spec (BoundedTensorSpec): describing the action.
            trunc_net_ctor (Callable): called as ``trunk_net_ctor(input_tensor_spec=observation_spec)``
                to created a network which taks observation as input and output a
                hidden representation which will be used as input for predicting
                value, reward, action_distribution and game_over_logit
            initial_game_over_bias (float): initial bias for predicting the.
                logit of game_over. Sugguest to use ``log(game_over_prob/(1 - game_over_prob))``
        """
        super().__init__(observation_spec, name="SimplePredictionNet")

        self._trunk_net = trunk_net_ctor(input_tensor_spec=observation_spec)
        dim = self._trunk_net.output_spec.shape[0]
        self._value_layer = alf.layers.FC(
            dim, 1, kernel_initializer=torch.nn.init.zeros_)
        self._reward_layer = alf.layers.FC(
            dim, 1, kernel_initializer=torch.nn.init.zeros_)

        if action_spec.is_continuous:
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

        self._game_over_logit_thresh = 1.0
        self._game_over_layer = alf.layers.FC(
            dim,
            1,
            kernel_initializer=torch.nn.init.zeros_,
            bias_init_value=initial_game_over_bias)

    def forward(self, input, state=()):
        """Predict (value, reward, action_distribution, game_over_logit)

        Args:
            input (Tensor): observation
            state: not used.
        Returns:
            A tuple of: (value, reward, action_distribution, game_over_logit), ()
        """
        # TODO: transform reward/value and use softmax to estimate the value and
        # reward as in appendix F.
        x = self._trunk_net(input)[0]
        value = self._value_layer(x).squeeze(1)
        reward = self._reward_layer(x).squeeze(1)
        action_distribution = self._action_net(x)[0]
        game_over_logit = self._game_over_layer(x).squeeze(1)

        return (value, reward, action_distribution, game_over_logit), ()


def create_simple_prediction_net(observation_spec, action_spec):
    return SimplePredictionNet(
        observation_spec,
        action_spec,
        trunk_net_ctor=partial(EncodingNetwork, fc_layer_params=(256, )))


def create_simple_encoding_net(observation_spec):
    return EncodingNetwork(
        input_tensor_spec=observation_spec, fc_layer_params=(256, 256))


@gin.configurable
class SimpleMCTSModel(MCTSModel):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 num_sampled_actions=None,
                 encoding_net_ctor=create_simple_encoding_net,
                 dynamics_net_ctor=create_simple_dynamics_net,
                 prediction_net_ctor=create_simple_prediction_net,
                 game_over_logit_thresh=1.0,
                 train_reward_function=True,
                 train_game_over_function=True,
                 debug_summaries=False,
                 name="SimpleMCTSModel"):
        """
        Args:
            observation_spec (TensorSpec): representing the observations.
            action_spec (BoundedTensorSpec): representing the actions.
            num_sampled_actions (int): the number of actions sampled from the
                action distribution. For continuous action or multi-dimensional
                discrete action, so many actions will be sampled from the action
                distribution. For 1 dimensional (scalar) discrete action, the
                ``num_sampled_actions`` actions with the largest probability
                will be chosen.
            dynamics_net_ctor (Callable): Called as ``dynamics_net_ctor((observation_spec, action_spec))``
                to create the dynamics net. The created net should take a tuple of
                (observation, action) as input and output the next observation.
            prediction_net_ctor (Callable): Called as ``prediction_net_ctor(observation_spec, action_spec)``
                to create the prediction net. The created net should take the latent_state
                as input and output the prediction for (value, reward, action_distribution, game_over_logit).
            game_over_logit_thresh (float): the threshold of treating the
                state as game over if the logit for game is greater than this.
        """
        super().__init__(
            train_reward_function=train_reward_function,
            train_game_over_function=train_game_over_function,
            debug_summaries=debug_summaries,
            name=name)
        self._num_sampled_actions = num_sampled_actions
        self._encoding_net = encoding_net_ctor(observation_spec)
        repr_spec = self._encoding_net.output_spec
        self._dynamics_net = dynamics_net_ctor(
            input_tensor_spec=(repr_spec, action_spec))
        self._prediction_net = prediction_net_ctor(repr_spec, action_spec)
        self._action_spec = action_spec

        self._sample_actions = False
        if action_spec.is_continuous or action_spec.numel > 1:
            self._sample_actions = True
            assert num_sampled_actions is not None, (
                "num_sampled_actions needs "
                "to be provided for continuous actions or multi-dimensional "
                "discrete actions: action_spec=" % action_spec)

        if not action_spec.is_continuous:
            num_actions = action_spec.maximum - action_spec.minimum + 1
            if num_sampled_actions is None:
                self._actions = torch.arange(
                    num_actions, dtype=torch.int64).unsqueeze(0)
            else:
                assert num_sampled_actions < num_actions, (
                    "For scalar discrete action"
                    "num_sampled_acitons should be smaller than num_actions. Got"
                    "num_sampled_actions=%s, num_actions=%s" %
                    (num_sampled_actions, num_actions))
        self._game_over_logit_thresh = game_over_logit_thresh

    def initial_inference(self, observation):
        state = self._encoding_net(observation)[0]
        state = self._normalize_state(state)
        return self.prediction_model(state)

    def recurrent_inference(self, state, action):
        new_state = self._dynamics_net((state, action))[0]
        new_state = self._normalize_state(new_state)
        return self.prediction_model(new_state)

    def _normalize_state(self, state):
        # normalize state to [0, 1] as suggested in Appendix G
        batch_size = state.shape[0]
        shape = [1] * state.ndim
        shape[0] = batch_size
        min = state.reshape(batch_size, -1).min(dim=1)[0].reshape(shape)
        max = state.reshape(batch_size, -1).max(dim=1)[0].reshape(shape)
        return (state - min) / (max - min + 1e-10)

    def prediction_model(self, state):
        # TODO: transform reward/value and use softmax to estimate the value and
        # reward as in appendix F.
        value, reward, action_distribution, game_over_logit = self._prediction_net(
            state)[0]
        game_over = game_over_logit > self._game_over_logit_thresh
        if self._sample_actions:
            # [num_sampled_actions, B, ...]
            actions = action_distribution.rsample(
                (self._num_sampled_actions, ))
            # [B, num_sampled_actions]
            log_probs = action_distribution.log_prob(actions).transpose(0, 1)
            action_probs = F.softmax(log_probs, dim=1)
            actions = actions.transpose(0, 1)
        else:
            action_probs = action_distribution.probs
            if self._num_sampled_actions is None:
                actions = ()
            else:
                action_probs, actions = action_probs.topk(
                    self._num_sampled_actions, sorted=False)

        if not self._train_reward_function:
            reward = ()
        if not self._train_game_over_function:
            game_over = ()
            game_over_logit = ()

        return ModelOutput(
            value=value,
            reward=reward,
            game_over=game_over,
            actions=actions,
            action_probs=action_probs,
            state=state,
            action_distribution=action_distribution,
            game_over_logit=game_over_logit)
