# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as td
from typing import Callable

import alf
from alf.data_structures import LossInfo, namedtuple
from alf.networks import EncodingNetwork, StableNormalProjectionNetwork, CategoricalProjectionNetwork
from alf.utils import dist_utils, tensor_utils, summary_utils
from alf.utils.losses import element_wise_squared_loss, multi_quantile_huber_loss
from alf.utils.schedulers import ConstantScheduler

ModelState = namedtuple(
    'ModelState',
    [
        'state',  # the actual latent state of the model
        'step',  # the current unroll step of the model
    ],
    default_value=())

ModelOutput = namedtuple(
    'ModelOutput',
    [
        'value',  # [B], value for the player 0
        'reward',  # [B], reward for the player 0
        'game_over',  # [B], whether the game is over

        # [B, K, ...], candidate actions, () all available discrete actions
        'actions',

        # [B, K], probabilities of the candidate actions. prob of 0 indicates invalid action.
        # In the case when the candidate actions are sampled from the original action space,
        # action_probs should bee normalized over the sampled candidate action set.
        # i.e. action_probs[i, :] should sum to 1
        'action_probs',

        # [B, ...], latent state
        'state',

        # The following fields are used by calc_loss
        'action_distribution',
        'game_over_logit',
        'value_quantiles',
        'reward_quantiles',
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

        # [B, unroll_steps + 1]
        'observation',
    ],
    default_value=())


@alf.configurable
class MCTSModel(nn.Module, metaclass=abc.ABCMeta):
    """The interface for the model used by MCTSAlgorithm."""

    def __init__(self,
                 num_unroll_steps,
                 representation_net,
                 dynamics_net,
                 prediction_net,
                 train_reward_function,
                 train_game_over_function,
                 train_repr_prediction=False,
                 predict_reward_sum=False,
                 value_loss_weight=1.0,
                 reward_loss_weight=1.0,
                 policy_loss_weight=1.0,
                 game_over_loss_weight=1.0,
                 repr_prediction_loss_weight=1.0,
                 initial_alpha=0.0,
                 target_entropy=None,
                 alpha_adjust_rate=0.001,
                 debug_summaries=False,
                 name="MCTSModel"):
        """
        Args:
            representation_net (Network): the network for generating initial
                latent representation from observation. It is called as
                ``representation_net(observation)``.
            dynamics_net (Network): the network for generating the next latent
                representation given the current latent representation and action.
                It is called as ``dynamics_net((current_latent_representation, action))``
            train_reward_function (bool): whether to predict reward
            train_game_over_function (bool): whether to predict game over
            train_repr_prediction (bool): whether to train to predict future
                latent representation.
            predict_reward_sum (bool): If True, the loss for reward is the mean
                square error between the sum of predicted reward over unroll
                steps and the sum of actual reward over unroll steps. If False,
                the loss for reward is the mean square error between the predicted
                reward and the actual reward.
            value_loss_weight (float): the weight for value prediction loss.
            reward_loss_weight (float): the weight for reward prediction loss
            policy_loss_weight (float): the weight for policy prediction loss
            repr_prediction_loss_weight (float): the weight for the loss of
                predicting latent representation.
            initial_alpha (float): initial value for the weight of entropy regulariation
            target_entropy (float): if provided, will adjust alpha automatically
                so that the entropy is not smaller than this.
            alpha_adjust_rate (float): the speed to adjust alpha
        """
        super().__init__()
        self._representation_net = representation_net
        self._dynamics_net = dynamics_net
        self._prediction_net = prediction_net
        self._debug_summaries = debug_summaries
        self._name = name
        self._train_reward_function = train_reward_function
        self._train_game_over_function = train_game_over_function
        self._train_repr_prediction = train_repr_prediction
        self._predict_reward_sum = predict_reward_sum
        if initial_alpha > 0:
            self.register_buffer("_log_alpha",
                                 torch.tensor(np.log(initial_alpha)))
        else:
            self._log_alpha = None
        if target_entropy is not None:
            if not isinstance(target_entropy, Callable):
                target_entropy = ConstantScheduler(target_entropy)
        self._target_entropy = target_entropy
        self._alpha_adjust_rate = alpha_adjust_rate
        self._value_loss_weight = value_loss_weight
        self._reward_loss_weight = reward_loss_weight
        self._policy_loss_weight = policy_loss_weight
        self._game_over_loss_weight = game_over_loss_weight
        self._repr_prediction_loss_weight = repr_prediction_loss_weight

        found1 = alf.layers.prepare_rnn_batch_norm(self._dynamics_net)
        found2 = alf.layers.prepare_rnn_batch_norm(self._prediction_net)
        self._handle_bn = found1 or found2
        if self._handle_bn:
            self._dynamics_net.set_batch_norm_max_steps(num_unroll_steps)
            self._prediction_net.set_batch_norm_max_steps(num_unroll_steps + 1)

    def initial_inference(self, observation):
        """Generate the initial prediction given observation.

        Returns:
            ModelOutput: the prediction
        """
        state = self._representation_net(observation)[0]
        if not self._handle_bn:
            model_output = self.prediction_model(state)
            return model_output._replace(
                state=ModelState(state=model_output.state))
        else:
            self._prediction_net.set_batch_norm_current_step(0)
            model_output = self.prediction_model(state)
            batch_size = model_output.value.shape[0]
            current_steps = torch.zeros(batch_size, dtype=torch.long)
            return model_output._replace(
                state=ModelState(model_output.state, current_steps))

    def recurrent_inference(self, state, action):
        """Generate prediction given state and action.

        Args:
            state (Tensor): the latent state of the model. The state should be from
                previous call of ``initial_inference`` or ``recurrent_inference``.
            action (Tensor): the imagined action
        Returns:
            ModelOutput: the prediction
        """
        if not self._handle_bn:
            state = self._dynamics_net((state.state, action))[0]
            model_output = self.prediction_model(state)
            return model_output._replace(
                state=ModelState(state=model_output.state))
        else:
            state, current_steps = state
            self._dynamics_net.set_batch_norm_current_step(current_steps)
            new_state = self._dynamics_net((state, action))[0]
            current_steps = current_steps + 1
            self._prediction_net.set_batch_norm_current_step(current_steps)
            model_output = self.prediction_model(new_state)
            return model_output._replace(
                state=ModelState(model_output.state, current_steps))

    def calc_loss(self, model_output: ModelOutput,
                  target: ModelTarget) -> LossInfo:
        """Calculate the loss.

        The shapes of the tensors in model_output are [B, unroll_steps+1, ...]
        Returns:
            LossInfo: the shapes of the tensors are [B]
        """
        batch_size = target.value.shape[0]
        num_unroll_steps = target.value.shape[1] - 1
        loss_scale = torch.ones((num_unroll_steps + 1, )) / num_unroll_steps
        loss_scale[0] = 1.0

        if isinstance(model_output.value_quantiles, torch.Tensor):
            value_loss = multi_quantile_huber_loss(
                model_output.value_quantiles, target.value)
        else:
            value_loss = element_wise_squared_loss(target.value,
                                                   model_output.value)
        value_loss = (loss_scale * value_loss).sum(dim=1)
        loss = self._value_loss_weight * value_loss

        reward_loss = ()
        if self._train_reward_function:
            if self._predict_reward_sum:
                reward = model_output.reward.cumsum(dim=1)
                target_reward = target.reward.cumsum(dim=1)
            else:
                reward = model_output.reward
                target_reward = target.reward
            if isinstance(model_output.reward_quantiles, torch.Tensor):
                if self._predict_reward_sum:
                    reward_quantiles = model_output.reward_quantiles.cumsum(
                        dim=1)
                else:
                    reward_quantiles = model_output.reward_quantiles
                reward_loss = multi_quantile_huber_loss(
                    reward_quantiles, target_reward)
            else:
                reward_loss = element_wise_squared_loss(reward, target_reward)
            reward_loss = (loss_scale * reward_loss).sum(dim=1)
            loss = loss + self._reward_loss_weight * reward_loss

        if target.action is ():
            # This condition is only possible for Categorical distribution
            assert isinstance(model_output.action_distribution, td.Categorical)
            policy_loss = -(target.action_policy *
                            model_output.action_distribution.logits).sum(dim=2)
        else:
            # target_action.shape is [B, unroll_steps+1, num_candidate]
            # log_prob() needs sample shape in the beginning
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
            loss = loss + self._game_over_loss_weight * game_over_loss

        policy_loss = (loss_scale * policy_loss).sum(dim=1)
        loss = loss + self._policy_loss_weight * policy_loss
        entropy, entropy_for_gradient = dist_utils.entropy_with_fallback(
            model_output.action_distribution)
        if self._log_alpha is not None:
            alpha = self._log_alpha.exp().detach()
            loss = loss - alpha * (loss_scale * entropy_for_gradient).sum(
                dim=1)
            if self._target_entropy is not None:
                # For some unknown reason, there are memory leaks for not using
                # detach()
                self._log_alpha -= self._alpha_adjust_rate * (
                    entropy.mean() - self._target_entropy()).sign().detach()

        repr_loss = ()
        if self._train_repr_prediction:

            def _flatten(x):
                # there is no need to predict the first latent
                return x[:, 1:, ...].reshape(-1, *x.shape[2:])

            # [B*unroll_steps, ...]
            observation = alf.nest.map_structure(_flatten, target.observation)
            with torch.no_grad():
                target_repr = self._representation_net(observation)[0]
            # [B*unroll_steps, ...]
            repr = _flatten(model_output.state.state)
            # [B*unroll_steps]
            repr_loss = self.calc_repr_prediction_loss(repr, target_repr)
            # [B, unroll_steps]
            repr_loss = repr_loss.reshape(batch_size, -1)
            repr_loss = repr_loss.mean(dim=1)
            loss = loss + self._repr_prediction_loss_weight * repr_loss

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar(
                    "explained_variance_of_value0",
                    tensor_utils.explained_variance(model_output.value[:, 0],
                                                    target.value[:, 0]))
                alf.summary.scalar(
                    "explained_variance_of_value1",
                    tensor_utils.explained_variance(
                        model_output.value[:, 1:], target.value[:, 1:],
                        dim=0).mean())
                if self._train_reward_function:
                    alf.summary.scalar(
                        "explained_variance_of_reward0",
                        tensor_utils.explained_variance(
                            reward[:, 0], target_reward[:, 0]))
                    alf.summary.scalar(
                        "explained_variance_of_reward1",
                        tensor_utils.explained_variance(
                            reward[:, 1:], target_reward[:, 1:], dim=0).mean())
                    summary_utils.add_mean_hist_summary(
                        "predicted_reward", reward)
                    summary_utils.add_mean_hist_summary(
                        "target_reward", target_reward)
                if self._train_game_over_function:

                    def _entropy(events):
                        p = events.to(torch.float32).mean()
                        p = torch.tensor([p, 1 - p])
                        return -(p * (p + 1e-30).log()).sum(), p[0]

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
                summary_utils.add_mean_hist_summary("entropy0", entropy[:, 0])
                summary_utils.add_mean_hist_summary("entropy1", entropy[:, 1:])
                summary_utils.summarize_distribution(
                    "action_dist", model_output.action_distribution)
                if self._target_entropy is not None:
                    alf.summary.scalar("alpha", alpha)

        return LossInfo(
            loss=loss,
            extra=dict(
                value=value_loss,
                reward=reward_loss,
                policy=policy_loss,
                repr_prediction=repr_loss,
                game_over=game_over_loss))

    def calc_repr_prediction_loss(self, repr, target_repr):
        """Calculate the loss given the predicted representation and target representation"""
        raise NotImplementedError

    def prediction_model(self, state) -> ModelOutput:
        raise NotImplementedError


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
    net = EncodingNetwork(
        input_tensor_spec,
        input_preprocessors=(None, preproc),
        preprocessing_combiner=alf.nest.utils.NestConcat(),
        fc_layer_params=(256, 256),
        last_layer_size=input_tensor_spec[0].numel,
        last_activation=torch.relu_)
    return alf.nn.Sequential(net, alf.math.normalize_min_max)


@alf.configurable
class SimplePredictionNet(alf.networks.Network):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 trunk_net_ctor,
                 num_quantiles=1,
                 discrete_projection_net_ctor=CategoricalProjectionNetwork,
                 continuous_projection_net_ctor=StableNormalProjectionNetwork,
                 initial_game_over_bias=0.0):
        """
        Args:
            observation_spec (TensorSpec): describing the observation.
            action_spec (BoundedTensorSpec): describing the action.
            trunk_net_ctor (Callable): called as ``trunk_net_ctor(input_tensor_spec=observation_spec)``
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
            dim, num_quantiles, kernel_initializer=torch.nn.init.zeros_)
        self._reward_layer = alf.layers.FC(
            dim, num_quantiles, kernel_initializer=torch.nn.init.zeros_)

        if action_spec.is_continuous:
            self._action_net = continuous_projection_net_ctor(
                input_size=dim, action_spec=action_spec)
        else:
            self._action_net = discrete_projection_net_ctor(
                input_size=dim, action_spec=action_spec)

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
    net = EncodingNetwork(
        input_tensor_spec=observation_spec, fc_layer_params=(256, 256))
    return alf.nn.Sequential(net, alf.math.normalize_min_max)


@alf.configurable
def create_repr_projection_net(input_tensor_spec,
                               hidden_size=128,
                               output_size=256):
    return EncodingNetwork(
        input_tensor_spec,
        input_preprocessors=alf.layers.Reshape(-1),
        fc_layer_params=(hidden_size, hidden_size),
        use_fc_bn=True,
        activation=torch.relu_,
        last_layer_size=output_size,
        last_activation=alf.math.identity,
        last_use_fc_bn=False)


@alf.configurable
def create_repr_prediction_net(input_tensor_spec,
                               hidden_size=128,
                               output_size=256):
    return EncodingNetwork(
        input_tensor_spec,
        fc_layer_params=(hidden_size, ),
        use_fc_bn=True,
        activation=torch.relu_,
        last_layer_size=output_size,
        last_activation=alf.math.identity,
        last_use_fc_bn=False)


@alf.configurable
class SimpleMCTSModel(MCTSModel):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 num_unroll_steps,
                 num_sampled_actions=None,
                 encoding_net_ctor=create_simple_encoding_net,
                 dynamics_net_ctor=create_simple_dynamics_net,
                 prediction_net_ctor=create_simple_prediction_net,
                 game_over_logit_thresh=1.0,
                 initial_alpha=0.0,
                 target_entropy=None,
                 alpha_adjust_rate=0.001,
                 train_reward_function=True,
                 train_game_over_function=True,
                 train_repr_prediction=False,
                 repr_projection_net_ctor=create_repr_projection_net,
                 repr_prediction_net_ctor=create_repr_prediction_net,
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
            initial_alpha (float): initial value for the weight of entropy regularization
            target_entropy (float): if provided, will adjust alpha automatically
                so that the entropy is not smaller than this.
            alpha_adjust_rate (float): the speed to adjust alpha
            train_reward_function (bool): whether to predict reward
            train_game_over_function (bool): whether to predict game over
            train_repr_prediction (bool): whether to train to predict future
                latent representation. This implements the self-supervised
                consistency loss described in `Ye et. al. Mastering Atari Games
                with Limited Data <https://arxiv.org/abs/2111.00210>`_. The loss
                is ``-cosine(prediction_net(projection_net(x)), projection_net(y))``,
                where x is the representation calcuated by dynamics_net and
                y is the representation calcualted by representation_net
                from the corresponding future observations.
            repr_projection_net_ctor (Callable): called as
                ``repr_projection_net_ctor(repr_spec)`` to construct the
                projection_net described above
            repr_prediction_net_ctor (Callable): called as
                ``repr_prediction_net_ctor(projection_net.output_spec)`` to
                construct the prediction_net described above
        """
        encoding_net = encoding_net_ctor(observation_spec)
        repr_spec = encoding_net.output_spec
        dynamics_net = dynamics_net_ctor(
            input_tensor_spec=(repr_spec, action_spec))
        prediction_net = prediction_net_ctor(repr_spec, action_spec)
        super().__init__(
            num_unroll_steps=num_unroll_steps,
            representation_net=encoding_net,
            dynamics_net=dynamics_net,
            prediction_net=prediction_net,
            train_repr_prediction=train_repr_prediction,
            train_reward_function=train_reward_function,
            train_game_over_function=train_game_over_function,
            initial_alpha=initial_alpha,
            target_entropy=target_entropy,
            alpha_adjust_rate=alpha_adjust_rate,
            debug_summaries=debug_summaries,
            name=name)
        self._num_sampled_actions = num_sampled_actions

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
        if train_repr_prediction:
            self._repr_projection_net = repr_projection_net_ctor(repr_spec)
            self._repr_prediction_net = repr_prediction_net_ctor(
                self._repr_projection_net.output_spec)

    def calc_repr_prediction_loss(self, repr, target_repr):
        if alf.summary.should_record_summaries():
            repr = summary_utils.summarize_tensor_gradients(
                "SimpleMCTSModel/repr_grad", repr, clone=True)
        with torch.no_grad():
            projected_target_repr = self._repr_projection_net(target_repr)[0]
            projected_target_repr = F.normalize(
                projected_target_repr.detach(), dim=1)
        projected_repr = self._repr_projection_net(repr)[0]
        predicted_projected_repr = self._repr_prediction_net(projected_repr)[0]
        predicted_projected_repr = F.normalize(predicted_projected_repr, dim=1)
        return -(projected_target_repr * predicted_projected_repr).sum(dim=1)

    def prediction_model(self, state):
        # TODO: transform reward/value and use softmax to estimate the value and
        # reward as in appendix F.
        value, reward, action_distribution, game_over_logit = self._prediction_net(
            state)[0]
        value_quantiles = ()
        if value.ndim == 2:
            # assuming it is a quantile model
            if self.training:
                value_quantiles = value
            value = value.mean(dim=-1)

        reward_quantiles = ()
        if isinstance(reward, torch.Tensor) and reward.ndim == 2:
            # assuming it is a quantile model
            if self.training:
                reward_quantiles = reward
            reward = reward.mean(dim=-1)

        if self._sample_actions:
            # [num_sampled_actions, B, ...]
            actions = action_distribution.rsample(
                (self._num_sampled_actions, ))
            # [B, num_sampled_actions, ...]
            actions = actions.transpose(0, 1)
            # According to the following paper, we should use 1/K as action_probs
            # for sampled actions.
            # Hubert et. al. Learning and Planning in Complex Action Spaces, 2021
            action_probs = torch.ones(
                actions.shape[:2]) / self._num_sampled_actions
        else:
            action_probs = action_distribution.probs
            if self._num_sampled_actions is None:
                actions = ()
            else:
                action_probs, actions = action_probs.topk(
                    self._num_sampled_actions, sorted=False)
                action_probs = action_probs / action_probs.sum(
                    dim=-1, keepdim=True)

        if not self._train_reward_function:
            reward = ()
        if not self._train_game_over_function:
            game_over = ()
            game_over_logit = ()
        else:
            game_over = game_over_logit > self._game_over_logit_thresh

        return ModelOutput(
            value=value,
            value_quantiles=value_quantiles,
            reward=reward,
            reward_quantiles=reward_quantiles,
            game_over=game_over,
            actions=actions,
            action_probs=action_probs,
            state=state,
            action_distribution=action_distribution,
            game_over_logit=game_over_logit)
