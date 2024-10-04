# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from typing import Callable, NamedTuple
import torch
import alf

from alf.algorithms.actor_critic_loss import normalize, ActorCriticLossInfo
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.data_structures import LossInfo, AlgStep, StepType, TimeStep
from alf.nest.utils import convert_device
from alf.networks import ActorDistributionNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, dist_utils, tensor_utils, value_ops
from alf.utils.losses import element_wise_squared_loss
from alf.utils.model_averager import create_averaged_model
from alf.utils.summary_utils import safe_mean_hist_summary


class PartialValueNetwork(alf.nn.Network):
    """
    Let the shape of action be (L, A), ParitalValueNetwork outputs L values

    Q(s, a[:i]) for i=0, 1, ..., A-1


    Args:

        fc_layer_params (tuple[int]): a tuple of integers representing hidden
            FC layer sizes.
        activation (nn.functional): activation used for hidden layers. The
            last layer will not be activated.
        use_fc_bn (bool): whether use Batch Normalization for the internal
            FC layers (i.e. FC layers beside the last one).
        use_fc_ln (bool): whether use Layer Normalization for the internal
            fc layers (i.e. FC layers except the last one).
    """

    def __init__(self,
                 observation_spec,
                 reward_dim,
                 action_factor_dim,
                 num_action_factors,
                 observation_preprocessor=None,
                 fc_layer_params=(),
                 activation=torch.relu_,
                 cumsum=False,
                 use_fc_bn=False,
                 use_fc_ln=False):
        super().__init__(
            input_tensor_spec=(observation_spec,
                               alf.TensorSpec((num_action_factors,
                                               action_factor_dim))))

        obs_preprocessor = observation_preprocessor or alf.layers.Identity()
        self._obs_preprocessor = alf.nn.wrap_as_network(
            obs_preprocessor, input_tensor_spec=self._input_tensor_spec[0])
        assert self._obs_preprocessor.output_spec.ndim == 1
        obs_dim = self._obs_preprocessor.output_spec.numel
        n = num_action_factors * reward_dim
        dim = fc_layer_params[0]
        self._fc_obs = alf.layers.FC(obs_dim, dim).make_parallel(n)
        self._fc_action = alf.layers.FC(action_factor_dim * num_action_factors,
                                        dim).make_parallel(n)
        self._fc_mask = alf.layers.FC(num_action_factors, dim).make_parallel(n)
        fcs = []
        for i in range(1, len(fc_layer_params)):
            fc = alf.layers.FC(
                dim,
                fc_layer_params[i],
                activation=activation,
                use_bn=use_fc_bn,
                use_ln=use_fc_ln)
            fcs.append(fc.make_parallel(num_action_factors * reward_dim))
            dim = fc_layer_params[i]
        fc = alf.layers.FC(dim, 1, kernel_init_gain=0.0)
        fcs.append(fc.make_parallel(num_action_factors * reward_dim))
        self._fcs = torch.nn.Sequential(*fcs)
        self._mask = torch.ones(
            num_action_factors,
            num_action_factors).tril_(diagonal=-1).unsqueeze(-1)
        self._num_action_factors = num_action_factors
        self._reward_dim = reward_dim
        self._cumsum = cumsum
        self._activation = activation

    def forward(self, inputs, state):
        obs, action = inputs
        obs = self._obs_preprocessor(obs)[0]
        B = obs.shape[0]
        L = self._num_action_factors
        R = self._reward_dim
        action = action.reshape(B, -1, L).transpose(1, 2)
        action = action[:, None, :, :]  # [B, 1, L, D]
        action = action * self._mask  # [B, L, L, D]
        action = action.reshape(B, L, 1, -1).expand(B, L, R, -1).reshape(
            B, L * R, -1)
        mask = self._mask.reshape(1, L, 1, L).expand(B, L, R, L).reshape(
            B, L * R, L)
        obs = obs[:, None, :].expand(B, L * R, -1)
        x = self._fc_obs(obs) + self._fc_action(action) + self._fc_mask(mask)
        x = self._activation(x)
        out = self._fcs(x)
        out = out.reshape(B, L, R)
        if self._cumsum:
            out = out.cumsum(dim=1)
        return out, state


class FactoredPPOInfo(NamedTuple):
    step_type: torch.Tensor = ()  # [B]
    discount: torch.Tensor = ()  # [B]
    reward: torch.Tensor = ()  # [B, R]
    action: torch.Tensor = ()  # [B, L, A]
    rollout_log_prob: torch.Tensor = ()  # [B, L]
    log_prob: torch.Tensor = ()  # [B, L]
    rollout_action_distribution = ()  # Not used
    action_distribution: torch.distributions.Distribution = ()  # [B, A*L]
    factored_action_distribution: torch.distributions.Distribution = (
    )  # [B, L, A]
    returns: torch.Tensor = ()  # [B, L, R]
    advantages: torch.Tensor = ()  # [B, L, R]
    value: torch.Tensor = ()  # [B, L, R]
    reward_weights: torch.Tensor = ()  # [B, R]
    normalized_advantages: torch.Tensor = ()  # [B, L, R]


class FactoredPPOState(NamedTuple):
    actor: torch.Tensor = ()
    value: torch.Tensor = ()


@alf.configurable
class FactoredPPOLoss(PPOLoss):
    def forward(self, info: FactoredPPOInfo):
        # batch_size, batch_length, num_action_factors, reward_dim
        T, B, L, R = info.value.shape

        value = info.value
        returns = info.returns
        advantages = info.advantages

        # [T, B, L, R]
        td_loss = self._td_error_loss_fn(returns.detach(), value)
        td_loss = td_loss.mean(dim=3)  # [T, B, L]
        pg_loss = self._pg_loss(
            info._replace(
                action_distribution=info.factored_action_distribution),
            info.normalized_advantages)
        loss = pg_loss + self._td_loss_weight * td_loss  # [T, B, L]

        entropy_loss = ()
        if self._entropy_regularization is not None:
            entropy, entropy_for_gradient = dist_utils.entropy_with_fallback(
                info.factored_action_distribution, return_sum=False)
            entropy_loss = alf.nest.map_structure(lambda x: -x, entropy)
            loss -= self._entropy_regularization * sum(
                alf.nest.flatten(entropy_for_gradient))

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):

                def _summarize(v, r, adv, suffix):
                    alf.summary.scalar("values" + suffix, v.mean())
                    alf.summary.scalar("returns" + suffix, r.mean())
                    safe_mean_hist_summary('advantages' + suffix, adv)
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r))

                for i in range(value.shape[3]):
                    suffix = '/' + str(i)
                    _summarize(value[..., i], returns[..., i],
                               advantages[..., i], suffix)

        loss_info = LossInfo(
            loss=loss,
            extra=ActorCriticLossInfo(
                td_loss=td_loss, pg_loss=pg_loss, neg_entropy=entropy_loss))

        loss_info = alf.nest.map_structure(
            lambda x: x.reshape(T, L, B).mean(dim=1), loss_info)
        return loss_info


@alf.configurable
class FactoredPPOAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 num_action_factors,
                 reward_spec=TensorSpec(()),
                 reward_weights=None,
                 actor_network_ctor: Callable = ActorDistributionNetwork,
                 value_network_ctor: Callable = PartialValueNetwork,
                 distribution_adapter_ctor=None,
                 epsilon_greedy=None,
                 env=None,
                 config: TrainerConfig = None,
                 loss_class: Callable = FactoredPPOLoss,
                 predict_average_type: str = "none",
                 optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 name="ActorCriticAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
                action_spec.ndim must be 1. And the action can be reshaped as
                `[batch_size, action_dim, num_action_factors]`.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the v values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the v values is used.
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            actor_network_ctor (Callable): Function to construct the actor network.
                ``actor_network_ctor`` needs to accept ``input_tensor_spec`` and
                ``action_spec`` as its arguments and return an actor network.
                The constructed network will be called with ``forward(observation, state)``.
            value_network_ctor (None | Callable): Function to construct the value network.
                ``value_network_ctor`` needs to accept ``input_tensor_spec`` as its
                arguments and return a value network. The constructed network will be
                called with ``forward(observation, state)`` and returns value tensor for
                each observation given observation and network state. Note that if the
                algorithm is constructed for evaluation or deployment only, the
                value_network_ctor can be set to None and the value network will not be
                constructed at all.
            loss_ctor: the class of the loss. The signature of its
                constructor: ``loss_class(debug_summaries)``
            optimizer (torch.optim.Optimizer): The optimizer for training
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.

        """
        assert action_spec.ndim <= 1
        assert action_spec.numel % num_action_factors == 0
        assert action_spec.is_continuous
        assert distribution_adapter_ctor is None, "Not supported yet"

        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy
        actor_network = actor_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        value_network = value_network_ctor(
            observation_spec=observation_spec,
            reward_dim=reward_spec.numel,
            action_factor_dim=action_spec.numel // num_action_factors,
            num_action_factors=num_action_factors)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            reward_weights=reward_weights,
            predict_state_spec=FactoredPPOState(
                actor=actor_network.state_spec),
            train_state_spec=FactoredPPOState(
                actor=actor_network.state_spec,
                value=value_network.state_spec),
            env=env,
            config=config,
            optimizer=optimizer,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._value_network = value_network
        loss = loss_class(
            reward_dim=reward_spec.numel, debug_summaries=debug_summaries)
        self._loss = loss

        self._predict_model = create_averaged_model(self._actor_network,
                                                    predict_average_type)
        # num_features of adv_norm created by ActorCriticLoss is 1
        # we need to use num_features=num_action_factors
        self._adv_norm = torch.nn.BatchNorm1d(
            num_features=num_action_factors,
            eps=1e-8,
            momentum=self._loss._adv_norm.momentum,
            affine=False,
            track_running_stats=self._loss._adv_norm.track_running_stats)
        self._num_action_factors = num_action_factors

    def after_update(self, root_inputs: TimeStep, info: FactoredPPOInfo):
        if self._predict_model != self._actor_network:
            self._predict_model.update_parameters(self._actor_network)

    def _trainable_attributes_to_ignore(self):
        return ['_predict_model']

    def convert_train_state_to_predict_state(self, state):
        return state._replace(value=())

    def predict_step(self, inputs: TimeStep, state: FactoredPPOState):
        """Predict for one step."""
        B = inputs.observation.shape[0]
        L = self.action_spec.shape[0]

        action_dist, actor_state = self._predict_model(
            inputs.observation, state=state.actor)

        action = dist_utils.epsilon_greedy_sample(action_dist,
                                                  self._epsilon_greedy)
        return AlgStep(
            output=action,
            state=FactoredPPOState(actor=actor_state),
            info=FactoredPPOInfo(action_distribution=action_dist))

    def _step(self, inputs: TimeStep, state: FactoredPPOState):
        """Rollout for one step."""
        B = inputs.observation.shape[0]
        L = self._num_action_factors

        action_distribution, actor_state = self._actor_network(
            inputs.observation, state=state.actor)

        # The shape of action_distribution is [B, A * L]. We need to reshape it
        # to [B, L, A]
        builder, params = dist_utils._get_builder(action_distribution)
        params = alf.nest.map_structure(
            lambda x: x.reshape(B, -1, L).transpose(1, 2), params)
        factored_action_distribution = builder(**params)

        action, log_prob = dist_utils.sample_action_distribution(
            factored_action_distribution, return_log_prob=True)

        value, value_state = self._value_network((inputs.observation, action),
                                                 state=state.value)

        if self.has_multidim_reward():
            reward_weights = tensor_utils.tensor_extend_new_dim(
                self.reward_weights, dim=0, n=value.shape[0])
        else:
            reward_weights = ()
        return AlgStep(
            output=action.transpose(1, 2).reshape(B, -1),
            state=FactoredPPOState(actor=actor_state, value=value_state),
            info=FactoredPPOInfo(
                action=common.detach(action),
                log_prob=common.detach(log_prob),
                value=value,
                step_type=inputs.step_type,
                reward=inputs.reward,
                discount=inputs.discount,
                action_distribution=action_distribution,
                factored_action_distribution=factored_action_distribution,
                reward_weights=reward_weights))

    def rollout_step(self, inputs: TimeStep, state: FactoredPPOState):
        return self._step(inputs, state)

    def train_step(self, inputs: TimeStep, state: FactoredPPOState,
                   rollout_info: FactoredPPOInfo):
        alg_step = self._step(inputs, state)
        return alg_step._replace(
            info=rollout_info._replace(
                step_type=alg_step.info.step_type,
                reward=alg_step.info.reward,
                discount=alg_step.info.discount,
                action_distribution=alg_step.info.action_distribution,
                factored_action_distribution=alg_step.info.
                factored_action_distribution,
                value=alg_step.info.value,
                reward_weights=alg_step.info.reward_weights))

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        info = rollout_info
        # batch_length, batch_size, num_action_factors, reward_dim
        B, T, L, R = info.value.shape

        # The device of rollout_info can be different from the default device
        # when ReplayBuffer.gather_all.convert_to_default_device is configured
        # to False to save gpu memory.
        info_step_type = convert_device(info.step_type)
        info_discount = convert_device(info.discount)
        info_reward = convert_device(info.reward)
        info_value = convert_device(info.value)

        step_type = torch.full((B, T, L),
                               StepType.MID,
                               dtype=info_step_type.dtype)
        step_type[:, :, 0] = torch.where(info_step_type == StepType.LAST,
                                         StepType.MID, info_step_type)
        step_type[:, :, -1] = torch.where(info_step_type == StepType.FIRST,
                                          StepType.MID, info_step_type)
        step_type = step_type.reshape(B, T * L)

        discount = info_discount[:, :, None, None]
        # need to clone so that the '*=' after it can work
        discount = discount.expand(-1, -1, L, R).clone()
        discount[:, :, -1, :] *= self._loss._gamma
        discount = discount.reshape(B, T * L, R)

        reward = torch.zeros((B, T, L, R), dtype=info_reward.dtype)
        reward[:, :, 0, :] = info_reward.reshape(B, T, R)
        reward = reward.reshape(B, T * L, R)

        value = info_value.reshape(B, T * L, R)

        # [B, T*L-1, R]
        advantages = value_ops.generalized_advantage_estimation(
            rewards=reward,
            values=value,
            step_types=step_type,
            discounts=discount,
            td_lambda=self._loss._lambda**(1 / L),
            time_major=False)
        # [B, T-1, L, R]
        advantages = advantages[:, :(T - 1) * L, :].reshape(B, T - 1, L, R)

        assert self._loss.normalizing_scalar_advantages

        if self.has_multidim_reward():
            # [B, T-1, L]
            scalar_advantages = (advantages * self.reward_weights).sum(-1)
        else:
            # [B, T-1, L]
            scalar_advantages = advantages.squeeze(-1)
        normalized_advantages = normalize(self._adv_norm,
                                          scalar_advantages.reshape(-1, L))
        normalized_advantages = normalized_advantages.reshape_as(
            scalar_advantages)
        normalized_advantages = tensor_utils.tensor_extend_zero(
            normalized_advantages, dim=1)

        advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)
        returns = info_value + advantages

        return root_inputs, FactoredPPOInfo(
            rollout_log_prob=rollout_info.log_prob,
            returns=returns,
            action=rollout_info.action,
            advantages=advantages,
            normalized_advantages=normalized_advantages,
        )

    def calc_loss(self, info: FactoredPPOInfo):
        """Calculate loss."""
        return self._loss(info)
