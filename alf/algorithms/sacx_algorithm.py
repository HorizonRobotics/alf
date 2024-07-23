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
"""Soft Actor Critic Algorithm."""

import copy
from typing import Any, NamedTuple

import torch
import torch.distributions as td
from torch import Tensor

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.ppo_algorithm import PPOLoss
from alf.algorithms.td_loss import TDLoss
from alf.data_structures import TimeStep, LossInfo
from alf.data_structures import AlgStep, StepType, make_experience
from alf.nest import nest
from alf.nest.utils import convert_device
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, dist_utils, math_ops
from alf.utils import value_ops, tensor_utils
from alf.utils.summary_utils import safe_mean_hist_summary

NestedTensor = Any


class SacXState(NamedTuple):
    actor: NestedTensor = ()
    actor_critic: NestedTensor = ()  # the critic state for training actor
    critic: NestedTensor = ()
    value: NestedTensor = ()


class SacXInfo(NamedTuple):
    reward: Tensor = ()
    step_type: Tensor = ()
    discount: Tensor = ()
    returns: Tensor = ()
    advantages: Tensor = ()
    reward_weights: Tensor = ()
    value: Tensor = ()
    critic: Tensor = ()
    action: Tensor = ()
    actor: LossInfo = ()
    action_distribution: td.Distribution = ()
    rollout_log_prob: Tensor = ()
    rollout_action_distribution: td.Distribution = ()


@alf.configurable
class SacXAlgorithm(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_ctor=ActorDistributionNetwork,
                 critic_network_ctor=CriticNetwork,
                 value_network_ctor=None,
                 reward_weights=None,
                 epsilon_greedy=None,
                 num_critic_replicas=2,
                 dqda_clipping=None,
                 optimizer=None,
                 critic_optimizer=None,
                 loss_ctor=PPOLoss,
                 mode='ppo',
                 kld_weight=1.0,
                 num_additional_updates_for_critic=0,
                 num_critic_steps=0,
                 deterministic=False,
                 config: TrainerConfig = None,
                 debug_summaries=False,
                 name="SacX"):
        actor_network = actor_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        if mode != 'ppo':
            critic_network = critic_network_ctor(
                input_tensor_spec=(observation_spec, action_spec))
            critic_networks = critic_network.make_parallel(
                num_critic_replicas * reward_spec.numel)
        else:
            critic_networks = None
        value_network = value_network_ctor(input_tensor_spec=observation_spec)
        if reward_spec.numel > 1:
            value_network = value_network.make_parallel(
                reward_spec.numel)  # value->[B,n]

        train_state_spec = SacXState(
            actor=actor_network.state_spec,
            critic=critic_networks.state_spec if critic_networks else (),
            actor_critic=critic_networks.state_spec if critic_networks else (),
            value=value_network.state_spec)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=train_state_spec._replace(critic=()),
            predict_state_spec=SacXState(actor=actor_network.state_spec),
            reward_weights=reward_weights,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy
        self._dqda_clipping = dqda_clipping

        self._num_critic_replicas = num_critic_replicas
        self._actor_network = actor_network
        self._critic_networks = critic_networks
        self._value_network = value_network
        self._ppo_loss = loss_ctor(
            pg_only=True, debug_summaries=debug_summaries)
        self._td_loss = SacXTDLoss(
            gamma=self._ppo_loss.gamma,
            td_error_loss_fn=self._ppo_loss._td_error_loss_fn,
            td_lambda=self._ppo_loss._lambda,
            debug_summaries=True,
            name="SaxTDLoss")

        self._kld_weight = kld_weight
        self._mode = mode
        self._num_critic_steps = num_critic_steps
        self._num_update_steps = 0
        self._first_train_step = False

        if deterministic:
            assert mode == 'dqda', "deterministic should only be used with mode='dqda'"
        self._deterministic = deterministic

        self._critic_trainer = None
        if num_additional_updates_for_critic > 0:
            critic_trainer = SacXCriticTrainer(
                observation_spec=observation_spec,
                action_spec=action_spec,
                critic_networks=critic_networks,
                value_network=value_network,
                loss=self._td_loss,
                optimizer=critic_optimizer,
                num_updates_per_train_iter=num_additional_updates_for_critic,
                config=config,
                reward_spec=reward_spec)
            self._critic_trainer = setup_ddp(critic_trainer)

    def _trainable_attributes_to_ignore(self):
        return ['_critic_trainer']

    def predict_step(self, inputs: TimeStep, state: SacXState):
        """Predict for one step."""
        action_dist, actor_state = self._actor_network(
            inputs.observation, state=state.actor)

        action = dist_utils.epsilon_greedy_sample(action_dist,
                                                  self._epsilon_greedy)
        return AlgStep(
            output=action,
            state=SacXState(actor=actor_state),
            info=SacXInfo(action_distribution=action_dist))

    def rollout_step(self, inputs: TimeStep, state: SacXState):
        action_dist, actor_state = self._actor_network(
            inputs.observation, state=state.actor)

        if self._deterministic:
            action = alf.nest.map_structure(dist_utils.get_rmode, action_dist)
            log_prob = ()
        else:
            action, log_prob = dist_utils.sample_action_distribution(
                action_dist, return_log_prob=True)

        value, value_state = self._value_network(inputs.observation,
                                                 state.value)

        alg_step = AlgStep(
            output=action,
            state=SacXState(actor=actor_state, value=value_state),
            info=SacXInfo(
                reward=inputs.reward,
                step_type=inputs.step_type,
                discount=inputs.discount,
                action=action,
                value=value,
                rollout_log_prob=log_prob,
                action_distribution=action_dist))

        if self._critic_trainer is not None:
            self._first_train_step = True
            experience_for_observe = make_experience(inputs, alg_step, state)
            self._critic_trainer.observe_for_replay(experience_for_observe)

        return alg_step

    def _apply_reward_weights(self, critics):
        critics = critics * self.reward_weights
        critics = critics.sum(dim=-1)
        return critics

    def _compute_critics(self,
                         observation,
                         action,
                         critics_state,
                         replica_min=True,
                         apply_reward_weights=True):
        observation = (observation, action)
        critics, critics_state = self._critic_networks(
            observation, state=critics_state)
        # For multi-dim reward, do
        #   [B, replicas * reward_dim] -> [B, replicas, reward_dim]
        # For scalar reward, do nothing
        if self.has_multidim_reward():
            critics = critics.reshape(-1, self._num_critic_replicas,
                                      *self._reward_spec.shape)

        if replica_min:
            if self.has_multidim_reward():
                sign = self.reward_weights.sign()
                critics = (critics * sign).min(dim=1)[0] * sign
            else:
                critics = critics.min(dim=1)[0]

        if apply_reward_weights and self.has_multidim_reward():
            critics = self._apply_reward_weights(critics)

        # The returns have the following shapes in different circumstances:
        # [replica_min=True, apply_reward_weights=True]: critics shape [B]
        # [replica_min=True, apply_reward_weights=False]: critics shape [B, reward_dim]
        # [replica_min=False, apply_reward_weights=False]: critics shape [B, replicas, reward_dim]
        return critics, critics_state

    def _critic_train_step(self, observation, rollout_action, state):
        critics, critic_state = self._compute_critics(
            observation,
            rollout_action,
            state.critic,
            replica_min=False,
            apply_reward_weights=False)

        return critics, state._replace(critic=critic_state)

    def _actor_train_step(self, observation, action, rollout_info, state):
        action_dist, actor_state = self._actor_network(
            observation, state=state.actor)
        if self._deterministic:
            action = alf.nest.map_structure(dist_utils.get_rmode, action_dist)
        else:
            action = dist_utils.rsample_action_distribution(action_dist)
        if self._mode != 'dqda':
            return action, action_dist, (), state._replace(actor=actor_state)

        q_value, actor_critic_state = self._compute_critics(
            observation,
            action,
            state.actor_critic,
            apply_reward_weights=True,
            replica_min=True)

        q_value = q_value / q_value.std(dim=0, keepdim=True).detach()
        dqda = nest_utils.grad(action, q_value.sum())

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        dqda_loss = nest.map_structure(actor_loss_fn, dqda, action)
        dqda_loss = math_ops.add_n(nest.flatten(dqda_loss))
        kld = alf.nest.map_structure(
            td.kl_divergence, rollout_info.action_distribution, action_dist)
        actor_loss = LossInfo(
            loss=dqda_loss + self._kld_weight * sum(alf.nest.flatten(kld)),
            extra={
                'dqda_loss': dqda_loss,
                'kld': kld,
            })

        return action, action_dist, actor_loss, state._replace(
            actor=actor_state, actor_critic=actor_critic_state)

    def train_step(self, inputs: TimeStep, state: SacXState,
                   rollout_info: SacXInfo):
        if self._first_train_step:
            self._critic_trainer.train_from_replay_buffer()
            self._first_train_step = False
        observation = inputs.observation

        if self._num_update_steps >= self._num_critic_steps:
            action, action_dist, actor_loss, state = self._actor_train_step(
                observation, state, rollout_info, state)
        else:
            action_dist = rollout_info.action_distribution
            action = alf.nest.map_structure(torch.zeros_like,
                                            rollout_info.action)
            zeros = torch.zeros_like(inputs.discount)
            actor_loss = LossInfo(
                loss=zeros, extra={
                    'dqda_loss': zeros,
                    'kld': zeros
                })

        if self._critic_networks is not None:
            critic, state = self._critic_train_step(observation,
                                                    rollout_info.action, state)
        else:
            critic = ()

        value, value_state = self._value_network(observation, state)
        state = state._replace(value=value_state)

        if self.has_multidim_reward():
            reward_weights = tensor_utils.tensor_extend_new_dim(
                self.reward_weights, dim=0, n=value.shape[0])
        else:
            reward_weights = ()

        info = SacXInfo(
            returns=rollout_info.returns,
            advantages=rollout_info.advantages,
            value=value,
            critic=critic,
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            action_distribution=action_dist,
            actor=actor_loss,
            rollout_log_prob=rollout_info.rollout_log_prob,
            rollout_action_distribution=rollout_info.action_distribution,
            reward_weights=reward_weights)
        return AlgStep(action, state, info)

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        return _preprocess_experience(self, root_inputs, rollout_info,
                                      self._td_loss)

    def calc_loss(self, info: SacXInfo):
        """Calculate loss."""
        extra = {}
        td_loss = self._td_loss(info)
        extra.update(td_loss.extra)

        if self._num_update_steps >= self._num_critic_steps:
            if self._mode != 'dqda':
                if self._mode == 'q_adv_pg':
                    critics = info.critic
                    if self.has_multidim_reward():
                        sign = self.reward_weights.sign()
                        critics = (critics * sign).min(dim=2)[0] * sign
                    else:
                        critics = critics.min(dim=2)[0]
                    info = info._replace(
                        advantages=(critics - info.value).detach())
                ppo_loss = self._ppo_loss(info)
                loss = ppo_loss.loss
                extra.update(ppo_loss.extra._asdict())
            else:
                loss = info.actor.loss
                extra.update(info.actor.extra)
            loss = loss + td_loss.loss
        else:
            loss = td_loss.loss

        self._num_update_steps += 1
        return LossInfo(loss=loss, extra=extra)

    def after_train_iter(self, inputs: TimeStep, info: SacXInfo):
        self._num_update_steps = 0


class SacXCriticTrainer(OffPolicyAlgorithm):
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 critic_networks,
                 value_network,
                 loss,
                 num_updates_per_train_iter,
                 config: TrainerConfig,
                 optimizer: torch.optim.Optimizer,
                 reward_spec=TensorSpec(()),
                 name='SacXCriticTrainer'):

        config = copy.copy(config)
        config.data_transformer_ctor = None
        config.num_updates_per_train_iter = num_updates_per_train_iter
        config.mask_out_loss_for_last_step = True

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=SacXState(
                critic=critic_networks.state_spec if critic_networks else (),
                value=value_network.state_spec),
            optimizer=optimizer,
            config=config,
            name=name)
        self._critic_networks = critic_networks
        self._value_network = value_network
        self._loss = loss
        if critic_networks is not None:
            self._num_critic_replicas = critic_networks.output_spec.numel // reward_spec.numel

    def _set_replay_buffer(self, sample_exp):
        self._replay_buffer_max_length = self._config.unroll_length + 1
        self._replay_buffer_num_envs = sample_exp.step_type.shape[0]
        self._prioritized_sampling = False
        super()._set_replay_buffer(sample_exp)

    def train_step(self, inputs, state: SacXState, rollout_info: SacXInfo):
        if self._critic_networks is not None:
            critic, critic_state = self._critic_networks(
                (inputs.observation, rollout_info.action), state=state.critic)
            if self.has_multidim_reward():
                critic = critic.reshape(-1, self._num_critic_replicas,
                                        *self._reward_spec.shape)
        else:
            critic = ()
            critic_state = ()
        value, value_state = self._value_network(inputs.observation, state)
        state = SacXState(critic=critic_state, value=value_state)

        info = SacXInfo(
            returns=rollout_info.returns,
            value=value,
            critic=critic,
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount)
        return AlgStep(rollout_info.action, state, info)

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info,
                              batch_info):
        return _preprocess_experience(self, root_inputs, rollout_info,
                                      self._loss)

    def calc_loss(self, info):
        return self._loss(info)

    def train_from_replay_buffer(self):
        summary_enabled = alf.summary.is_summary_enabled()
        alf.summary.enable_summary(True)
        with alf.summary.scope(self._name):
            super().train_from_replay_buffer(update_global_counter=False)
        alf.summary.enable_summary(summary_enabled)


def setup_ddp(algorithm):
    from alf.utils.per_process_context import PerProcessContext
    ddp_rank = PerProcessContext().ddp_rank
    if ddp_rank >= 0:
        # Activate the DDP training
        algorithm.activate_ddp(ddp_rank)
        # Make sure the BN statistics of different processes are synced
        # https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm
        # This conversion needs to be performed before wrapping modules with DDP.
        algorithm = alf.utils.common.convert_sync_batchnorm(algorithm)
    return algorithm


class SacXTDLoss(TDLoss):
    def forward(self, info: SacXInfo):
        value_loss = self._calc_value_loss(info)
        if info.critic == ():
            return LossInfo(loss=value_loss, extra={'value': value_loss})
        critic_loss = self._calc_critic_loss(info)
        return LossInfo(
            loss=value_loss + critic_loss,
            extra={
                'value': value_loss,
                'critic': critic_loss
            })

    def _calc_critic_loss(self, info: SacXInfo):
        value = info.critic
        returns = info.returns
        returns = returns[:, :, None, ...].expand_as(value)

        td_error = returns - value

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('critics' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("critic_td_error" + suffix, td,
                                           mask)

                num_critic_replicas = value.shape[2]
                for r in range(num_critic_replicas):
                    if value.ndim == 3:
                        _summarize(value[..., r], returns[..., r],
                                   td_error[..., r], f'/replica_{r}')
                    else:
                        for i in range(value.shape[3]):
                            suffix = f'/replica_{r}/{i}'
                            _summarize(value[..., r, i], returns[..., r, i],
                                       td_error[..., r, i], suffix)

        loss = self._td_error_loss_fn(returns, value)
        loss = loss.reshape(*loss.shape[:2], -1).mean(-1)

        return loss

    def _calc_value_loss(self, info: SacXInfo):
        value = info.value
        returns = info.returns
        td_error = returns - value

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('values' + suffix, v, mask)
                    safe_mean_hist_summary("value_td_error" + suffix, td, mask)

                if value.ndim == 2:
                    _summarize(value, returns, td_error, '/value')
                else:
                    for i in range(value.shape[2]):
                        suffix = f'/{i}'
                        _summarize(value[..., i], returns[..., i],
                                   td_error[..., i], suffix)

        loss = self._td_error_loss_fn(returns, value)
        loss = loss.reshape(*loss.shape[:2], -1).mean(-1)
        return loss


def _preprocess_experience(self, root_inputs: TimeStep, rollout_info, loss):
    """Compute advantages and put it into exp.rollout_info."""

    # The device of rollout_info can be different from the default device
    # when ReplayBuffer.gather_all.convert_to_default_device is configured
    # to False to save gpu memory.
    step_type = convert_device(rollout_info.step_type)
    discount = convert_device(rollout_info.discount)
    reward = convert_device(rollout_info.reward)
    value = convert_device(rollout_info.value)

    if rollout_info.reward.ndim == 3:
        # [B, T, D] or [B, T, 1]
        discounts = discount.unsqueeze(-1) * loss.gamma
    else:
        # [B, T]
        discounts = discount * loss.gamma

    advantages = value_ops.generalized_advantage_estimation(
        rewards=reward,
        values=value,
        step_types=step_type,
        discounts=discounts,
        td_lambda=loss._lambda,
        time_major=False)
    advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)
    returns = value + advantages
    return root_inputs, rollout_info._replace(
        returns=returns, advantages=advantages)
