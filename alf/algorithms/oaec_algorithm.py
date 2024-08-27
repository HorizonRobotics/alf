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
"""Optimistic Actor and Epistemic Critic Algorithm."""

import numpy as np
import functools
import torch
import torch.distributions as td
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm

from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep
from alf.optimizers import AdamTF
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork
from alf.networks.param_networks import CriticDistributionParamNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops, summary_utils


OaecRolloutInfo = namedtuple(
    'OaecRolloutInfo', ["action", "mask", "reward_noise"], default_value=())
OaecCriticState = namedtuple("OaecCriticState",
                             ['critics', 'target_actor', 'target_critics'])
OaecCriticInfo = namedtuple("OaecCriticInfo", ["q_values", "target_q_values"])
OaecActorState = namedtuple("OaecActorState", ['actor', 'critics'])
OaecState = namedtuple("OaecState", ['actor', 'critics'])
OaecInfo = namedtuple(
    "OaecInfo", [
        "reward", "step_type", "discount", "action", "action_distribution",
        "actor_loss", "critic", "discounted_return"
    ],
    default_value=())
OaecLossInfo = namedtuple('OaecLossInfo', ('actor', 'critic'))


@alf.configurable
class OaecAlgorithm(OffPolicyAlgorithm):
    r"""Optimistic Actor and Epistemic Critic Algorithm. """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_ctor=CriticNetwork,
                 reward_weights=None,
                 reward_noise_scale=0.1,
                 epsilon_greedy=None,
                 calculate_priority=False,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 num_rollout_sampled_actions=10,
                 num_bootstrapped_critics=1,
                 critic_replicas_deepcopy=False,
                 bootstrap_mask_prob=0.8,
                 beta_ub=1.0,
                 beta_lb=0.5,
                 target_update_tau=0.05,
                 target_update_period=1,
                 rollout_random_action=0.,
                 dqda_clipping=None,
                 action_l2=0,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 debug_summaries=False,
                 name="OaecAlgorithm"):
        r"""
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            actor_network_ctor (Callable): Function to construct the actor network.
                ``actor_network_ctor`` needs to accept ``input_tensor_spec`` and
                ``action_spec`` as its arguments and return an actor network.
                The constructed network will be called with ``forward(observation, state)``.
            critic_network_ctor (Callable): Function to construct the critic
                network. ``critic_netwrok_ctor`` needs to accept ``input_tensor_spec``
                which is a tuple of ``(observation_spec, action_spec)``. The
                constructed network will be called with
                ``forward((observation, action), state)``.
            reward_weights (list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor.
            reward_noise_scale (float): scale of the gaussian noise added to the
                bootstrapped critic values.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            calculate_priority (bool): whether to calculate priority. This is
                only useful if priority replay is enabled.
            num_rollout_sampled_actions (int): number of sampled actions in rollout.
                The one with the highest Q_value + epistemic_std will be executed.
            num_bootstrapped_critics (int): a positive number of bootstrapped critics 
                for uncertainty estimation. Default is 1.
            critic_replicas_deepcopy (bool): whether to deepcopy the critic_network
                for replicas. Default is False, meaning that each critic_replica
                will have different independently instantiated parameters.
            bootstrap_mask_prob (float): the parameter of the Binomial distribution
                for independently masking out a transition to simulate bootstrapping.
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. ``env`` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            beta_ub (float): parameter for computing the upperbound of Q value:
                :math:`Q_ub(s,a) = \mu_Q(s,a) + \beta_ub * \sigma_Q(s,a)`    
            beta_lb
            rollout_random_action (float): the probability of taking a uniform
                random action during a ``rollout_step()``. 0 means always directly
                taking actions added with OU noises and 1 means always sample
                uniformly random actions. A bigger value results in more
                exploration during rollout.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between ``[-dqda_clipping, dqda_clipping]``.
                Does not perform clipping if ``dqda_clipping == 0``.
            action_l2 (float): weight of squared action l2-norm on actor loss.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        assert num_bootstrapped_critics >= 1, (
            "OaecAlgorithm requires a positive num_bootstrapped_critics.")

        self._calculate_priority = calculate_priority
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

        critic_network = critic_network_ctor(
            input_tensor_spec=(observation_spec, action_spec),
            output_tensor_spec=reward_spec)
        actor_network = actor_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        # There is one default critic, plus a number of bootstrapped critics,
        # each critic is paired with an auxiliary critic for optimization
        # variability correction.
        # The order of critic_networks: 
        # - the default critic
        # - bootstrapped critics
        # - auxiliary critic for the default critic
        # - auxiliary critics for bootstrapped critics
        self._num_bootstrapped_critics = num_bootstrapped_critics
        self._num_aux_critics = num_bootstrapped_critics + 1
        critic_networks = critic_network.make_parallel(
            self._num_aux_critics * 2, deepcopy=critic_replicas_deepcopy)

        self._action_l2 = action_l2
        self._reward_noise_scale = reward_noise_scale
        self._beta_ub = beta_ub
        self._beta_lb = beta_lb
        self._num_rollout_sampled_actions = num_rollout_sampled_actions
        self._bootstrap_mask_prob = bootstrap_mask_prob

        train_state_spec = OaecState(
            actor=OaecActorState(
                actor=actor_network.state_spec,
                critics=critic_networks.state_spec),
            critics=OaecCriticState(
                critics=critic_networks.state_spec,
                target_actor=actor_network.state_spec,
                target_critics=critic_networks.state_spec))

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            reward_weights=reward_weights,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])

        self._actor_network = actor_network
        self._num_critic_replicas = num_critic_replicas
        self._critic_networks = critic_networks

        self._target_actor_network = actor_network.copy(
            name='target_actor_networks')
        self._target_critic_networks = critic_networks.copy(
            name='target_critic_networks')

        self._rollout_random_action = float(rollout_random_action)

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        self._critic_losses = [None] * num_critic_replicas
        for i in range(num_critic_replicas):
            self._critic_losses[i] = critic_loss_ctor(
                name=("critic_loss" + str(i)))

        self._update_target = common.TargetUpdater(
            models=[self._actor_network, self._critic_networks],
            target_models=[
                self._target_actor_network, self._target_critic_networks
            ],
            tau=target_update_tau,
            period=target_update_period)

        self._dqda_clipping = dqda_clipping

    def _predict_action(self, 
                        observation, 
                        state: OaecActorState, 
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False):
        action_dist, actor_state = self._actor_network(observation, state=state.actor)
        critic_states = state.critics

        if explore:
            if self._training_started:
                ## use Q_value + epistemic_std to select action for exploration
                # [n_sampled, B, ...]
                actions = action_dist.sample(
                    sample_shape=(self._num_rollout_sampled_actions))  
                # [n_sampled * B, ...]
                critic_observations = observation.repeat(
                    [self._num_rollout_sampled_actions,] + [1] * observation_spec.ndim)
                # [n_sampled * B, ...]
                critic_actions = actions.reshape(
                    actions.shape[0] * actions.shape[1], *actions.shape[2:])
                # [n_sampled * B, n, ...]
                q_values, critic_states = self._critic_networks(
                    (critic_observations, critic_actions), state=state.critics)
                # compute epistemic std
                n_aux = self._num_aux_critics
                q_values_corrected = q_values[:, :n_aux, ...] - q_values[:, n_aux:, ...]
                q_values_corrected_bootstrap = q_values_corrected[:, 2:, ...] - q_values_corrected[:, 0, ...]
                # [n_sampled, B, n_aux, ...]
                q_values_corrected_bootstrap = q_values_corrected_bootstrap.reshape(
                    actions.shape[0], -1, *q_values_corrected_bootstrap.shape[1:])
                # [n_sampled, B, ...]
                q_epi_std = (q_values_corrected_bootstrap ** 2).mean(dim=2).sqrt()
                q_values_ub = q_values[:, 0, ...].reshape(
                    actions.shape[0], actions.shape[1], -1) + self._beta_ub * q_epi_std
                action_idx = q_values_ub.squeeze(-1)max(dim=0)[1]  # [B, ...]
                batch_idx = torch.arange(
                    action.shape[0]).type_as(action_idx)
                action = actions[action_idx, batch_idx, ...]
            else:
                # This uniform sampling during initial collect stage is
                # important since current explore_network is deterministic
                action = alf.nest.map_structure(
                    lambda spec: spec.sample(outer_dims=observation.shape[:1]),
                    self._action_spec)
        else:
            if eps_greedy_sampling:
                action = dist_utils.epsilon_greedy_sample(action_dist, epsilon_greedy)
            else:
                action = dist_utils.rsample_action_distribution(action_dist)

        return action, actor_state, critic_states

    def predict_step(self, inputs: TimeStep, state: OaecState):
        action, actor_state, _ = self._predict_action(
            inputs.observation,
            state=state.actor,
            epsilon_greedy=self._epsilon_greedy,
            eps_greedy_sampling=True)

        return AlgStep(
            output=action,
            state=OaecState(actor=actor_state),
            info=OaecInfo(action=action))

    def rollout_step(self, inputs: TimeStep, state=None):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by OaecAlgorithm")

        action, actor_state, critic_states = self._predict_action(
            inputs.observation, 
            state.actor, 
            explore=True)
        reward_noise = torch.randn(
            inputs.reward.shape + (self._num_bootstrapped_critics,)) * self._reward_noise_scale
        prob_t = torch.full((inputs.reward.shape[0], self._num_bootstrapped_critics), 
                            self._bootstrap_mask_prob)
        mask = torch.bernoulli(prob_t)

        new_state = OaecState(
            actor=actor_state,
            critics=critic_states)
        info = OaecRolloutInfo(action=action,
                               mask=mask,
                               reward_noise=reward_noise)
        return AlgStep(
            output=action, state=new_state, info=info)

    def _critic_train_step(self, inputs: TimeStep, state: OaecCriticState,
                           rollout_info: OaecRolloutInfo):
        target_action, target_actor_state = self._target_actor_network(
            inputs.observation, state=state.target_actor)
        target_q_values, target_critic_states = self._target_critic_networks(
            (inputs.observation, target_action), state=state.target_critics)
        q_values, critic_states = self._critic_networks(
            (inputs.observation, rollout_info.action), state=state.critics)

        state = OaecCriticState(
            critics=critic_states,
            target_actor=target_actor_state,
            target_critics=target_critic_states)

        info = OaecCriticInfo(
            q_values=q_values, target_q_values=target_q_values)

        return state, info

    def _actor_train_step(self, inputs: TimeStep, state: OaecActorState):
        action, actor_state = self._actor_network(
            inputs.observation, state=state.actor)

        q_values, critic_states = self._critic_networks(
            (inputs.observation, action), state=state.critics)
        if self.has_multidim_reward():
            # Multidimensional reward: [B, replicas, reward_dim]
            q_values = q_values * self.reward_weights
        # use the default critic
        q_value = q_values[:, 0]

        # This sum() will reduce all dims so q_value can be any rank
        dqda = nest_utils.grad(action, q_value.sum())

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            if self._action_l2 > 0:
                assert action.requires_grad
                loss += self._action_l2 * (action**2)
            loss = loss.sum(list(range(1, loss.ndim)))
            return loss

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        state = OaecActorState(actor=actor_state, critics=critic_states)
        info = LossInfo(loss=sum(nest.flatten(actor_loss)), extra=actor_loss)
        return AlgStep(output=action, state=state, info=info)

    def train_step(self, inputs: TimeStep, state: OaecState,
                   rollout_info: OaecRolloutInfo):
        critic_states, critic_info = self._critic_train_step(
            inputs=inputs, state=state.critics, rollout_info=rollout_info)
        policy_step = self._actor_train_step(inputs=inputs, state=state.actor)
        return policy_step._replace(
            state=OaecState(actor=policy_step.state, critics=critic_states),
            info=OaecInfo(
                reward=inputs.reward,
                reward_noise=rollout_info.reward_noise,
                step_type=inputs.step_type,
                discount=inputs.discount,
                action_distribution=policy_step.output,
                critic=critic_info,
                actor_loss=policy_step.info))

    def calc_loss(self, info: OaecInfo):
        n_aux = self._num_aux_critics
        # compute total std of the target critic for estimation of overestimation
        target_values = info.critic.target_q_values[:, :, 1:n_aux, ...] - info.critic.target_q_values[:, :, 0, ...]
        target_std = (target_values ** 2).mean(dim=2).sqrt()  # [T, B, ...]

        critic_losses = [None] * (n_aux * 2)
        # original critic
        critic_losses[0] = self._critic_losses[0](
            info=info,
            value=info.critic.q_values[:, :, 0, ...],
            target_value=info.critic.target_q_values[:, :, 0, ...] - self._beta_lb * target_std).loss
        # original critic's aux critic
        critic_losses[n_aux] = self._critic_losses[n_aux](
            info=info._replace(
                # reward=torch.zeros_like(info.reward)),
                reward=torch.randn_like(info.reward) * self._reward_noise_scale),
            value=info.critic.q_values[:, :, n_aux, ...],
            target_value=info.critic.target_q_values[:, :, n_aux, ...]).loss
        # bootstrapped critics and their aux critics
        for i in range(1, n_aux):
            mask = info.mask[:, :, i - 1] / self._bootstrap_mask_prob
            critic_losses[i] = mask * self._critic_losses[i](
                info=info._replace(reward=info.reward + info.reward_noise[:, :, :, i - 1]),
                value=info.critic.q_values[:, :, i, ...],
                target_value=info.critic.target_q_values[:, :, i, ...] - self._beta_lb * target_std).loss

            critic_losses[i + n_aux] = mask * info.mask[:, :, i - 1] * self._critic_losses[i + n_aux](
                info=info._replace(
                    # reward=torch.zeros_like(info.reward)),
                    reward=torch.randn_like(info.reward) * self._reward_noise_scale),
                value=info.critic.q_values[:, :, i + n_aux, ...],
                target_value=info.critic.target_q_values[:, :, i + n_aux, ...] - self._beta_lb * target_std).loss

        critic_loss = math_ops.add_n(critic_losses)

        if self._calculate_priority:
            valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = (
                (critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        actor_loss = info.actor_loss

        return LossInfo(
            loss=critic_loss + actor_loss.loss,
            priority=priority,
            extra=OaecLossInfo(critic=critic_loss, actor=actor_loss.extra))

    def after_update(self, root_inputs, info: DdpgInfo):
        self._update_target()

    def _trainable_attributes_to_ignore(self):
        return ['_target_actor_network', '_target_critic_networks']
