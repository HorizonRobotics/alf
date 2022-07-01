# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Thompson Sampling Actor and Bayesian Critic Algorithm."""

from absl import logging
import numpy as np
import functools
import torch
import torch.distributions as td
import torch.nn as nn
from typing import Callable

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.algorithms.multiswag_algorithm import MultiSwagAlgorithm
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.optimizers import Adam, AdamTF
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorNetwork, ActorDistributionNetwork
from alf.networks.param_networks import CriticDistributionParamNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops, summary_utils

TsabcActionState = namedtuple(
    "TsabcActionState", ["actor_network", "explore_network"], default_value=())

TsabcCriticState = namedtuple("TsabcCriticState",
                              ["critics", "target_critics"])

TsabcState = namedtuple(
    "TsabcState", ["action", "actor", "explore", "critic"], default_value=())

TsabcCriticInfo = namedtuple(
    "TsabcCriticInfo",
    ["observation", "rollout_action", "critic_state", "target_critic"])

TsabcActorInfo = namedtuple(
    "TsabcActorInfo", ["actor_loss", "neg_entropy"], default_value=())

TsabcExploreInfo = namedtuple(
    "TsabcExploreInfo", ["explore_loss", "neg_entropy"], default_value=())

TsabcInfo = namedtuple(
    "TsabcInfo", [
        "reward", "step_type", "discount", "action", "action_distribution",
        "explore_action_distribution", "actor", "explore", "critic", "alpha",
        "explore_alpha", "log_pi", "log_pi_explore"
    ],
    default_value=())

TsabcLossInfo = namedtuple(
    'TsabcLossInfo', ['actor', 'explore', 'critic', 'alpha', 'explore_alpha'],
    default_value=())


@alf.configurable
class TsabcAlgorithm(OffPolicyAlgorithm):
    r"""Soft Actor and Bayesian Critic Algorithm. """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 explore_network_cls=ActorDistributionNetwork,
                 critic_module_cls=FuncParVIAlgorithm,
                 num_critic_replicas=10,
                 deterministic_critic=False,
                 reward_weights=None,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 beta_ub=1.,
                 beta_lb=1.,
                 entropy_regularization_weight=1.,
                 entropy_regularization=None,
                 target_entropy=None,
                 initial_log_alpha=0.0,
                 max_log_alpha=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 explore_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 explore_alpha_optimizer=None,
                 debug_summaries=False,
                 name="TsabcAlgorithm"):
        """
        Args:
            explore_network_cls
            critic_module_cls
            deterministic_critic
            beta_ub (float): parameter for computing the upperbound of Q value:
                :math:`Q_ub(s,a) = \mu_Q(s,a) + \beta_ub * \sigma_Q(s,a)`    
            beta_lb
            explore_optimizer
            explore_alpha_optimizer
        """
        assert action_spec.is_continuous, "Only continuous action is supported"

        actor_network, explore_network, critic_module, target_critic_network = \
            self._make_modules(observation_spec, action_spec, reward_spec,
                               actor_network_cls, explore_network_cls,
                               critic_module_cls, critic_optimizer,
                               num_critic_replicas, deterministic_critic)

        explore_networks = explore_network.make_parallel(num_critic_replicas)
        # explore_networks = nn.ModuleList()
        # for i in range(num_critic_replicas):
        #     explore_networks.append(explore_network_cls(
        #         input_tensor_spec=observation_spec,
        #         action_spec=action_spec))

        # explore_state_spec = alf.nest.map_structure(
        #     lambda spec: alf.TensorSpec((num_critic_replicas, ) + spec.shape,
        #                                  spec.dtype),
        #     explore_network.state_spec)

        target_critic_params = critic_module.particles.detach().clone()

        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value(
                'TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy

        def _init_log_alpha():
            return nn.Parameter(torch.tensor(float(initial_log_alpha)))

        log_alpha = _init_log_alpha()
        # log_explore_alpha = _init_log_alpha()

        action_state_spec = TsabcActionState(
            actor_network=actor_network.state_spec,
            explore_network=explore_networks.state_spec)
        # explore_network=explore_state_spec)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=TsabcState(
                action=action_state_spec,
                actor=target_critic_network.state_spec,
                explore=target_critic_network.state_spec,
                critic=TsabcCriticState(
                    critics=target_critic_network.state_spec,
                    target_critics=target_critic_network.state_spec)),
            predict_state_spec=TsabcState(action=action_state_spec),
            reward_weights=reward_weights,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if explore_optimizer is not None:
            self.add_optimizer(explore_optimizer, [explore_networks])
        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, nest.flatten(log_alpha))
        # if explore_alpha_optimizer is not None:
        #     self.add_optimizer(explore_alpha_optimizer,
        #                        nest.flatten(log_explore_alpha))

        self._log_alpha = log_alpha
        # self._log_explore_alpha = log_explore_alpha
        if max_log_alpha is not None:
            self._max_log_alpha = torch.tensor(float(max_log_alpha))
            # self._max_log_explore_alpha = torch.tensor(float(max_log_alpha))
        else:
            self._max_log_alpha = None
            # self._max_log_explore_alpha = None

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        self._critic_loss = critic_loss_ctor(name="critic_loss")

        self._target_entropy = _set_target_entropy(
            self.name, target_entropy, nest.flatten(self._action_spec))

        self._mini_batch_size = alf.get_config_value(
            'TrainerConfig.mini_batch_size')
        self._mini_batch_length = alf.get_config_value(
            'TrainerConfig.mini_batch_length')
        replay_buffer_length = alf.get_config_value(
            'TrainerConfig.replay_buffer_length')
        self._unroll_length = alf.get_config_value(
            'TrainerConfig.unroll_length')
        self._min_entropy_regularization = self._mini_batch_size / replay_buffer_length

        self._idx = 0
        self._cyclic_unroll_steps = 0
        self._init_rollout_steps = 0
        self._beta_ub = beta_ub
        self._beta_lb = beta_lb
        self._entropy_regularization_weight = entropy_regularization_weight
        self._entropy_regularization = entropy_regularization

        self._use_entropy_reward = use_entropy_reward
        self._dqda_clipping = dqda_clipping

        self._actor_network = actor_network
        self._explore_networks = explore_networks

        self._critic_module = critic_module
        self._num_critic_replicas = num_critic_replicas
        self._deterministic_critic = deterministic_critic
        self._target_critic_params = torch.nn.Parameter(target_critic_params)
        target_critic_network.set_parameters(self._target_critic_params)
        self._target_critic_network = target_critic_network

        self._update_target_critic_params = common.TargetUpdater(
            models=self._critic_module.particles,
            target_models=self._target_critic_params,
            tau=target_update_tau,
            period=target_update_period)

    def _make_modules(self, observation_spec, action_spec, reward_spec,
                      actor_network_cls, explore_network_cls,
                      critic_module_cls, critic_optimizer, num_critic_replicas,
                      deterministic_critic):

        assert actor_network_cls is not None, (
            "ActorNetwork must be provided!")
        actor_network = actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        assert explore_network_cls is not None, (
            "ExploreNetwork must be provided!")
        explore_network = explore_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        input_tensor_spec = (observation_spec, action_spec)
        critic_network = CriticDistributionParamNetwork(
            input_tensor_spec=input_tensor_spec,
            output_tensor_spec=reward_spec,
            deterministic=deterministic_critic)
        target_critic_network = critic_network.copy(
            name='TargetCriticDistributionParamNetwork')

        if critic_optimizer is None:
            critic_optimizer = AdamTF(lr=3e-4)
        critic_module = critic_module_cls(
            input_tensor_spec=input_tensor_spec,
            param_net=critic_network,
            num_particles=num_critic_replicas,
            optimizer=critic_optimizer)

        return actor_network, explore_network, critic_module, target_critic_network

    def _predict_action(self,
                        observation,
                        state: TsabcActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False,
                        train=False):

        new_state = TsabcActionState()
        if explore:
            # action_dist, explore_network_state = self._explore_network(
            #     observation, state=state.explore_network)
            # action, explore_network_state = self._explore_network(
            #     observation, state=state.explore_network)
            action_dist = ()
            action, explore_network_state = self._explore_networks(
                observation, state=state.explore_network)
            new_state = new_state._replace(
                explore_network=explore_network_state)
            # if train:
            # output_states = []
            # for i in range(self._num_critic_replicas):
            #     s = alf.nest.map_structure(
            #         lambda x: x[:, i, ...], state.explore_network)
            #     ret = self._explore_networks[i](observation, state=s)
            #     ret = alf.nest.map_structure(lambda x: x.unsqueeze(1), ret)
            #     output_states.append(ret)
            # action, explore_network_state = alf.nest.map_structure(
            #     lambda *tensors: torch.cat(tensors, dim=1), *output_states)
            # new_state = new_state._replace(explore_network=explore_network_state)
            # else:
            if not train:
                if self._cyclic_unroll_steps == 0:
                    self._idx = torch.randint(self._num_critic_replicas, ())
                action = action[:, self._idx, :]
                # import pdb; pdb.set_trace()

                # action, explore_network_state = self._explore_networks[self._idx](
                #     observation, state=state.explore_network)

                self._cyclic_unroll_steps += 1
                if self._cyclic_unroll_steps >= 100:
                    self._cyclic_unroll_steps = 0
        else:
            action_dist, actor_network_state = self._actor_network(
                observation, state=state.actor_network)
            new_state = new_state._replace(actor_network=actor_network_state)

            if eps_greedy_sampling:
                action = dist_utils.epsilon_greedy_sample(
                    action_dist, epsilon_greedy)
            else:
                action = dist_utils.rsample_action_distribution(action_dist)

        return action_dist, action, new_state

    def predict_step(self, inputs: TimeStep, state: TsabcState):
        action_dist, action, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=self._epsilon_greedy,
            eps_greedy_sampling=True)
        return AlgStep(
            output=action,
            state=TsabcState(action=action_state),
            info=TsabcInfo(action_distribution=action_dist))

    ## for remote_eval
    def get_predict_module_state(self):
        """get state_dict of the predict_step module. """
        return self._actor_network.state_dict()

    ## for remote_eval
    def load_predict_module_state(self, state_dict):
        """load state_dict to the predict_step module. """
        self._actor_network.load_state_dict(state_dict)

    def rollout_step(self, inputs: TimeStep, state: TsabcState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        if self._init_rollout_steps <= 10000:
            explore = False
        else:
            explore = True

        action_dist, action, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=1.0,
            eps_greedy_sampling=True,
            explore=explore)

        self._init_rollout_steps += 1

        if self.need_full_rollout_state():
            critics_state = self._critic_module.predict_step(
                intputs=(inputs.observation, action),
                state=state.critic.critics).state
            _, target_critics_state = self._target_critic_network(
                (inputs.observation, action), state.critic.target_critics)
            critic_state = TsabcCriticState(
                critics=critics_state, target_critics=target_critics_state)
            actor_state = critics_state
            explore_state = critics_state
        else:
            actor_state = state.actor
            explore_state = state.explore
            critic_state = state.critic

        new_state = TsabcState(
            action=action_state,
            actor=actor_state,
            explore=explore_state,
            critic=critic_state)
        return AlgStep(
            output=action, state=new_state, info=TsabcInfo(
                action=action))  #, explore_action_distribution=action_dist))

    def _actor_train_step(self, inputs: TimeStep, state, action, log_pi):
        critic_step = self._critic_module.predict_step(
            inputs=(inputs.observation, action), state=state)
        critics_state = critic_step.state
        if self._deterministic_critic:
            critics = critic_step.output
        else:
            critics_dist = critic_step.output
            critics = critics_dist.mean

        q_mean = critics.mean(1)
        q_std = critics.std(1)
        # q_value = q_mean
        q_value = q_mean - self._beta_lb * q_std

        with alf.summary.scope(self._name):
            summary_utils.add_mean_hist_summary("critics_batch_mean", q_mean)
            summary_utils.add_mean_hist_summary("critics_std", q_std)

        dqda = nest_utils.grad(action, q_value.sum())

        cont_alpha = torch.exp(self._log_alpha).detach()
        entropy_loss = cont_alpha * log_pi
        neg_entropy = sum(nest.flatten(log_pi))

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))
        actor_info = LossInfo(
            loss=actor_loss + entropy_loss,
            extra=TsabcActorInfo(
                actor_loss=actor_loss, neg_entropy=neg_entropy))
        return critics_state, actor_info

    def _explore_train_step(self, inputs: TimeStep, state, action,
                            log_pi_explore):
        critic_step = self._critic_module.predict_step(
            inputs=(inputs.observation, action), state=state)
        critics_state = critic_step.state
        if self._deterministic_critic:
            critics = critic_step.output
        else:
            critics_dist = critic_step.output
            critics = critics_dist.mean

        # q_mean = critics.mean(1)
        # q_std = critics.std(1)
        # q_value = q_mean + self._beta_ub * q_std
        q_value = critics

        # with alf.summary.scope(self._name):
        #     q_batch_mean = critics.mean(1)
        #     summary_utils.add_mean_hist_summary(
        #         "critics_batch_mean", q_batch_mean)
        # summary_utils.add_mean_hist_summary(
        #     "critics_std", q_std)

        dqda = nest_utils.grad(action, q_value.sum())

        # cont_explore_alpha = torch.exp(self._log_explore_alpha).detach()
        # entropy_loss = cont_explore_alpha * log_pi_explore
        # neg_entropy = sum(nest.flatten(log_pi_explore))

        def explore_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            # loss = dqda.detach() * action
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        explore_loss = nest.map_structure(explore_loss_fn, dqda, action)
        explore_loss = math_ops.add_n(nest.flatten(explore_loss))
        explore_info = LossInfo(
            # loss=explore_loss + entropy_loss,
            loss=explore_loss,
            # extra=TsabcExploreInfo(explore_loss=explore_loss, neg_entropy=neg_entropy))
            extra=TsabcExploreInfo(explore_loss=explore_loss))
        return critics_state, explore_info

    def _critic_train_step(self, inputs: TimeStep, state: TsabcCriticState,
                           rollout_info: TsabcInfo, action):
        target_critics_dist, target_critics_state = self._target_critic_network(
            (inputs.observation, action), state.target_critics)

        if self._deterministic_critic:
            target_critics = target_critics_dist.squeeze(-1)
        else:
            target_critics = target_critics_dist.mean

        # target_critics_mean = target_critics.mean(1)
        # target_critics_std = target_critics.std(1)
        # targets = target_critics_mean - self._beta_lb * target_critics_std
        targets = target_critics

        targets = targets.detach()

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                target_critics_mean = target_critics.mean(1)
                summary_utils.add_mean_hist_summary(
                    "target_critics_batch_mean", target_critics_mean)
                target_critics_std = target_critics.std(1)
                summary_utils.add_mean_hist_summary("target_critics_std",
                                                    target_critics_std)

        critic_info = TsabcCriticInfo(
            observation=inputs.observation,
            rollout_action=rollout_info.action,
            critic_state=state.critics,
            target_critic=targets)

        state = TsabcCriticState(
            critics=(), target_critics=target_critics_state)

        return state, critic_info

    def _alpha_train_step(self, log_pi):
        alpha_loss = nest.map_structure(
            lambda la, lp, t: la * (-lp - t).detach(), self._log_alpha, log_pi,
            self._target_entropy)
        return sum(nest.flatten(alpha_loss))

    # def _explore_alpha_train_step(self, log_pi_explore):
    #     explore_alpha_loss = nest.map_structure(
    #         lambda la, lp, t: la * (-lp - t).detach(),
    #         self._log_explore_alpha, log_pi_explore,
    #         self._target_entropy)
    #     return sum(nest.flatten(explore_alpha_loss))

    def train_step(self, inputs: TimeStep, state: TsabcState,
                   rollout_info: TsabcInfo):

        # train actor_network
        (action_dist, action, action_state) = self._predict_action(
            inputs.observation, state=state.action)
        log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                    action_dist, action)
        log_pi = sum(nest.flatten(log_pi))
        actor_state, actor_loss = self._actor_train_step(
            inputs, state.actor, action, log_pi)

        # train explore_network
        (explore_action_dist, explore_action, explore_action_state) = \
            self._predict_action(
                inputs.observation, state=state.action, explore=True, train=True)
        action_state = action_state._replace(
            explore_network=explore_action_state.explore_network)
        # log_pi_explore = nest.map_structure(lambda dist, a: dist.log_prob(a),
        #                                     explore_action_dist, explore_action)
        # log_pi_explore = sum(nest.flatten(log_pi_explore))
        log_pi_explore = ()

        explore_state, explore_loss = self._explore_train_step(
            inputs, state.explore, explore_action, log_pi_explore)

        # train critic_module
        critic_state, critic_info = self._critic_train_step(
            inputs, state.critic, rollout_info, action)

        # train alpha and explore_alpha
        alpha_loss = self._alpha_train_step(log_pi)
        # explore_alpha_loss = self._explore_alpha_train_step(log_pi_explore)

        state = TsabcState(
            action=action_state,
            actor=actor_state,
            explore=explore_state,
            critic=critic_state)
        info = TsabcInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            action_distribution=action_dist,
            explore_action_distribution=explore_action_dist,
            alpha=alpha_loss,
            # explore_alpha=explore_alpha_loss,
            actor=actor_loss,
            explore=explore_loss,
            critic=critic_info,
            log_pi=log_pi,
            log_pi_explore=log_pi_explore)
        return AlgStep(action, state, info)

    def after_update(self, root_inputs, info: TsabcInfo):
        self._update_target_critic_params()
        self._target_critic_network.set_parameters(self._target_critic_params)
        if self._max_log_alpha is not None:
            nest.map_structure(
                lambda la: la.data.copy_(torch.min(la, self._max_log_alpha)),
                self._log_alpha)

        # if self._max_log_explore_alpha is not None:
        #     nest.map_structure(
        #         lambda la: la.data.copy_(torch.min(la, self._max_log_explore_alpha)),
        #         self._log_explore_alpha)

    def calc_loss(self, info: TsabcInfo):
        alpha_loss = info.alpha
        # explore_alpha_loss = info.explore_alpha
        actor_loss = info.actor
        explore_loss = info.explore

        # if self._use_entropy_reward:
        #     with torch.no_grad():
        #         entropy_reward = nest.map_structure(
        #             lambda la, lp: -torch.exp(la) * lp, self._log_explore_alpha,
        #             info.log_pi_explore)
        #         entropy_reward = sum(nest.flatten(entropy_reward))
        #         gamma = self._critic_loss.gamma
        #         info = info._replace(
        #             reward=info.reward + entropy_reward * gamma)

        exp_size = self._unroll_length * alf.summary.get_global_counter()
        entropy_regularization = self._entropy_regularization
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization_weight * max(
                self._mini_batch_size / exp_size,
                self._min_entropy_regularization)

        critic_step = self._critic_module.train_step(
            inputs=None,
            entropy_regularization=entropy_regularization,
            loss_func=functools.partial(self._neglogprob, info))
        critic_loss, _ = self._critic_module.update_with_gradient(
            critic_step.info)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha", self._log_alpha.exp())
                # alf.summary.scalar("explore_alpha", self._log_explore_alpha.exp())
                summary_utils.add_mean_hist_summary("critics_losses",
                                                    critic_loss.extra)

        return LossInfo(
            # loss=math_ops.add_ignore_empty(actor_loss.loss, alpha_loss),
            loss=math_ops.add_ignore_empty(actor_loss.loss + explore_loss.loss,
                                           alpha_loss),
            # loss=math_ops.add_ignore_empty(actor_loss.loss + explore_loss.loss,
            #                                alpha_loss + explore_alpha_loss),
            extra=TsabcLossInfo(
                actor=actor_loss.extra,
                explore=explore_loss.extra,
                # explore_alpha=explore_alpha_loss,
                alpha=alpha_loss))

    def _neglogprob(self, info: TsabcInfo, params):
        """Function computing the negative log_prob for critics.

        This function will be used as the loss_func for functional_particle_vi
        of the critics_module.

        Args:
            info (TsabcInfo): 
            params (Tensor): particles of the critics_module, representing 
                params of the CriticDistributionParamNetwork.

        Returns:
            neg_logprob (Tensor): of shape [B, n] or [B] (n=1)
        """
        self._critic_module.reset_param_net(params)
        num_particles = params.shape[0]
        critic_info = info.critic
        targets = critic_info.target_critic

        observation = critic_info.observation.reshape(
            -1, self._observation_spec.shape[0])
        action = critic_info.rollout_action.reshape(-1,
                                                    self._action_spec.shape[0])

        critic_step = self._critic_module.predict_step(
            inputs=(observation, action), state=critic_info.critic_state)
        critics_dist = critic_step.output

        neg_logprob = []
        if self._deterministic_critic:
            critics = critics_dist.squeeze(-1)
            critics_state = critic_step.state
            critics = critics.reshape(self._mini_batch_length, -1,
                                      *critics.shape[1:])
            for i in range(num_particles):
                neg_logprob.append(
                    self._critic_loss(
                        info=info,
                        value=critics[:, :, i, ...],
                        # target_value=targets).loss)
                        target_value=targets[:, :, i, ...]).loss)
        else:
            # td_targets = self._critic_loss.compute_td_target(info, targets)
            for i in range(num_particles):
                td_targets = self._critic_loss.compute_td_target(
                    info, targets[:, :, i, ...])
                critics_mean = critics_dist.base_dist.mean.reshape(
                    self._mini_batch_length, -1,
                    *critics_dist.base_dist.mean.shape[1:])
                critics_std = critics_dist.base_dist.stddev.reshape(
                    self._mini_batch_length, -1,
                    *critics_dist.base_dist.stddev.shape[1:])
                value_dist = td.Normal(
                    loc=critics_mean[:-1, :, i, ...],
                    scale=critics_std[:-1, :, i, ...])
                neg_logprob.append(-value_dist.log_prob(td_targets))
        neg_logprob = torch.stack(neg_logprob).reshape(num_particles, -1)
        return neg_logprob.mean(-1)

    def _trainable_attributes_to_ignore(self):
        return ['_critic_module', '_target_critic_params']
