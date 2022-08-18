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
"""Optimistic Actor and Bayesian Critic Algorithm."""

import numpy as np
import functools
import torch
import torch.distributions as td
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
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

OabcActionState = namedtuple("OabcActionState",
                             ["actor_network", "explore_network", "critic"],
                             default_value=())

OabcCriticState = namedtuple("OabcCriticState", ["critics", "target_critics"])

OabcState = namedtuple("OabcState", ["action", "actor", "explore", "critic"],
                       default_value=())

OabcCriticInfo = namedtuple(
    "OabcCriticInfo",
    ["observation", "rollout_action", "critic_state", "target_critic"])

OabcActorInfo = namedtuple("OabcActorInfo", ["actor_loss", "neg_entropy"],
                           default_value=())

OabcExploreInfo = namedtuple("OabcExploreInfo",
                             ["explore_loss", "neg_entropy"],
                             default_value=())

OabcInfo = namedtuple("OabcInfo", [
    "reward", "step_type", "discount", "action", "action_distribution",
    "explore_action_distribution", "actor", "explore", "critic", "alpha",
    "explore_alpha", "log_pi", "log_pi_explore"
],
                      default_value=())

OabcLossInfo = namedtuple(
    'OabcLossInfo', ['actor', 'explore', 'critic', 'alpha', 'explore_alpha'],
    default_value=())

def get_target_updater(param, target_param, tau=1.0, period=1, copy=True):
    r"""Performs a soft update of the target parameter.
    For param :math:`w_s` and its corresponding target_param :math:`w_t`, 
    a soft update is:
    .. math::
        w_t = (1 - \tau) * w_t + \tau * w_s.
    Args:
        params (Tensor | Parameter): the current tensor or parameter.
        target_models (Parameter): the parameter to be updated.
        tau (float): A float scalar in :math:`[0, 1]`. Default :math:`\tau=1.0`
            means hard update.
        period (int): Step interval at which the target param is updated.
        copy (bool): If True, also copy ``param`` to ``target_param`` in the
            beginning.
    Returns:
        Callable: a callable that performs a soft update of the target parameter.
    """
    def _copy_parameter(s, t):
        t.data.copy_(s)

    def _lerp_parameter(s, t):
        t.data.lerp_(s, tau)

    if copy:
        _copy_parameter(param, target_param)

    def update():
        if tau != 1.0:
            _lerp_parameter(param, target_param)
        else:
            _copy_parameter(param, target_param)

    return common.Periodically(update, period, 'periodic_update_targets')

@alf.configurable
class OabcAlgorithm(OffPolicyAlgorithm):
    r"""Soft Actor and Bayesian Critic Algorithm. """
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 explore_network_cls=ActorDistributionNetwork,
                 critic_module_cls=FuncParVIAlgorithm,
                 deterministic_actor=True,
                 deterministic_critic=False,
                 reward_weights=None,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 beta_ub=1.,
                 beta_lb=1.,
                 common_td_target: bool = True,
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
                 name="OabcAlgorithm"):
        """
        Args:
            explore_network_cls
            critic_module_cls
            deterministic_actor
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
                               deterministic_critic)

        target_critic_params = critic_module.particles.detach().clone()

        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value(
                'TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy

        def _init_log_alpha():
            if alpha_optimizer is None:
                return torch.tensor(float(initial_log_alpha))
            else:
                return nn.Parameter(torch.tensor(float(initial_log_alpha)))

        log_alpha = _init_log_alpha()
        # log_explore_alpha = _init_log_alpha()

        action_state_spec = OabcActionState(
            actor_network=actor_network.state_spec,
            explore_network=explore_network.state_spec)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=OabcState(
                action=action_state_spec,
                actor=target_critic_network.state_spec,
                explore=target_critic_network.state_spec,
                critic=OabcCriticState(
                    critics=target_critic_network.state_spec,
                    target_critics=target_critic_network.state_spec)),
            predict_state_spec=OabcState(action=action_state_spec),
            reward_weights=reward_weights,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if explore_optimizer is not None:
            self.add_optimizer(explore_optimizer, [explore_network])
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
        critic_loss_ctor = functools.partial(critic_loss_ctor,
                                             debug_summaries=debug_summaries)
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

        self._training_started = False
        self._beta_ub = beta_ub
        self._beta_lb = beta_lb
        self._common_td_target = common_td_target
        self._entropy_regularization_weight = entropy_regularization_weight
        self._entropy_regularization = entropy_regularization

        self._use_entropy_reward = use_entropy_reward
        self._dqda_clipping = dqda_clipping

        self._actor_network = actor_network
        self._explore_network = explore_network

        self._critic_module = critic_module
        self._deterministic_actor = deterministic_actor
        self._deterministic_critic = deterministic_critic
        self._target_critic_params = torch.nn.Parameter(target_critic_params)
        target_critic_network.set_parameters(self._target_critic_params)
        self._target_critic_network = target_critic_network

        self._update_target_critic_params = get_target_updater(
            param=self._critic_module.sample_particles(),
            target_param=self._target_critic_params,
            tau=target_update_tau,
            period=target_update_period)

    def _make_modules(self, observation_spec, action_spec, reward_spec,
                      actor_network_cls, explore_network_cls,
                      critic_module_cls, critic_optimizer,
                      deterministic_critic):

        assert actor_network_cls is not None, (
            "ActorNetwork must be provided!")
        actor_network = actor_network_cls(input_tensor_spec=observation_spec,
                                          action_spec=action_spec)

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
        critic_module = critic_module_cls(input_tensor_spec=input_tensor_spec,
                                          param_net=critic_network,
                                          optimizer=critic_optimizer)

        return actor_network, explore_network, critic_module, target_critic_network

    def _predict_action(self,
                        observation,
                        state: OabcActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False,
                        train=False):
        del train
        new_state = OabcActionState()
        if explore:
            # action_dist, explore_network_state = self._explore_network(
            #     observation, state=state.explore_network)
            action, explore_network_state = self._explore_network(
                observation, state=state.explore_network)
            action_dist = ()

            new_state = new_state._replace(
                explore_network=explore_network_state)
        else:
            if self._deterministic_actor:
                action_dist = ()
                action, actor_network_state = self._actor_network(
                    observation, state=state.actor_network)
            else:
                action_dist, actor_network_state = self._actor_network(
                    observation, state=state.actor_network)

                if eps_greedy_sampling:
                    action = dist_utils.epsilon_greedy_sample(
                        action_dist, epsilon_greedy)
                else:
                    action = dist_utils.rsample_action_distribution(
                        action_dist)
            new_state = new_state._replace(actor_network=actor_network_state)

        return action_dist, action, new_state

    def predict_step(self, inputs: TimeStep, state: OabcState):
        action_dist, action, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=self._epsilon_greedy,
            eps_greedy_sampling=True)
        return AlgStep(output=action,
                       state=OabcState(action=action_state),
                       info=OabcInfo(action_distribution=action_dist))

    def rollout_step(self, inputs: TimeStep, state: OabcState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        action_dist, action, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=1.0,
            eps_greedy_sampling=True,
            explore=self._training_started)

        if self.need_full_rollout_state():
            critics_state = self._critic_module.predict_step(
                intputs=(inputs.observation, action),
                state=state.critic.critics).state
            _, target_critics_state = self._target_critic_network(
                (inputs.observation, action), state.critic.target_critics)
            critic_state = OabcCriticState(critics=critics_state,
                                           target_critics=target_critics_state)
            actor_state = critics_state
            explore_state = critics_state
        else:
            actor_state = state.actor
            explore_state = state.explore
            critic_state = state.critic

        new_state = OabcState(action=action_state,
                              actor=actor_state,
                              explore=explore_state,
                              critic=critic_state)
        return AlgStep(
            output=action, state=new_state, info=OabcInfo(
                action=action))  #, explore_action_distribution=action_dist))

    def _get_actor_train_q_value(self, critics, explore):
        q_mean = critics.mean(1)
        q_std = critics.std(1)
        if explore:
            q_value = q_mean + self._beta_ub * q_std
        else:
            # q_value = q_mean
            q_value = q_mean - self._beta_lb * q_std

        prefix = "explore_" if explore else ""
        with alf.summary.scope(self._name):
            summary_utils.add_mean_hist_summary(prefix + "critics_batch_mean",
                                                q_mean)
            summary_utils.add_mean_hist_summary(prefix + "critics_std", q_std)

        return q_value

    def _actor_train_step(self,
                          inputs: TimeStep,
                          state,
                          action,
                          log_pi,
                          explore=False):
        critic_step = self._critic_module.predict_step(
            inputs=(inputs.observation, action), state=state)
        critics_state = critic_step.state
        if self._deterministic_critic:
            critics = critic_step.output
        else:
            critics_dist = critic_step.output
            critics = critics_dist.mean

        q_value = self._get_actor_train_q_value(critics, explore)

        dqda = nest_utils.grad(action, q_value.sum())

        if self._deterministic_actor or explore:
            neg_entropy = ()
            entropy_loss = 0.
        else:
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

        if explore:
            extra = OabcExploreInfo(explore_loss=actor_loss)
        else:
            extra = OabcActorInfo(actor_loss=actor_loss,
                                  neg_entropy=neg_entropy)

        actor_info = LossInfo(loss=actor_loss + entropy_loss, extra=extra)
        return critics_state, actor_info

    def _critic_train_step(self, inputs: TimeStep, state: OabcCriticState,
                           rollout_info: OabcInfo, action):
        target_critics_dist, target_critics_state = self._target_critic_network(
            (inputs.observation, action), state.target_critics)

        if self._deterministic_critic:
            target_critics = target_critics_dist.squeeze(-1)
        else:
            target_critics = target_critics_dist.mean

        # if use common td_target for all critic
        if self._common_td_target:
            target_critics_mean = target_critics.mean(1)
            target_critics_std = target_critics.std(1)
            targets = target_critics_mean - self._beta_lb * target_critics_std
        else:
            # if use separate td_target for each critic
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

        critic_info = OabcCriticInfo(observation=inputs.observation,
                                     rollout_action=rollout_info.action,
                                     critic_state=state.critics,
                                     target_critic=targets)

        state = OabcCriticState(critics=(),
                                target_critics=target_critics_state)

        return state, critic_info

    def _alpha_train_step(self, log_pi):
        alpha_loss = nest.map_structure(
            lambda la, lp, t: la * (-lp - t).detach(), self._log_alpha, log_pi,
            self._target_entropy)
        return sum(nest.flatten(alpha_loss))

    def train_step(self, inputs: TimeStep, state: OabcState,
                   rollout_info: OabcInfo):
        self._training_started = True

        # train actor_network
        (action_dist, action,
         action_state) = self._predict_action(inputs.observation,
                                              state=state.action)
        if self._deterministic_actor:
            log_pi = ()
        else:
            log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                        action_dist, action)
            log_pi = sum(nest.flatten(log_pi))
        actor_state, actor_loss = self._actor_train_step(
            inputs, state.actor, action, log_pi)

        # train explore_network
        if self._explore_network is None:
            explore_state = state.actor
            explore_loss = LossInfo(
                loss=torch.zeros_like(actor_loss.loss),
                extra=OabcExploreInfo(
                    explore_loss=torch.zeros_like(actor_loss.loss)))
            log_pi_explore = ()
            explore_action_dist = ()
        else:
            (explore_action_dist, explore_action, explore_action_state) = \
                self._predict_action(
                    inputs.observation, state=state.action, explore=True, train=True)
            action_state = action_state._replace(
                explore_network=explore_action_state.explore_network)
            # log_pi_explore = nest.map_structure(lambda dist, a: dist.log_prob(a),
            #                                     explore_action_dist, explore_action)
            # log_pi_explore = sum(nest.flatten(log_pi_explore))
            log_pi_explore = ()

            explore_state, explore_loss = self._actor_train_step(
                inputs,
                state.explore,
                explore_action,
                log_pi_explore,
                explore=True)

        # train critic_module
        critic_state, critic_info = self._critic_train_step(
            inputs, state.critic, rollout_info, action)

        # train alpha and explore_alpha
        if self._deterministic_actor:
            alpha_loss = ()
        else:
            alpha_loss = self._alpha_train_step(log_pi)
        # explore_alpha_loss = self._alpha_train_step(log_pi_explore)

        state = OabcState(action=action_state,
                          actor=actor_state,
                          explore=explore_state,
                          critic=critic_state)
        info = OabcInfo(
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

    def after_update(self, root_inputs, info: OabcInfo):
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

    def calc_loss(self, info: OabcInfo):
        alpha_loss = info.alpha
        # explore_alpha_loss = info.explore_alpha
        actor_loss = info.actor
        explore_loss = info.explore

        assert not self._use_entropy_reward

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

        if self._deterministic_actor:
            loss = math_ops.add_ignore_empty(actor_loss.loss,
                                             explore_loss.loss)
            alpha_loss = ()
        else:
            loss = math_ops.add_ignore_empty(
                actor_loss.loss + explore_loss.loss, alpha_loss)
        return LossInfo(
            # loss=math_ops.add_ignore_empty(actor_loss.loss, alpha_loss),
            # loss=math_ops.add_ignore_empty(actor_loss.loss + explore_loss.loss,
            #                                alpha_loss + explore_alpha_loss),
            loss=loss,
            extra=OabcLossInfo(
                actor=actor_loss.extra,
                explore=explore_loss.extra,
                # explore_alpha=explore_alpha_loss,
                alpha=alpha_loss))

    def _neglogprob(self, info: OabcInfo, params):
        """Function computing the negative log_prob for critics.

        This function will be used as the loss_func for functional_particle_vi
        of the critics_module.

        Args:
            info (OabcInfo): 
            params (Tensor): particles of the critics_module, representing 
                params of the CriticDistributionParamNetwork.

        Returns:
            neg_logprob (Tensor): of shape [B, n] or [B] (n=1)
        """
        self._critic_module.reset_param_net(params)
        num_particles = params.shape[0]
        critic_info = info.critic
        targets = critic_info.target_critic

        observation = critic_info.observation[:-1, ...]
        action = critic_info.rollout_action[:-1, ...]
        observation = observation.reshape(
            -1, self._observation_spec.shape[0])
        action = action.reshape(-1, self._action_spec.shape[0])

        # observation = critic_info.observation.reshape(
        #     -1, self._observation_spec.shape[0])
        # action = critic_info.rollout_action.reshape(-1,
        #                                             self._action_spec.shape[0])

        critic_step = self._critic_module.predict_step(
            inputs=(observation, action),
            training=True,
            state=critic_info.critic_state)
        critics_dist = critic_step.output

        # speed up
        if self._deterministic_critic:
            critics = critics_dist.squeeze(-1)
            critics = critics.reshape(self._mini_batch_length, -1,
                                      *critics.shape[1:])
            if self._common_td_target:
                neg_logprob = [self._critic_loss(
                    info=info,
                    value=critics[:, :, i, ...],
                    target_value=targets).loss for i in range(num_particles)]
            else:
                neg_logprob = [self._critic_loss(
                    info=info,
                    value=critics[:, :, i, ...],
                    target_value=targets[:, :, i, ...]).loss \
                        for i in range(num_particles)]
            neg_logprob = torch.stack(neg_logprob).reshape(num_particles, -1)
        else:
            critics_mean = critics_dist.base_dist.mean.reshape(
                self._mini_batch_length-1, -1,
                *critics_dist.base_dist.mean.shape[1:])
            critics_std = critics_dist.base_dist.stddev.reshape(
                self._mini_batch_length-1, -1,
                *critics_dist.base_dist.stddev.shape[1:])
            if self._common_td_target:
                td_targets = self._critic_loss.compute_td_target(info, targets)
                td_targets = td_targets.unsqueeze(2)
            else:
                td_targets = [self._critic_loss.compute_td_target(
                    info, targets[:, :, i, ...]) for i in range(num_particles)]
                td_targets = torch.stack(td_targets, dim=2)
            
            value_dist = td.Normal(loc=critics_mean,
                                    scale=critics_std)
            neg_logprob = -value_dist.log_prob(td_targets)
            neg_logprob = neg_logprob.transpose(0, 2)
            neg_logprob = neg_logprob.reshape(num_particles, -1)


        # neg_logprob = []
        # if self._deterministic_critic:
        #     critics = critics_dist.squeeze(-1)
        #     critics = critics.reshape(self._mini_batch_length, -1,
        #                               *critics.shape[1:])
        #     for i in range(num_particles):
        #         if self._common_td_target:
        #             target_value = targets
        #         else:
        #             target_value = targets[:, :, i, ...]
        #         neg_logprob.append(
        #             self._critic_loss(info=info,
        #                               value=critics[:, :, i, ...],
        #                               target_value=target_value).loss)
        # else:
        #     if self._common_td_target:
        #         td_targets = self._critic_loss.compute_td_target(info, targets)
        #     for i in range(num_particles):
        #         if not self._common_td_target:
        #             td_targets = self._critic_loss.compute_td_target(
        #                 info, targets[:, :, i, ...])
        #
        #         critics_mean = critics_dist.base_dist.mean.reshape(
        #             self._mini_batch_length, -1,
        #             *critics_dist.base_dist.mean.shape[1:])
        #         critics_std = critics_dist.base_dist.stddev.reshape(
        #             self._mini_batch_length, -1,
        #             *critics_dist.base_dist.stddev.shape[1:])
        #         value_dist = td.Normal(loc=critics_mean[:-1, :, i, ...],
        #                                scale=critics_std[:-1, :, i, ...])
        #         neg_logprob.append(-value_dist.log_prob(td_targets))
        # neg_logprob = torch.stack(neg_logprob).reshape(num_particles, -1)

        return neg_logprob.mean(-1)

    def _trainable_attributes_to_ignore(self):
        return ['_critic_module', '_target_critic_params']
