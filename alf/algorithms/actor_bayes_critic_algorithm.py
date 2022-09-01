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
"""Generic Actor and Bayesian Critic Algorithm."""

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

AbcActionState = namedtuple("AbcActionState",
                             ["actor_network", "explore_network", "critic"],
                             default_value=())

AbcCriticState = namedtuple("AbcCriticState", ["critics", "target_critics"])

AbcState = namedtuple("AbcState", ["action", "actor", "explore", "critic"],
                       default_value=())

AbcCriticInfo = namedtuple(
    "AbcCriticInfo", ["critic_state", "target_critic"], default_value=())

AbcActorInfo = namedtuple("AbcActorInfo", ["actor_loss", "neg_entropy"],
                           default_value=())

AbcExploreInfo = namedtuple("AbcExploreInfo",
                             ["explore_loss", "neg_entropy"],
                             default_value=())

AbcInfo = namedtuple("AbcInfo", [
    "observation", "reward", "step_type", "discount", "action", 
    "action_distribution", "actor", "explore", "critic", "alpha", "log_pi"
    ],
    default_value=())

AbcLossInfo = namedtuple(
    'AbcLossInfo', ['actor', 'explore', 'critic', 'alpha', 'explore_alpha'],
    default_value=())


def get_target_updater(param_fn, target_param, tau=1.0, period=1, copy=True):
    r"""Performs a soft update of the target parameter.
    For param :math:`w_s` and its corresponding target_param :math:`w_t`, 
    a soft update is:
    .. math::
        w_t = (1 - \tau) * w_t + \tau * w_s.
    Args:
        param_fn (Callable): param_fn() returns the current parameter.
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
        _copy_parameter(param_fn(), target_param)

    def update():
        param = param_fn()
        if tau != 1.0:
            _lerp_parameter(param, target_param)
        else:
            _copy_parameter(param, target_param)

    return common.Periodically(update, period, 'periodic_update_targets')


@alf.configurable
class AbcAlgorithm(OffPolicyAlgorithm):
    r"""Actor and Bayesian Critic Algorithm. """
    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 explore_network_cls=ActorDistributionNetwork,
                 critic_module_cls=FuncParVIAlgorithm,
                 deterministic_actor=False,
                 deterministic_critic=False,
                 reward_weights=None,
                 weighted_critic_training=False,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 use_q_mean_train_actor=True,
                 use_basin_mean_for_target_critic=True,
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
                 use_epistemic_alpha=True,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 explore_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 explore_alpha_optimizer=None,
                 debug_summaries=False,
                 name="AbcAlgorithm"):
        """
        Args:
            explore_network_cls
            critic_module_cls
            deterministic_actor
            deterministic_critic
            weighted_critic_training (bool): whether or not weight :math:`(s,a)`
                pairs for critic training according to opt_std of :math:`Q(s,a)`
            beta_ub (float): parameter for computing the upperbound of Q value:
                :math:`Q_ub(s,a) = \mu_Q(s,a) + \beta_ub * \sigma_Q(s,a)`    
            beta_lb
            use_epistemic_alpha
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

        action_state_spec = AbcActionState(
            actor_network=actor_network.state_spec)
        if explore_network is not None:
            action_state_spec._replace(
            explore_network=explore_network.state_spec)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=AbcState(
                action=action_state_spec,
                actor=target_critic_network.state_spec,
                explore=target_critic_network.state_spec,
                critic=AbcCriticState(
                    critics=target_critic_network.state_spec,
                    target_critics=target_critic_network.state_spec)),
            predict_state_spec=AbcState(action=action_state_spec),
            reward_weights=reward_weights,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if explore_optimizer is not None and explore_network is not None:
            self.add_optimizer(explore_optimizer, [explore_network])
        if alpha_optimizer is None:
            self._fixed_alpha = True
        else:
            self._fixed_alpha = False
            self.add_optimizer(alpha_optimizer, nest.flatten(log_alpha))

        self._log_alpha = log_alpha
        if max_log_alpha is not None:
            self._max_log_alpha = torch.tensor(float(max_log_alpha))
        else:
            self._max_log_alpha = None

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
        self._use_q_mean_train_actor = use_q_mean_train_actor
        self._use_epistemic_alpha = use_epistemic_alpha
        self._dqda_clipping = dqda_clipping
        self._weighted_critic_training = weighted_critic_training

        self._actor_network = actor_network
        self._explore_network = explore_network

        self._critic_module = critic_module
        self._deterministic_actor = deterministic_actor
        self._deterministic_critic = deterministic_critic
        self._target_critic_params = torch.nn.Parameter(target_critic_params)
        target_critic_network.set_parameters(self._target_critic_params)
        self._target_critic_network = target_critic_network

        if use_basin_mean_for_target_critic:
            param_fn = self._critic_module.get_particles
        else:
            param_fn = lambda: self._critic_module.particles

        self._update_target_critic_params = get_target_updater(
            param_fn=param_fn,
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

        if explore_network_cls is not None:
            explore_network = explore_network_cls(
                input_tensor_spec=observation_spec, action_spec=action_spec)
        else:
            explore_network = None

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
                        state: AbcActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False,
                        train=False):
        """Predict action input observation.

        Args:
            observation (Tensor): inputs for prediction.
            state (Tensor): network state (for RNN).
            epsilon_greedy (float):
            eps_greedy_sampling (bool):
            explore (bool): whether or not predict exploration action
            train (bool): whether or not predict action for training
        Returns:
            action_dist (torch.distributions): action distribution 
            action (Tensor): action 
            state (AbcActionState)
        """
        raise NotImplementedError()

    def predict_step(self, inputs: TimeStep, state: AbcState):
        action_dist, action, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=self._epsilon_greedy,
            eps_greedy_sampling=True)

        return AlgStep(output=action,
                       state=AbcState(action=action_state),
                       info=AbcInfo(action_distribution=action_dist))

    def rollout_step(self, inputs: TimeStep, state: AbcState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        _, action, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=1.0,
            eps_greedy_sampling=True,
            explore=self._training_started)

        new_state = AbcState(action=action_state,
                             actor=state.actor,
                             explore=state.explore,
                             critic=state.critic)
        return AlgStep(
            output=action, state=new_state, info=AbcInfo(
                action=action))

    def _consensus_q_for_actor_train(self, critics, explore, info=()):
        """Get q_value for _actor_train_step. 

        Args:
            critics (Tensor): output of critic_step.
            explore (bool): whether or not to include UCB-like bonus for 
                exploration.
            info (namedtuple): info of critic_step. If critic_module is 
                ``MultiBoostEnsemble``, it contains ``total_std`` and
                ``opt_std``.

        Returns:
            q_value (Tensor): the q_value for actor training.
        """
        q_mean = critics.mean(1)
        if hasattr(info, "total_std"):
            q_total_std = info.total_std
        else:
            q_total_std = critics.std(1)  # [bs, d_out] or [bs]
        if hasattr(info, "opt_std"):
            q_opt_std = info.opt_std  # [bs, d_out] or [bs]
            q_epi_std = q_total_std - q_opt_std
        else:
            q_opt_std = None
            q_epi_std = q_total_std

        if explore:
            q_value = q_mean + self._beta_ub * q_epi_std
        else:
            if self._use_q_mean_train_actor:
                q_value = q_mean
            else:
                q_value = q_mean - self._beta_lb * q_total_std

        prefix = "explore_" if explore else ""
        with alf.summary.scope(self._name):
            summary_utils.add_mean_hist_summary(prefix + "critics_batch_mean",
                                                q_mean)
            summary_utils.add_mean_hist_summary(
                prefix + "critics_total_std", q_total_std)
            if q_opt_std is not None:
                summary_utils.add_mean_hist_summary(
                    prefix + "critic_opt_std", q_opt_std)

        return q_value, q_epi_std

    def _actor_train_step(self,
                          inputs: TimeStep,
                          state,
                          action,
                          log_pi=(),
                          explore=False):
        critic_step = self._critic_module.predict_step(
            inputs=(inputs.observation, action), state=state)
        critics_state = critic_step.state
        if self._deterministic_critic:
            critics = critic_step.output
        else:
            critics_dist = critic_step.output
            critics = critics_dist.mean
        critics_info = critic_step.info

        q_value, q_epi_std = self._consensus_q_for_actor_train(
            critics, explore, critics_info)
        dqda = nest_utils.grad(action, q_value.sum())

        if self._deterministic_actor or explore:
            neg_entropy = ()
            entropy_loss = 0.
        else:
            cont_alpha = torch.exp(self._log_alpha).detach()
            entropy_loss = cont_alpha * log_pi
            if self._use_epistemic_alpha:
                entropy_loss = q_epi_std.detach() * entropy_loss
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
            extra = AbcExploreInfo(explore_loss=actor_loss)
        else:
            extra = AbcActorInfo(actor_loss=actor_loss,
                                 neg_entropy=neg_entropy)
        actor_info = LossInfo(loss=actor_loss + entropy_loss, extra=extra)

        return critics_state, actor_info

    def _compute_critic_train_info(self, 
                                   inputs: TimeStep, 
                                   state: AbcCriticState,
                                   rollout_info: AbcInfo, 
                                   action):
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

        critic_info = AbcCriticInfo(critic_state=state.critics,
                                    target_critic=targets)

        state = AbcCriticState(critics=(),
                               target_critics=target_critics_state)

        return state, critic_info

    def _alpha_train_step(self, log_pi):
        alpha_loss = nest.map_structure(
            lambda la, lp, t: la * (-lp - t).detach(), 
            self._log_alpha, log_pi, self._target_entropy)
        return sum(nest.flatten(alpha_loss))

    def train_step(self, inputs: TimeStep, state: AbcState,
                   rollout_info: AbcInfo):

        self._training_started = True

        # train actor_network
        action_dist, action, action_state = self._predict_action(
            inputs.observation, state=state.action)
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
                extra=AbcExploreInfo(
                    explore_loss=torch.zeros_like(actor_loss.loss)))
        else:
            _, explore_action, explore_action_state = \
                self._predict_action(
                    inputs.observation, 
                    state=state.action, 
                    explore=True, 
                    train=True)
            action_state = action_state._replace(
                explore_network=explore_action_state.explore_network)

            explore_state, explore_loss = self._actor_train_step(
                inputs,
                state.explore,
                explore_action,
                explore=True)

        # compute train_info for critic_module, trained in calc_loss
        critic_state, critic_info = self._compute_critic_train_info(
            inputs, state.critic, rollout_info, action)

        if not self._deterministic_actor and not self._fixed_alpha:
            alpha_loss = self._alpha_train_step(log_pi)

        state = AbcState(action=action_state,
                         actor=actor_state,
                         explore=explore_state,
                         critic=critic_state)
        info = AbcInfo(
            observation=inputs.observation,
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            action_distribution=action_dist,
            alpha=alpha_loss,
            actor=actor_loss,
            explore=explore_loss,
            critic=critic_info,
            log_pi=log_pi)
        return AlgStep(action, state, info)

    def after_update(self, root_inputs, info: AbcInfo):
        self._update_target_critic_params()
        self._target_critic_network.set_parameters(self._target_critic_params)
        if self._max_log_alpha is not None:
            nest.map_structure(
                lambda la: la.data.copy_(torch.min(la, self._max_log_alpha)),
                self._log_alpha)

    def calc_loss(self, info: AbcInfo):
        alpha_loss = info.alpha
        actor_loss = info.actor
        explore_loss = info.explore

        assert not self._use_entropy_reward

        # train critic_module
        critic_step = self._critic_module.train_step(
            inputs=None,
            loss_func=functools.partial(self._neglogprob, info))
        critic_loss, _ = self._critic_module.update_with_gradient(
            critic_step.info)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha", self._log_alpha.exp())
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
            loss=loss,
            extra=AbcLossInfo(
                actor=actor_loss.extra,
                explore=explore_loss.extra,
                alpha=alpha_loss))

    def _neglogprob(self, info: AbcInfo, params):
        """Function computing the negative log_prob for critics.

        This function will be used as the loss_func for functional_particle_vi
        of the critics_module.

        Args:
            info (AbcInfo): 
            params (Tensor): particles of the critics_module, representing 
                params of the CriticDistributionParamNetwork.

        Returns:
            neg_logp (Tensor): of shape [B, n] or [B] (n=1)
        """
        self._critic_module.reset_param_net(params)
        num_particles = params.shape[0]
        critic_info = info.critic
        targets = critic_info.target_critic

        observation = info.observation[:-1, ...]
        action = info.action[:-1, ...]

        observation = observation.reshape(
            -1, self._observation_spec.shape[0])
        action = action.reshape(-1, self._action_spec.shape[0])

        critic_step = self._critic_module.predict_step(
            inputs=(observation, action),
            training=True,
            state=critic_info.critic_state)
        critics_dist = critic_step.output
        critics_info = critic_step.info

        # reweight training (s, a) paris with opt_std
        if hasattr(critics_info, "opt_std") and self._weighted_critic_training:
            weights = critics_info.opt_std  # [bs, d_out] or [bs]
            weights = weights.reshape(
                self._mini_batch_length-1, -1, *weights.shape[1:])
        else:
            weights = torch.ones_like(targets[1:, ...])
        weights = weights.unsqueeze(-1) / weights.sum()

        # get rewards noise if needed for initial train stage
        if self._critic_module.initial_train_stage() and \
            hasattr(self._critic_module, 'reward_perturbation'):
            reward_noise = self._critic_module.reward_perturbation(info)
        else:
            reward_noise = None

        if self._deterministic_critic:
            critics = critics_dist.squeeze(-1)
            critics = critics.reshape(self._mini_batch_length-1, -1,
                                      *critics.shape[1:])
            # in order to work with alf TDLoss, expand critics such that
            # its first dimension is of size T
            zeros = torch.zeros(1, *critics.shape[1:])
            critics = torch.cat([critics, zeros], dim=0)
            if reward_noise is not None:
                if self._common_td_target:
                    neg_logp = [self._critic_loss(
                        info=info._replace(reward=info.reward + reward_noise[i]), 
                        value=critics[:,:,i,...],
                        target_value=targets).loss for i in range(num_particles)]
                else:
                    neg_logp = [self._critic_loss( 
                        info=info._replace(reward=info.reward + reward_noise[i]), 
                        value=critics[:,:,i,...],
                        target_value=targets[:,:,i,...]).loss \
                            for i in range(num_particles)]
            else:
                if self._common_td_target:
                    neg_logp = [self._critic_loss(
                        info=info, 
                        value=critics[:,:,i,...],
                        target_value=targets).loss for i in range(num_particles)]
                else:
                    neg_logp = [self._critic_loss( 
                        info=info, 
                        value=critics[:,:,i,...],
                        target_value=targets[:,:,i,...]).loss \
                            for i in range(num_particles)]
            neg_logp = torch.stack(neg_logp).reshape(num_particles, -1)
        else:
            critics_mean = critics_dist.base_dist.mean.reshape(
                self._mini_batch_length-1, -1,
                *critics_dist.base_dist.mean.shape[1:])
            critics_std = critics_dist.base_dist.stddev.reshape(
                self._mini_batch_length-1, -1,
                *critics_dist.base_dist.stddev.shape[1:])
            if reward_noise is not None:
                if self._common_td_target:
                    td_targets = [self._critic_loss.compute_td_target(
                        info._replace(reward=info.reward + reward_noise[i]), 
                        targets) for i in range(num_particles)]
                else:
                    td_targets = [self._critic_loss.compute_td_target(
                        info._replace(reward=info.reward + reward_noise[i]), 
                            targets[:,:,i,...]) for i in range(num_particles)]
                td_targets = torch.stack(td_targets, dim=2)
            else:
                if self._common_td_target:
                    td_targets = self._critic_loss.compute_td_target(info, targets)
                    td_targets = td_targets.unsqueeze(2)
                else:
                    td_targets = [self._critic_loss.compute_td_target(
                        info, targets[:, :, i, ...]) for i in range(num_particles)]
                    td_targets = torch.stack(td_targets, dim=2)

            value_dist = td.Normal(loc=critics_mean,
                                    scale=critics_std)
            neg_logp = -value_dist.log_prob(td_targets) * weights
            neg_logp = neg_logp.transpose(0, 2)
            neg_logp = neg_logp.reshape(num_particles, -1)

        return neg_logp.mean(-1)

    def _trainable_attributes_to_ignore(self):
        return ['_critic_module', '_target_critic_params']
