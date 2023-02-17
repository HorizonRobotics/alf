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

import torch

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.algorithms.actor_bayes_critic_algorithm import AbcAlgorithm, ignore
from alf.algorithms.actor_bayes_critic_algorithm import AbcActionState
from alf.algorithms.actor_bayes_critic_algorithm import AbcCriticInfo
from alf.algorithms.actor_bayes_critic_algorithm import AbcCriticState
from alf.algorithms.actor_bayes_critic_algorithm import AbcInfo
from alf.algorithms.actor_bayes_critic_algorithm import AbcState, AbcActorInfo
from alf.algorithms.actor_bayes_critic_algorithm import AbcExploreInfo
from alf.data_structures import LossInfo, namedtuple, StepType, TimeStep
from alf.optimizers import AdamTF
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork
from alf.networks.param_networks import CriticDistributionParamNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import dist_utils, losses, math_ops
from alf.utils.summary_utils import safe_mean_hist_summary


@alf.configurable
class TsoabcAlgorithm(AbcAlgorithm):
    r"""Thompson Sampling Optimistic Actor and Bayesian Critic Algorithm. """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 explore_network_cls=ActorDistributionNetwork,
                 critic_module_cls=FuncParVIAlgorithm,
                 deterministic_actor=False,
                 deterministic_critic=False,
                 per_basin_explorer=True,
                 reward_weights=None,
                 critic_training_weight=1.0,
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
                 epistemic_alpha_coeff=None,
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
            deterministic_actor
            deterministic_critic
            per_basin_explorer,
            critic_training_weight (float|None): if not None, weight :math:`(s,a)`
                pairs for critic training according to opt_std of :math:`Q(s,a)`
                with exponent ``critic_training_weight``.
            beta_ub (float): parameter for computing the upperbound of Q value:
                :math:`Q_ub(s,a) = \mu_Q(s,a) + \beta_ub * \sigma_Q(s,a)`
            beta_lb
            epistemic_alpha_coeff (float|None): if not None, use epistemic_std 
                to the power of epistemic_alpha_coeff as alpha weights.
            explore_optimizer
            explore_alpha_optimizer
        """
        self._per_basin_explorer = per_basin_explorer
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            explore_network_cls=explore_network_cls,
            critic_module_cls=critic_module_cls,
            deterministic_actor=deterministic_actor,
            deterministic_critic=deterministic_critic,
            reward_weights=reward_weights,
            critic_training_weight=critic_training_weight,
            epsilon_greedy=epsilon_greedy,
            use_entropy_reward=use_entropy_reward,
            use_q_mean_train_actor=use_q_mean_train_actor,
            use_basin_mean_for_target_critic=use_basin_mean_for_target_critic,
            env=env,
            config=config,
            critic_loss_ctor=critic_loss_ctor,
            beta_ub=beta_ub,
            beta_lb=beta_lb,
            common_td_target=common_td_target,
            entropy_regularization_weight=entropy_regularization_weight,
            entropy_regularization=entropy_regularization,
            target_entropy=target_entropy,
            initial_log_alpha=initial_log_alpha,
            max_log_alpha=max_log_alpha,
            epistemic_alpha_coeff=epistemic_alpha_coeff,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            dqda_clipping=dqda_clipping,
            actor_optimizer=actor_optimizer,
            explore_optimizer=explore_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            explore_alpha_optimizer=explore_alpha_optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._explore_networks = self._explore_network

    def _make_modules(self, observation_spec, action_spec, reward_spec,
                      actor_network_cls, explore_network_cls,
                      critic_module_cls, critic_optimizer,
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
            optimizer=critic_optimizer)
        if self._per_basin_explorer:
            self._num_explorer_replicas = critic_module.num_basins
            self._num_critics_per_explorer = critic_module.num_particles_per_basin
        else:
            self._num_explorer_replicas = critic_module.num_particles
        explore_network = explore_network.make_parallel(
            self._num_explorer_replicas)

        return actor_network, explore_network, critic_module, target_critic_network

    def _predict_action(self,
                        observation,
                        state: AbcActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False,
                        train=False):

        new_state = AbcActionState()
        if explore:
            if self._training_started:
                # explore_network is deterministic
                action, explore_network_state = self._explore_networks(
                    observation, state=state.explore_network)
                new_state = new_state._replace(
                    explore_network=explore_network_state)
                if not train:
                    # use optimistic Q-values to select action from
                    # explore_networks's outputs
                    critic_observation = observation.repeat_interleave(
                        action.shape[1], dim=0)
                    critic_action = action.reshape(
                        action.shape[0] * action.shape[1], *action.shape[2:])
                    critic_step = self._critic_module.predict_step(
                        inputs=(critic_observation, critic_action),
                        state=state.critic)
                    critics_state = critic_step.state
                    if self._deterministic_critic:
                        critics = critic_step.output
                    else:
                        critics_dist = critic_step.output
                        critics = critics_dist.mean  # [bs*num_actions, num_critics, 1]
                    critics_info = critic_step.info
                    assert hasattr(critics_info, "epi_std")
                    if ignore(critics_info.epi_std):
                        q_epi_std = critics.std(1, unbiased=True)
                    else:
                        q_epi_std = critics_info.epi_std
                    action_q = critics.mean(1) + self._beta_ub * q_epi_std
                    action_q = action_q.reshape(
                        action.shape[0], action.shape[1], *action_q.shape[1:])
                    action_idx = action_q.squeeze(-1).max(dim=-1)[1]
                    batch_idx = torch.arange(
                        action.shape[0]).type_as(action_idx)
                    action = action[batch_idx, action_idx, :]
            else:
                # This uniform sampling during initial collect stage is
                # important since current explore_network is deterministic
                action = alf.nest.map_structure(
                    lambda spec: spec.sample(outer_dims=observation.shape[:1]),
                    self._action_spec)
            action_dist = ()
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

    def _consensus_q_for_actor_train(self, critics, explore, info=()):
        """Get q_value for _actor_train_step. 

        Args:
            critics (Tensor): output of critic_step.
            explore (bool): whether or not the outputs are for training
                the explore actor.
            info (namedtuple): info of critic_step. If critic_module is 
                ``MultiBoostEnsemble``, it contains ``total_std``,
                ``epi_std`` and ``opt_std``.

        Returns:
            q_value (Tensor): the q_value for actor training.
        """
        q_mean = critics.mean(1)
        q_total_std = critics.std(1, unbiased=True)
        if hasattr(info, "total_std"):
            if not ignore(info.total_std):
                q_total_std = info.total_std
        q_epi_std = q_total_std
        if hasattr(info, "epi_std"):
            if not ignore(info.epi_std):
                q_epi_std = info.epi_std

        if explore:
            q_value = critics
            if self._per_basin_explorer and hasattr(info, "basin_means"):
                if not ignore(info.basin_means):
                    q_value = info.basin_means
        else:
            if self._use_q_mean_train_actor:
                q_value = q_mean
            else:
                q_value = q_mean - self._beta_lb * q_total_std

        prefix = "explore_" if explore else ""
        with alf.summary.scope(self._name):
            safe_mean_hist_summary(prefix + "critics_batch_mean", q_mean)
            safe_mean_hist_summary(prefix + "critics_total_std", q_total_std)
            safe_mean_hist_summary(prefix + "critic_epi_std", q_epi_std)

        return q_value, q_epi_std

    def _actor_train_step(self,
                          inputs: TimeStep,
                          state,
                          action,
                          log_pi=(),
                          explore=False):
        if explore and self._per_basin_explorer:

            action_input = action.unsqueeze(2).expand(
                *action.shape[:2], self._num_critics_per_explorer,
                *action.shape[2:]).reshape(
                    action.shape[0],
                    action.shape[1] * self._num_critics_per_explorer,
                    *action.shape[2:])
        else:
            action_input = action
        critic_step = self._critic_module.predict_step(
            inputs=(inputs.observation, action_input), state=state)
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
            if self._epistemic_alpha_coeff is not None:
                entropy_loss *= (q_epi_std.squeeze().detach()) \
                                ** self._epistemic_alpha_coeff
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
            extra = AbcActorInfo(
                actor_loss=actor_loss, neg_entropy=neg_entropy)
        actor_info = LossInfo(loss=actor_loss + entropy_loss, extra=extra)

        return critics_state, actor_info
