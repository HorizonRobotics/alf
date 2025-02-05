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
"""RLPD Algorithm."""

import torch
from typing import Callable, Optional, Union

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import ActionType, SacAlgorithm
from alf.algorithms.sac_algorithm import SacState, SacCriticState
from alf.algorithms.sac_algorithm import SacActorInfo, SacCriticInfo, SacInfo
from alf.data_structures import TimeStep, AlgStep, LossInfo
from alf.nest import nest
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common
from alf.utils.schedulers import Scheduler


@alf.configurable
class RlpdAlgorithm(SacAlgorithm):
    r"""RLPD algorithm, described in:

    ::

        Ball et al "Efficient Online Reinforcement Learning with Offline Data", arXiv:2302.02948

    Currently, only continuous action spaces are supported.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 repr_alg_ctor: Optional[Callable] = None,
                 reward_weights=None,
                 train_eps_greedy=1.0,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 normalize_entropy_reward=False,
                 calculate_priority=False,
                 num_critic_replicas=2,
                 num_critic_targets=2,
                 critic_utd_only=True,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 prior_actor_ctor=None,
                 target_kld_per_dim=3.,
                 initial_log_alpha=0.0,
                 max_log_alpha=None,
                 target_update_tau: Union[float, Scheduler] = 0.05,
                 target_update_period: Union[int, Scheduler] = 1,
                 parameter_reset_period: Union[int, Scheduler] = -1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 name="RlpdAlgorithm"):
        """
        Refer to SacAlgorithm for more details for kwargs

        Args:
            name (str): The name of this algorithm.
            num_critic_targets (int): Number of sampled subset of target critics
                for computing TD target in critic training.
            critic_utd_only (bool): Whether to only update critics following the 
                UTD setting in the ``TrainerConfig.num_updates_per_train_iter``.
                This follows the original setting in the RLPD paper.
        """
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            critic_network_cls=critic_network_cls,
            repr_alg_ctor=repr_alg_ctor,
            reward_weights=reward_weights,
            train_eps_greedy=train_eps_greedy,
            epsilon_greedy=epsilon_greedy,
            use_entropy_reward=use_entropy_reward,
            normalize_entropy_reward=normalize_entropy_reward,
            calculate_priority=calculate_priority,
            num_critic_replicas=num_critic_replicas,
            env=None,
            config=config,
            critic_loss_ctor=critic_loss_ctor,
            target_entropy=target_entropy,
            prior_actor_ctor=prior_actor_ctor,
            target_kld_per_dim=target_kld_per_dim,
            initial_log_alpha=initial_log_alpha,
            max_log_alpha=max_log_alpha,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            parameter_reset_period=parameter_reset_period,
            dqda_clipping=dqda_clipping,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        assert self._act_type == ActionType.Continuous, (
            "RLPD algorithm only supports continuous action spaces.")

        self._num_critic_targets = num_critic_targets
        self._critic_utd_only = critic_utd_only
        self._utd = alf.config_util.get_config_value("num_updates_per_train_iter")
        self._critic_train_counter = 0

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         critics_state,
                         replica_consensus='mean',
                         sample_subset=False,
                         apply_reward_weights=True):
        observation = (observation, action)
        # critics shape [B, replicas]
        critics, critics_state = critic_net(observation, state=critics_state)

        # For multi-dim reward, do
        # [B, replicas * reward_dim] -> [B, replicas, reward_dim]
        # For scalar reward, do nothing
        if self.has_multidim_reward():
            remaining_shape = critics.shape[2:]
            critics = critics.reshape(-1, self._num_critic_replicas,
                                      *self._reward_spec.shape,
                                      *remaining_shape)

        if sample_subset and self._num_critic_targets < self._num_critic_replicas:
            critics = critics[:,
                              torch.randperm(self._num_critic_replicas
                                             )[:self._num_critic_targets], ...]

        if replica_consensus == 'min':
            if self.has_multidim_reward():
                sign = self.reward_weights.sign()
                critics = (critics * sign).min(dim=1)[0] * sign
            else:
                critics = critics.min(dim=1)[0]
        elif replica_consensus == 'mean':
            if self.has_multidim_reward():
                sign = self.reward_weights.sign()
                critics = (critics * sign).mean(dim=1) * sign
            else:
                critics = critics.mean(dim=1)

        if apply_reward_weights and self.has_multidim_reward():
            critics = self._apply_reward_weights(critics)

        # The returns have the following shapes in different circumstances:
        # [replica_consensus!=None, apply_reward_weights=True]
        #   critics shape [B]
        # [replica_consensus!=None, apply_reward_weights=False]
        #   critics shape [B, reward_dim]
        # [replica_consensus=None, apply_reward_weights=False]
        #   critics shape [B, replicas, reward_dim]
        return critics, critics_state

    def _critic_train_step(self, observation, target_observation,
                           state: SacCriticState, rollout_info: SacInfo,
                           action, action_distribution):
        critics, critics_state = self._compute_critics(
            self._critic_networks,
            observation,
            rollout_info.action,
            state.critics,
            replica_consensus=None,
            apply_reward_weights=False)

        with torch.no_grad():
            target_critics, target_critics_state = self._compute_critics(
                self._target_critic_networks,
                target_observation,
                action,
                state.target_critics,
                replica_consensus='min',
                sample_subset=True,
                apply_reward_weights=False)

        target_critic = target_critics.reshape(target_critics.shape[0],
                                               *self._reward_spec.shape)

        target_critic = target_critic.detach()

        state = SacCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = SacCriticInfo(critics=critics, target_critic=target_critic)

        return state, info

    def train_step(self, inputs: TimeStep, state: SacState,
                   rollout_info: SacInfo):
        if not self._critic_utd_only:
            return super().train_step(inputs, state, rollout_info)
        elif self._critic_train_counter % self._utd == 0:
            self._critic_train_counter += 1
            return super().train_step(inputs, state, rollout_info)

        assert not self._is_eval
        self._training_started = True
        if self._target_repr_alg is not None:
            # We calculate the target observation first so that the peak memory
            # usage can be reduced because its computation graph will not be kept.
            with torch.no_grad():
                tgt_repr_step = self._target_repr_alg.predict_step(
                    inputs, state.target_repr)
                target_observation = tgt_repr_step.output
                target_repr_state = tgt_repr_step.state
        else:
            target_observation = inputs.observation
            target_repr_state = ()
        observation, new_state, info = self._repr_step("train", inputs, state,
                                                       rollout_info.repr)
        new_state = new_state._replace(target_repr=target_repr_state)

        (action_distribution, action, critics,
         action_state) = self._predict_action(
             observation, state=state.action)

        log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                    action_distribution, action)

        log_pi = sum(nest.flatten(log_pi))

        if self._prior_actor is not None:
            prior_step = self._prior_actor.train_step(inputs, ())
            log_prior = dist_utils.compute_log_probability(
                prior_step.output, action)
            log_pi = log_pi - log_prior

        critic_state, critic_info = self._critic_train_step(
            observation, target_observation, state.critic, rollout_info,
            action, action_distribution)
        self._critic_train_counter += 1

        new_state = new_state._replace(
            action=action_state, actor=state.actor, critic=critic_state)

        info = info._replace(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            action_distribution=action_distribution,
            actor=LossInfo(extra=SacActorInfo()),
            critic=critic_info,
            log_pi=log_pi,
            discounted_return=rollout_info.discounted_return)
        return AlgStep(action, new_state, info)
