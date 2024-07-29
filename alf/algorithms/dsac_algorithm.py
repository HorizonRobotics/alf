# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Distributional Soft Actor-Critic algorithm."""

import torch
from typing import Union, Callable, Optional

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import SacAlgorithm, SacInfo, ActionType
from alf.algorithms.sac_algorithm import SacCriticState
from alf.algorithms.sac_algorithm import SacCriticInfo, SacState, SacActorInfo
from alf.data_structures import TimeStep, AlgStep, namedtuple, LossInfo, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork
from alf.networks import CriticQuantileNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops

DSacInfo = namedtuple(
    "DSacInfo", [
        "reward", "step_type", "discount", "action", "action_distribution",
        "actor", "critic", "alpha", "log_pi", "discounted_return", "tau_info"
    ],
    default_value=())

TauInfo = namedtuple(
    "TauInfo", [
        "actor_tau_hat", "actor_delta_tau", "tau_hat", "delta_tau",
        "next_tau_hat", "next_delta_tau"
    ],
    default_value=())


@alf.configurable
class DSacAlgorithm(SacAlgorithm):
    """Distributional Soft Actor-Critic algorithm. 

    A SAC variant that applies the following implicit quantile regression based 
    distributional RL approach to model the critic function:

    ::
        Dabney et al "Implicit Quantile Networks for Distributional Reinforcement Learning",
        arXiv:1806.06923
    
    Currently, only continuous action space is supported, and ``need_full_rollout_state``
    is not supported.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 num_quantiles: int = 32,
                 tau_type: str = 'iqn',
                 actor_network_cls: Callable = ActorDistributionNetwork,
                 critic_network_cls: Callable = CriticQuantileNetwork,
                 repr_alg_ctor: Optional[Callable] = None,
                 epsilon_greedy: Optional[float] = None,
                 use_entropy_reward: bool = False,
                 normalize_entropy_reward: bool = False,
                 calculate_priority: bool = False,
                 num_critic_replicas: int = 2,
                 min_critic_by_critic_mean: bool = False,
                 env=None,
                 config: Optional[TrainerConfig] = None,
                 critic_loss_ctor: Optional[Callable] = None,
                 target_entropy: Optional[Union[float, Callable]] = None,
                 target_kld_per_dim: float = 3.,
                 initial_log_alpha: float = 0.0,
                 max_log_alpha: Optional[float] = None,
                 target_update_tau: float = 0.05,
                 target_update_period: int = 1,
                 dqda_clipping: Optional[float] = None,
                 actor_optimizer: Optional[torch.optim.Optimizer] = None,
                 critic_optimizer: Optional[torch.optim.Optimizer] = None,
                 alpha_optimizer: Optional[torch.optim.Optimizer] = None,
                 debug_summaries: bool = False,
                 name: str = "DSacAlgorithm"):
        r"""
        Refer to SacAlgorithm for Args beside the following. Args used for 
        discrete and mixed actions are omitted.

        Args:
            tau_type: type of the sampling method for the delta_tau of the 
                probability space, options are ['fixed', 'iqn'], indicating
                a fixed uniformly allocated probability and a random sampling
                as done in the IQN paper.
            num_quantiles: number of quantiles.
            min_critic_by_critic_mean: If True, compute the min quantile 
                distribution of critic replicas by choosing the one with the
                lowest distribution mean. Otherwise, compute the min quantile
                by taking a minimum value across all critic replicas for each
                quantile value.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
                If None, will use the following alpha based on the epistemic
                uncertainty (approximated by std) of the critics' values 
                instead of the automatic alpha optimization of SAC:

                .. math::
                    
                    \alpha = intial_alpha * std(critics)

        """
        assert tau_type in ('fixed',
                            'iqn'), f"Unsupported tau_type: {tau_type}."

        self._tau_type = tau_type
        self._num_quantiles = num_quantiles
        self._tau_spec = TensorSpec((num_quantiles, ))

        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            critic_network_cls=critic_network_cls,
            repr_alg_ctor=repr_alg_ctor,
            epsilon_greedy=epsilon_greedy,
            use_entropy_reward=use_entropy_reward,
            normalize_entropy_reward=normalize_entropy_reward,
            calculate_priority=calculate_priority,
            num_critic_replicas=num_critic_replicas,
            env=env,
            config=config,
            critic_loss_ctor=critic_loss_ctor,
            target_entropy=target_entropy,
            target_kld_per_dim=target_kld_per_dim,
            initial_log_alpha=initial_log_alpha,
            max_log_alpha=max_log_alpha,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            dqda_clipping=dqda_clipping,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            debug_summaries=debug_summaries,
            reproduce_locomotion=False,
            name=name)

        assert self._act_type == ActionType.Continuous, (
            "Only continuous action space is supported for dsac algorithm.")
        if self.has_multidim_reward():
            assert not min_critic_by_critic_mean, (
                "min_critic_by_critic_mean is not supported for multi-dim rewards."
            )

        self._min_critic_by_critic_mean = min_critic_by_critic_mean
        if alpha_optimizer is None:
            # use epistemic alpha rather than the SAC alpha optimization
            # note that self._log_alpha has been initialized as nn.Parameters
            # by SAC, so we need to delete it and reinitialize it as torch.Tensor
            self._epistemic_alpha = True
            self._min_critic_by_critic_mean = False
            self._use_entropy_reward = False
            delattr(self, '_log_alpha')
            self._log_alpha = torch.tensor(float(initial_log_alpha))
        else:
            self._epistemic_alpha = False

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       continuous_actor_network_cls, critic_network_cls,
                       q_network_cls):

        assert continuous_actor_network_cls is not None, (
            "An ActorDistributionNetwork must be provided"
            "for sampling continuous actions!")
        actor_network = continuous_actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        act_type = ActionType.Continuous
        assert critic_network_cls is not None, (
            "A CriticNetwork must be provided!")
        critic_input_spec = (observation_spec, action_spec)
        critic_network = critic_network_cls(
            input_tensor_spec=critic_input_spec,
            tau_spec=self._tau_spec,
            use_naive_parallel_network=True)
        critic_networks = critic_network.make_parallel(
            self._num_critic_replicas * reward_spec.numel)

        return critic_networks, actor_network, act_type

    def _get_tau(self, batch_size, tau_type=None):
        if tau_type is None:
            tau_type = self._tau_type
        if tau_type == 'fixed':
            delta_tau = torch.zeros(
                batch_size, self._num_quantiles) + 1. / self._num_quantiles
        elif tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            delta_tau = torch.rand(batch_size, self._num_quantiles) + 0.1
            delta_tau /= delta_tau.sum(dim=-1, keepdims=True)
        else:
            raise NotImplementedError

        # (B, N), note that they are tau_1...tau_N in the paper
        tau = torch.cumsum(delta_tau, dim=1)
        with torch.no_grad():
            tau_shift = torch.cat([torch.zeros(tau.shape[0], 1), tau[:, :-1]],
                                  dim=1)
            tau_hat = (tau + tau_shift) / 2
        return tau_hat, delta_tau

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         tau_hat,
                         critics_state,
                         delta_tau: torch.Tensor = None,
                         replica_min: bool = True):
        critic_inputs = (observation, action)
        critic_quantiles, critics_state = critic_net((critic_inputs, tau_hat),
                                                     state=critics_state)
        extra_info = ()

        # For multi-dim reward, do:
        # [B, replicas * reward_dim, n_quantiles] -> [B, replicas, reward_dim, n_quantiles]
        # For scalar reward, do nothing
        if self.has_multidim_reward():
            remaining_shape = critic_quantiles.shape[2:]
            critic_quantiles = critic_quantiles.reshape(
                -1, self._num_critic_replicas, *self._reward_spec.shape,
                *remaining_shape)
        if replica_min:
            if self._min_critic_by_critic_mean:
                # Compute the min quantile distribution of critic replicas by
                # choosing the one with the lowest distribution mean
                assert delta_tau is not None, (
                    "Input delta_tau is required for computing replica_min"
                    "by critic_mean.")
                # [B, replicas] or [B, replicas, reward_dim]
                critic_mean = (
                    critic_quantiles * delta_tau.unsqueeze(1)).sum(-1)
                idx = torch.min(
                    critic_mean, dim=1)[1]  # [B] or [B, reward_dim]
                critic_quantiles = critic_quantiles[torch.
                                                    arange(len(idx)), idx]
            else:
                # Compute the min quantile distribution by taking a minimum value
                # across all critic replicas for each quantile value
                critic_std_quantiles = critic_quantiles.std(1, unbiased=True)
                extra_info = critic_std_quantiles
                if self._num_critic_replicas == 2:
                    critic_quantiles = critic_quantiles.min(dim=1)[0]
                else:
                    critic_mean_quantiles = critic_quantiles.mean(1)
                    critic_quantiles = critic_mean_quantiles - critic_std_quantiles

        return critic_quantiles, critics_state, extra_info

    def _critic_train_step(self, inputs: TimeStep, state: SacCriticState,
                           rollout_info: SacInfo, action, tau_info: TauInfo):
        # Calculate the critics and value for both the current observation
        # and next observation
        # [B * T, n_quantiles] or [B, n_quantiles]
        critics, critics_state, _ = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_info.action,
            tau_info.tau_hat,
            state.critics,
            replica_min=False)

        with torch.no_grad():
            target_critics, target_critics_state, _ = self._compute_critics(
                self._target_critic_networks, inputs.observation, action,
                tau_info.next_tau_hat, state.target_critics,
                tau_info.next_delta_tau)

        target_critic = target_critics.detach()

        state = SacCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = SacCriticInfo(critics=critics, target_critic=target_critic)

        if self._debug_summaries and alf.summary.should_record_summaries():
            bs = tau_info.tau_hat.shape[0]
            eval_tau = self._get_tau(bs, tau_type="fixed")[0]
            with torch.no_grad():
                critics_eval = self._compute_critics(
                    self._critic_networks,
                    inputs.observation,
                    rollout_info.action,
                    eval_tau,
                    state.critics,
                    replica_min=False)[0]
                critics_eval = critics_eval.mean(dim=(0, 1))

            with alf.summary.scope(self._name):
                interval = self._num_quantiles // 5
                for idx, val in enumerate(critics_eval):
                    if idx % interval == 0:
                        quantile = int(eval_tau[0, idx] * 100)
                        alf.summary.scalar(f"ZQ_Qunatile_{quantile}", val)

        return state, info

    def _get_actor_q_value(self, inputs: TimeStep, state, action,
                           tau_info: TauInfo):
        # [B, num_quantiles]
        tau_hat, delta_tau = tau_info.actor_tau_hat, tau_info.actor_delta_tau
        critics, critics_state, critics_std = self._compute_critics(
            self._critic_networks, inputs.observation, action, tau_hat, state,
            delta_tau)

        # For multi-dim reward, critics is of shape [B, reward_dim, n_quantiles]
        # so for delta_tau: [B, n_quantiles] -> [B, 1, n_quantiles]
        # For scalar reward, do nothing
        if self.has_multidim_reward():
            delta_tau = delta_tau.unsqueeze(-2)
            critics_std = critics_std.mean(-2, keepdims=True)

        # This sum() will reduce all dims
        q_value = (critics * delta_tau).sum()
        if self._epistemic_alpha:
            q_std = (critics_std * delta_tau).sum(-1)
        else:
            q_std = ()
        return q_value, q_std, critics_state

    def _actor_train_step(self, inputs: TimeStep, state, action, log_pi,
                          tau_info: TauInfo):

        q_value, q_std, critics_state = self._get_actor_q_value(
            inputs, state, action, tau_info)
        dqda = nest_utils.grad(action, q_value)

        neg_entropy = sum(nest.flatten(log_pi))
        const_alpha = torch.exp(self._log_alpha).detach()
        entropy_loss = const_alpha * log_pi
        if self._epistemic_alpha:
            entropy_loss *= q_std.squeeze().detach()

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
            extra=SacActorInfo(actor_loss=actor_loss, neg_entropy=neg_entropy))
        return critics_state, actor_info

    def _get_tau_info(self, batch_size):
        tau_hat_samples, delta_tau_samples = self._get_tau(batch_size * 3)
        actor_tau_hat, tau_hat, next_tau_hat = tau_hat_samples.reshape(
            3, batch_size, -1)
        actor_delta_tau, delta_tau, next_delta_tau = delta_tau_samples.reshape(
            3, batch_size, -1)

        tau_info = TauInfo(
            actor_tau_hat=actor_tau_hat,
            actor_delta_tau=actor_delta_tau,
            tau_hat=tau_hat,
            delta_tau=delta_tau,
            next_tau_hat=next_tau_hat,
            next_delta_tau=next_delta_tau,
        )
        return tau_info

    def _alpha_train_step(self, log_pi):
        if self._epistemic_alpha:
            return ()
        else:
            return super()._alpha_train_step(log_pi)

    def train_step(self, inputs: TimeStep, state: SacState,
                   rollout_info: SacInfo):
        assert not self._is_eval
        self._training_started = True

        (action_distribution, action, _, action_state) = self._predict_action(
            inputs.observation, state=state.action)

        log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                    action_distribution, action)
        log_pi = sum(nest.flatten(log_pi))

        batch_size = nest.get_nest_shape(inputs.observation)[0]
        tau_info = self._get_tau_info(batch_size)

        actor_state, actor_loss = self._actor_train_step(
            inputs, state.actor, action, log_pi, tau_info)
        critic_state, critic_info = self._critic_train_step(
            inputs, state.critic, rollout_info, action, tau_info)
        alpha_loss = self._alpha_train_step(log_pi)

        state = SacState(
            action=action_state, actor=actor_state, critic=critic_state)
        info = DSacInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            action_distribution=action_distribution,
            actor=actor_loss,
            critic=critic_info,
            alpha=alpha_loss,
            log_pi=log_pi,
            tau_info=tau_info)
        return AlgStep(action, state, info)

    def _update_info_with_entropy_reward(self, info: DSacInfo):
        with torch.no_grad():
            log_pi = info.log_pi
            if self._entropy_normalizer is not None:
                log_pi = self._entropy_normalizer.normalize(log_pi)
            entropy_reward = nest.map_structure(
                lambda la, lp: -torch.exp(la) * lp, self._log_alpha, log_pi)
            entropy_reward = sum(nest.flatten(entropy_reward))
            discount = self._critic_losses[0].gamma * info.discount
            info = info._replace(
                reward=(info.reward + common.expand_dims_as(
                    entropy_reward * discount, info.reward)))
            return info

    def _calc_critic_loss(self, info: DSacInfo):
        if self._use_entropy_reward:
            info = self._update_info_with_entropy_reward(info)

        tau_info = info.tau_info
        critic_info = info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            critic_loss = l(
                info=info,
                value=critic_info.critics[:, :, i, ...],
                target_value=critic_info.target_critic,
                tau_hat=tau_info.tau_hat,
                delta_tau=tau_info.delta_tau,
                next_delta_tau=tau_info.next_delta_tau)
            critic_losses.append(critic_loss.loss)

        critic_loss = math_ops.add_n(critic_losses)

        if self._calculate_priority:
            valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = (
                (critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        return LossInfo(
            loss=critic_loss,
            priority=priority,
            extra=critic_loss / float(self._num_critic_replicas))
