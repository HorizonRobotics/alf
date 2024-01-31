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
"""Quantile Regression Soft Actor Critic Algorithm."""

import torch
import torch.distributions as td
from typing import Union, Callable, Optional

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import SacAlgorithm, SacInfo, ActionType
from alf.algorithms.sac_algorithm import SacActionState, SacCriticState
from alf.algorithms.sac_algorithm import SacCriticInfo, SacState
from alf.data_structures import TimeStep
from alf.data_structures import AlgStep
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import QNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import dist_utils


@alf.configurable
class QrsacAlgorithm(SacAlgorithm):
    """Quantile regression actor critic algorithm.

    A SAC variant that applies the following quantile regression based
    distributional RL approach to model the critic function:

    ::

        Dabney et al "Distributional Reinforcement Learning with Quantile Regression",
        arXiv:1710.10044

    Currently, only continuous action space is supported.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls: Callable = ActorDistributionNetwork,
                 critic_network_cls: Callable = CriticNetwork,
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
                 prior_actor_ctor: Optional[Callable] = None,
                 target_kld_per_dim: float = 3.,
                 initial_log_alpha: float = 0.0,
                 max_log_alpha: Optional[float] = None,
                 target_update_tau: float = 0.05,
                 target_update_period: int = 1,
                 dqda_clipping: Optional[float] = None,
                 actor_optimizer: Optional[torch.optim.Optimizer] = None,
                 critic_optimizer: Optional[torch.optim.Optimizer] = None,
                 alpha_optimizer: Optional[torch.optim.Optimizer] = None,
                 checkpoint: Optional[str] = None,
                 debug_summaries: bool = False,
                 reproduce_locomotion: bool = False,
                 name: str = "QrsacAlgorithm"):
        """
        Refer to SacAlgorithm for Args beside the following. Args used for
        discrete and mixed actions are omitted.

        Args:
            min_critic_by_critic_mean: If True, compute the min quantile
                distribution of critic replicas by choosing the one with the
                lowest distribution mean. Otherwise, compute the min quantile
                by taking a minimum value across all critic replicas for each
                quantile value.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
        """
        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            critic_network_cls=critic_network_cls,
            epsilon_greedy=epsilon_greedy,
            use_entropy_reward=use_entropy_reward,
            normalize_entropy_reward=normalize_entropy_reward,
            calculate_priority=calculate_priority,
            num_critic_replicas=num_critic_replicas,
            env=env,
            config=config,
            critic_loss_ctor=critic_loss_ctor,
            target_entropy=target_entropy,
            prior_actor_ctor=prior_actor_ctor,
            initial_log_alpha=initial_log_alpha,
            max_log_alpha=max_log_alpha,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            dqda_clipping=dqda_clipping,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            reproduce_locomotion=reproduce_locomotion,
            name=name)

        self._min_critic_by_critic_mean = min_critic_by_critic_mean
        assert self._act_type == ActionType.Continuous, (
            "Only continuous action space is supported for qrsac algorithm.")

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         critics_state,
                         replica_min: bool = True,
                         quantile_mean: bool = True):
        critic_inputs = (observation, action)
        critic_quantiles, critics_state = critic_net(
            critic_inputs, state=critics_state)

        # For multi-dim reward, do:
        #   [B, replicas * reward_dim, n_quantiles] -> [B, replicas, reward_dim, n_quantiles]
        # For scalar reward, do nothing
        if self.has_multidim_reward():
            remaining_shape = critic_quantiles.shape[2:]
            critic_quantiles = critic_quantiles.reshape(
                -1, self._num_critic_replicas, *self._reward_spec.shape,
                *remaining_shape)
        if replica_min:
            # Compute the min quantile distribution of critic replicas by
            # choosing the one with the lowest distribution mean
            if self._min_critic_by_critic_mean:
                # [B, replicas] or [B, replicas, reward_dim]
                critic_mean = critic_quantiles.mean(-1)
                idx = torch.min(
                    critic_mean, dim=1)[1]  # [B] or [B, reward_dim]
                if self.has_multidim_reward():
                    B, replicas, reward_dim = critic_mean.shape
                    critic_quantiles = critic_quantiles[
                        torch.arange(B)[:, None], idx,
                        torch.arange(reward_dim)]
                else:
                    # [B, n_quantiles]
                    critic_quantiles = critic_quantiles[torch.
                                                        arange(len(idx)), idx]
            # Compute the min quantile distribution by taking a minimum value
            # across all critic replicas for each quantile value
            else:
                critic_quantiles = critic_quantiles.min(dim=1)[0]
        if quantile_mean:
            critic_quantiles = critic_quantiles.mean(-1)

        return critic_quantiles, critics_state

    def _critic_train_step(self, observation, target_observation,
                           state: SacCriticState, rollout_info: SacInfo,
                           action, action_distribution):
        critics, critics_state = self._compute_critics(
            self._critic_networks,
            observation,
            rollout_info.action,
            state.critics,
            replica_min=False,
            quantile_mean=False)

        target_critics, target_critics_state = self._compute_critics(
            self._target_critic_networks,
            target_observation,
            action,
            state.target_critics,
            quantile_mean=False)

        target_critic = target_critics.detach()

        state = SacCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = SacCriticInfo(critics=critics, target_critic=target_critic)

        return state, info
