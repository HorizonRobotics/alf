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


def critic_network_cls(input_tensor_spec, quantile_size=64):
    return alf.nn.Sequential(
        alf.nn.EncodingNetwork(
            input_tensor_spec,
            fc_layer_params=(64, 64),
            last_layer_size=64,
            last_activation=alf.math.identity),
        alf.nn.QuantileProjectionNetwork(
            input_size=64, output_tensor_spec=TensorSpec((quantile_size, ))))


@alf.configurable
class QrsacAlgorithm(SacAlgorithm):
    """Quantile regression actor critic algorithm. """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=critic_network_cls,
                 epsilon_greedy=None,
                 use_entropy_reward=False,
                 calculate_priority=False,
                 num_critic_replicas=2,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 prior_actor_ctor=None,
                 target_kld_per_dim=3.,
                 initial_log_alpha=0.0,
                 max_log_alpha=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 debug_summaries=False,
                 name="QrsacAlgorithm"):
        """
        Refer to SacAlgorithm for Args. Currently only continuous action
        space is supported.
        """
        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            critic_network_cls=critic_network_cls,
            epsilon_greedy=epsilon_greedy,
            use_entropy_reward=use_entropy_reward,
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
            debug_summaries=debug_summaries,
            name=name)

        assert self._act_type == ActionType.Continuous, (
            "Only continuous action space is supported for qrsac algorithm.")

        self._num_quantiles = self._critic_networks.output_spec.shape[-1]

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         critics_state,
                         replica_min=True,
                         quantile_mean=True):
        critic_inputs = (observation, action)
        critic_quantiles, critics_state = critic_net(
            critic_inputs, state=critics_state)

        if replica_min:
            critic_quantiles = critic_quantiles.min(dim=1)[0]
        if quantile_mean:
            critic_quantiles = critic_quantiles.mean(-1)

        return critic_quantiles, critics_state

    def _critic_train_step(self, inputs: TimeStep, state: SacCriticState,
                           rollout_info: SacInfo, action, action_distribution):
        critics, critics_state = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_info.action,
            state.critics,
            replica_min=False,
            quantile_mean=False)

        target_critics, target_critics_state = self._compute_critics(
            self._target_critic_networks,
            inputs.observation,
            action,
            state.target_critics,
            quantile_mean=False)

        target_critic = target_critics.detach()

        state = SacCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = SacCriticInfo(critics=critics, target_critic=target_critic)

        return state, info
