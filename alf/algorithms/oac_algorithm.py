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
"""Optimistic Actor Critic algorithm."""

import torch
import torch.distributions as td
from typing import Callable, Optional

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import SacAlgorithm, SacInfo, ActionType
from alf.algorithms.sac_algorithm import SacActionState, SacCriticState, SacState
from alf.data_structures import TimeStep
from alf.data_structures import AlgStep
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import QNetwork
from alf.networks.projection_networks import NormalProjectionNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import dist_utils


def prepare_critic_action(action):
    action.requires_grad = True
    return action


def dist_transform_action(action, dist):
    for transform in dist.transforms:
        action = transform(action)
    return action


@alf.configurable
class OacAlgorithm(SacAlgorithm):
    """Optimistic Actor Critic algorithm, described in:

    ::

        Ciosek et al "Better Exploration with Optimistic Actor-Critic", arXiv:1910.12807
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 repr_alg_ctor: Optional[Callable] = None,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 calculate_priority=False,
                 num_critic_replicas=2,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 prior_actor_ctor=None,
                 target_kld_per_dim=3.,
                 initial_log_alpha=0.0,
                 explore_delta=6.8,
                 beta_ub=4.6,
                 max_log_alpha=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 parameter_reset_period=-1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 checkpoint=None,
                 debug_summaries=False,
                 reproduce_locomotion=False,
                 name="OacAlgorithm"):
        """
        Refer to SacAlgorithm for Args besides the following.

        Args:
            explore_delta (float): parameter controlling how optimistic in shifting
                the mean of the target policy to get the mean of the explore policy.
            beta_ub (float): parameter for computing the upperbound of Q value:
                :math:`Q_ub(s,a) = \mu_Q(s,a) + \beta_ub * \sigma_Q(s,a)`
        """
        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            critic_network_cls=critic_network_cls,
            repr_alg_ctor=repr_alg_ctor,
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
            parameter_reset_period=parameter_reset_period,
            dqda_clipping=dqda_clipping,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            reproduce_locomotion=reproduce_locomotion,
            name=name)

        assert self._act_type == ActionType.Continuous, (
            "Only continuous action space is supported for OacAlgorithm.")
        self._explore_delta = explore_delta
        self._beta_ub = beta_ub

    def _predict_action(self,
                        observation,
                        state: SacActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        rollout=False):
        """
        Differences between SacAlgorithm._predict_action:

        1. Only continuous actions are supported.

        2. OAC explore policy is constructed from the target policy (actor_network) 
        and used for action prediction during rollout.
        """
        new_state = SacActionState()
        action_dist, actor_network_state = self._actor_network(
            observation, state=state.actor_network)
        new_state = new_state._replace(actor_network=actor_network_state)

        if (rollout and not self._training_started):
            # get batch size with ``get_outer_rank`` and ``get_nest_shape``
            # since the observation can be a nest in the general case
            outer_rank = nest_utils.get_outer_rank(observation,
                                                   self._observation_spec)
            outer_dims = alf.nest.get_nest_shape(observation)[:outer_rank]
            # This uniform sampling seems important because for a squashed Gaussian,
            # even with a large scale, a random policy is not nearly uniform.
            action = alf.nest.map_structure(
                lambda spec: spec.sample(outer_dims=outer_dims),
                self._action_spec)
        elif rollout and self._training_started:
            action_dist_type = nest.map_structure(
                lambda dist: isinstance(dist, td.TransformedDistribution),
                action_dist)
            assert all(nest.flatten(action_dist_type)), (
                "Squashed distribution is expected from actor_network.")

            normal_dist = nest.map_structure(lambda dist: dist.base_dist,
                                             action_dist)
            normal_dist_type = nest.map_structure(
                lambda dist: isinstance(dist, dist_utils.DiagMultivariateNormal
                                        ), normal_dist)
            assert all(nest.flatten(normal_dist_type)), (
                "the base distribution should be diagonal multivariate normal."
            )

            unsquashed_mean = nest.map_structure(lambda dist: dist.mean,
                                                 normal_dist)
            unsquashed_std = nest.map_structure(lambda dist: dist.stddev,
                                                normal_dist)
            unsquashed_var = nest.map_structure(lambda dist: dist.variance,
                                                normal_dist)

            def mean_shift_fn(mu, dqda, sigma):
                if self._dqda_clipping:
                    dqda = torch.clamp(dqda, -self._dqda_clipping,
                                       self._dqda_clipping)
                norm = torch.sqrt(torch.sum(torch.mul(dqda * dqda,
                                                      sigma))) + 1e-6
                shift = self._explore_delta * torch.mul(sigma, dqda) / norm
                return mu + shift

            critic_action = nest.map_structure(prepare_critic_action,
                                               unsquashed_mean)
            with torch.enable_grad():
                transformed_action = nest.map_structure(
                    dist_transform_action, critic_action, action_dist)
                critics, critic_state = self._critic_networks(
                    (observation, transformed_action), state=state.critic)
                new_state = new_state._replace(critic=critic_state)
                if critics.ndim > 2:
                    critics = critics.squeeze()
                assert critics.ndim == 2
                q_mean = critics.mean(dim=1)
                q_std = torch.abs(critics[:, 0] - critics[:, 1]) / 2.0
                q_ub = q_mean + self._beta_ub * q_std
                dqda = nest_utils.grad(critic_action, q_ub.sum())

            shifted_mean = nest.map_structure(mean_shift_fn, unsquashed_mean,
                                              dqda, unsquashed_var)
            normal_dist = nest.map_structure(
                lambda dist_mean, dist_std: dist_utils.DiagMultivariateNormal(
                    loc=dist_mean, scale=dist_std), shifted_mean,
                unsquashed_std)
            action_dist = nest.map_structure(
                lambda base_dist, transform_dist: td.TransformedDistribution(
                    base_distribution=base_dist,
                    transforms=transform_dist.transforms), normal_dist,
                action_dist)
            action = dist_utils.sample_action_distribution(action_dist)
        else:
            if eps_greedy_sampling:
                action = dist_utils.epsilon_greedy_sample(
                    action_dist, epsilon_greedy)
            else:
                action = dist_utils.rsample_action_distribution(action_dist)

        return action_dist, action, None, new_state
