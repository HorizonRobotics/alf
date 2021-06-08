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


class OacNormalProjectionNetwork(NormalProjectionNetwork):
    """NormalProjectionNetwork for OacAlgorithm. 

    Differences between NormalProjectionNetwork:

    1. If squashed, normal_dist outputs both the squashed_dist and the normal_dist
    before squash.

    2. Wrap _normal_dist for external use.

    """

    def _normal_dist(self, means, stds):
        normal_dist = dist_utils.DiagMultivariateNormal(loc=means, scale=stds)
        if self._scale_distribution:
            # The transformed distribution can also do reparameterized sampling
            # i.e., `.has_rsample=True`
            # Note that in some cases kl_divergence might no longer work for this
            # distribution! Assuming the same `transforms`, below will work:
            # ````
            # kl_divergence(Independent, Independent)
            #
            # kl_divergence(TransformedDistribution(Independent, transforms),
            #               TransformedDistribution(Independent, transforms))
            # ````
            squashed_dist = td.TransformedDistribution(
                base_distribution=normal_dist, transforms=self._transforms)
            # return squashed_dist
            return squashed_dist, normal_dist
        else:
            return normal_dist

    def normal_dist(self, means, stds):
        return self._normal_dist(means, stds)


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
                 q_network_cls=QNetwork,
                 use_entropy_reward=True,
                 use_parallel_network=False,
                 num_critic_replicas=2,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 prior_actor_ctor=None,
                 target_kld_per_dim=3.,
                 initial_log_alpha=0.0,
                 explore=True,
                 explore_delta=6.8,
                 beta_ub=4.6,
                 max_log_alpha=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 debug_summaries=False,
                 name="OacAlgorithm"):
        """
        Refer to SacAlgorithm for Args besides the following.

        Args:
            explore (bool): whether to use the explore policy in rollout_step. Only
                continuous action space is supported when `explore` is True.
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
            q_network_cls=q_network_cls,
            use_entropy_reward=use_entropy_reward,
            use_parallel_network=use_parallel_network,
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

        if explore:
            assert self._act_type == ActionType.Continuous, (
                "Only continuous action space is supported for explore mode.")
        self._explore = explore
        self._explore_delta = explore_delta
        self._beta_ub = beta_ub

    def _predict_action(self,
                        observation,
                        state: SacActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False):
        """
        Differences between SacAlgorithm._predict_action:

        1. Only continuous actions are supported.

        2. Add a switch for explore mode where OAC explore policy is constructed 
        from the target policy (actor_network) and used for action prediction.
        """
        new_state = SacActionState()
        action_dist, actor_network_state = self._actor_network(
            observation, state=state.actor_network)
        assert isinstance(action_dist, tuple), (
            "both squashed dist and original normal dist are expected.")
        action_dist, normal_dist = action_dist
        unsquashed_mean = normal_dist.mean
        unsquashed_std = normal_dist.stddev
        unsquashed_var = normal_dist.variance

        # sampled_action = dist_utils.rsample_action_distribution(action_dist)
        new_state = new_state._replace(actor_network=actor_network_state)

        def mean_shift_fn(mu, dqda, sigma):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            norm = torch.sqrt(torch.sum(torch.mul(dqda * dqda, sigma))) + 1e-6
            shift = self._explore_delta * torch.mul(sigma, dqda) / norm
            return mu + shift

        if explore:
            critic_action = normal_dist.mean.detach().clone()
            critic_action.requires_grad = True
            critics, critic_state = self._critic_networks(
                (observation, critic_action), state=state.critic)
            critics = critics.view(-1)
            q_mean = critics.mean()
            q_std = torch.abs(critics[0] - critics[1]) / 2.0
            q_ub = q_mean + self._beta_ub * q_std
            dqda = nest_utils.grad(critic_action, q_ub.sum())
            shifted_mean = nest.map_structure(mean_shift_fn, unsquashed_mean,
                                              dqda, unsquashed_var)
            action_dist, _ = self._actor_network.projection_net.normal_dist(
                shifted_mean, unsquashed_std)
            action = dist_utils.rsample_action_distribution(action_dist)
        else:
            if eps_greedy_sampling:
                action = dist_utils.epsilon_greedy_sample(
                    action_dist, epsilon_greedy)
            else:
                action = dist_utils.rsample_action_distribution(action_dist)

        return action_dist, action, None, new_state

    def rollout_step(self, time_step: TimeStep, state: SacState):
        """Same as SacAlgorithm.rollout_step except that `explore` is set to be
        `self._explore` when calling `_predict_action`.
        """
        action_dist, action, _, action_state = self._predict_action(
            time_step.observation,
            state=state.action,
            epsilon_greedy=1.0,
            eps_greedy_sampling=True,
            explore=self._explore)

        if self.need_full_rollout_state():
            _, critics_state = self._compute_critics(
                self._critic_networks, time_step.observation, action,
                state.critic.critics)
            _, target_critics_state = self._compute_critics(
                self._target_critic_networks, time_step.observation, action,
                state.critic.target_critics)
            critic_state = SacCriticState(
                critics=critics_state, target_critics=target_critics_state)
            actor_state = critics_state
        else:
            actor_state = state.actor
            critic_state = state.critic

        new_state = SacState(
            action=action_state, actor=actor_state, critic=critic_state)
        return AlgStep(
            output=action,
            state=new_state,
            info=SacInfo(action_distribution=action_dist))
