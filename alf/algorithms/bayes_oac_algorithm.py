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
"""Optimistic Actor Critic with Bayesian Critics Algorithm."""

import torch
import torch.distributions as td

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.algorithms.actor_bayes_critic_algorithm import AbcAlgorithm
from alf.algorithms.actor_bayes_critic_algorithm import AbcActionState
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import dist_utils
from alf.utils.summary_utils import safe_mean_hist_summary


@alf.configurable
class BayesOacAlgorithm(AbcAlgorithm):
    r"""Optimistic Actor Critic with Bayesian Critics. """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 explore_network_cls=None,
                 critic_module_cls=FuncParVIAlgorithm,
                 deterministic_actor=False,
                 deterministic_critic=False,
                 deterministic_explore=True,
                 reward_weights=None,
                 critic_training_weight=1.0,
                 epsilon_greedy=None,
                 use_entropy_reward=True,
                 use_q_mean_train_actor=False,
                 use_basin_mean_for_target_critic=True,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 beta_ub=4.66,
                 beta_lb=1.,
                 explore_delta=6.86,
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
                 name="BayesOacAlgorithm"):
        """
        Refer to OacAlgorithm and AbcAlgorithm for Args.
        """
        assert not deterministic_actor, "The target policy should be stochastic!"
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            explore_network_cls=None,
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
            use_epistemic_alpha=use_epistemic_alpha,
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

        self._explore_delta = explore_delta
        self._deterministic_explore = deterministic_explore

    def _predict_action(self,
                        observation,
                        state: AbcActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False,
                        train=False):
        if explore:
            assert not train, ("Explore_network is not maintained in OAC!")

        new_state = AbcActionState()
        action_dist, actor_network_state = self._actor_network(
            observation, state=state.actor_network)
        assert isinstance(action_dist, td.TransformedDistribution), (
            "Squashed distribution is expected from actor_network.")
        assert isinstance(
            action_dist.base_dist, dist_utils.DiagMultivariateNormal
        ), ("the base distribution should be diagonal multivariate normal.")
        normal_dist = action_dist.base_dist
        unsquashed_mean = normal_dist.mean
        unsquashed_std = normal_dist.stddev
        unsquashed_var = normal_dist.variance
        new_state = new_state._replace(actor_network=actor_network_state)

        def mean_shift_fn(mu, dqda, sigma, deterministic):
            if deterministic:
                cov = 1.
            else:
                cov = sigma
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            norm = torch.sqrt(torch.sum(torch.mul(dqda * dqda, cov))) + 1e-6
            shift = self._explore_delta * torch.mul(cov, dqda) / norm
            return mu + shift

        if explore:
            if self._training_started:
                critic_action = normal_dist.mean.detach().clone()
                critic_action.requires_grad = True
                transformed_action = critic_action
                with torch.enable_grad():
                    for transform in action_dist.transforms:
                        transformed_action = transform(transformed_action)

                    critic_step = self._critic_module.predict_step(
                        inputs=(observation, transformed_action),
                        state=state.critic)
                    critics_state = critic_step.state
                    new_state = new_state._replace(critic=critics_state)
                    if self._deterministic_critic:
                        critics = critic_step.output
                    else:
                        critics_dist = critic_step.output
                        critics = critics_dist.mean
                    critics_info = critic_step.info

                    q_ub, _ = self._consensus_q_for_actor_train(
                        critics, explore=True, info=critics_info)
                    dqda = nest_utils.grad(critic_action, q_ub.sum())
                shifted_mean = nest.map_structure(
                    mean_shift_fn, unsquashed_mean, dqda, unsquashed_var,
                    self._deterministic_explore)
                if self._deterministic_explore:
                    action = shifted_mean
                    action_dist = ()
                else:
                    normal_dist = dist_utils.DiagMultivariateNormal(
                        loc=shifted_mean, scale=unsquashed_std)
                    action_dist = td.TransformedDistribution(
                        base_distribution=normal_dist,
                        transforms=action_dist.transforms)
                    action = dist_utils.rsample_action_distribution(
                        action_dist)
            else:
                action = alf.nest.map_structure(
                    lambda spec: spec.sample(outer_dims=observation.shape[:1]),
                    self._action_spec)
                action_dist = ()
        else:
            if eps_greedy_sampling:
                action = dist_utils.epsilon_greedy_sample(
                    action_dist, epsilon_greedy)
            else:
                action = dist_utils.rsample_action_distribution(action_dist)

        return action_dist, action, new_state

    def _consensus_q_for_actor_train(self, critics, explore, info=()):
        q_mean = critics.mean(1)
        if hasattr(info, "total_var"):
            q_total_var = info.total_var
        else:
            q_total_var = critics.var(1, unbiased=False)  # [bs, d_out] or [bs]
        q_total_std = torch.sqrt(q_total_var)
        if hasattr(info, "opt_var"):
            q_opt_var = info.opt_var  # [bs, d_out] or [bs]
            q_epi_var = q_total_var - q_opt_var
            q_epi_std = torch.sqrt(q_epi_var)
        else:
            q_opt_var = None
            q_epi_std = q_total_std

        if explore:
            q_value = q_mean + self._beta_ub * q_epi_std
        else:
            if self._use_q_mean_train_actor:
                q_value = q_mean
            else:
                q_value = q_mean - self._beta_lb * q_total_std

        if not explore:
            with alf.summary.scope(self._name):
                safe_mean_hist_summary("critics_batch_mean", q_mean)
                safe_mean_hist_summary("critics_total_var", q_total_var)
                if q_opt_var is not None:
                    safe_mean_hist_summary("critic_opt_var", q_opt_var)

        return q_value, q_epi_std
