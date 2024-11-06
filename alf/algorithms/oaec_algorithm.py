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
"""Optimistic Actor and Epistemic Critic Algorithm."""

import numpy as np
import functools
import torch
import torch.distributions as td
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm

from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops, summary_utils
from alf.utils.summary_utils import safe_mean_hist_summary


OaecRolloutInfo = namedtuple(
    'OaecRolloutInfo', ["action", "mask", "reward_noise"], default_value=())
OaecCriticState = namedtuple(
    "OaecCriticState", ['critics', 'target_critics'],
    default_value=())
OaecCriticInfo = namedtuple(
    "OaecCriticInfo", ["q_values", "target_q_values"], default_value=())
OaecActionState = namedtuple(
    "OaecActionState", ['actor_network', 'critics'], default_value=())
OaecState = namedtuple(
    "OaecState", ['action', 'actor', 'target_actor', 'critics'], default_value=())
OaecInfo = namedtuple(
    "OaecInfo", [
        "reward", "reward_noise", "mask", "step_type", "discount", 
        "action", "action_distribution", "actor_loss", "critic", "discounted_return"
    ],
    default_value=())
OaecLossInfo = namedtuple(
    'OaecLossInfo', ('actor', 'critic'), default_value=())


@alf.configurable
class OaecAlgorithm(OffPolicyAlgorithm):
    r"""Optimistic Actor and Epistemic Critic Algorithm.

        There is one default critic, paired with an auxiliary critic for 
        estimating its optimization variability.
        There are also n (>=1) bootstrapped critics.
        If ``correct_optimization_noise``, each bootstrapped critic is also 
        paired with an auxiliary critic.
        If ``align_optimization_noise``, each critic (original, bootstrapped,
        and their auxiliaries) will have an extra auxiliary critic trained with
        perturbed stochastic gradient steps, in order to estimate the optimization
        variability of each critic training, so that the optimization noise
        correction term can be align with the optimization of the original and
        bootstrapped critics.

        Options for critics initialization are as follows,

        1. The default setting, i.e., not ``correct_optimization_noise`` and not 
           ``align_optimization_noise``, (2n + 1) critics with the following order

            - the default critic
            - n bootstrapped critics
            - n optimization perturbed critics

        2. ``correct_optimization_noise`` but not ``align_optimization_noise``, 
           n more auxiliary critics, (2n + 2) in total

            - an auxiliary critic for the default critic and n auxiliary critics
              for the n bootstrapped critics

        3. ``align_optimization_noise`` and ``align_optimization_noise``, another 
           2n + 2 critics, (4n + 4) in total

            - (2n + 2) optimization perturbed critics for all critics in 2, 
              one for each.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 reward_weights=None,
                 reward_noise_scale=None,
                 epsilon_greedy=None,
                 calculate_priority=False,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 num_rollout_sampled_actions=10,
                 num_sampled_target_q_actions=0,
                 target_q_from_sampled_actions="max",
                 num_bootstrap_critics=1,
                 critic_replicas_deepcopy=True,
                 bootstrap_mask_prob=0.8,
                 opt_ptb_single_data=True,
                 opt_ptb_dist="exponential",
                 # correct_optimization_noise=False,
                 # align_optimization_noise=False,
                 beta_ub=1.0,
                 beta_lb=0.5,
                 output_target_critic=True,
                 use_target_actor=True,
                 std_for_overestimate='tot',
                 target_update_tau=0.05,
                 target_update_period=1,
                 initial_uniform_rollout=False,
                 dqda_clipping=None,
                 action_l2=0,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 debug_summaries=False,
                 name="OaecAlgorithm"):
        r"""
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            actor_network_ctor (Callable): Function to construct the actor network.
                ``actor_network_ctor`` needs to accept ``input_tensor_spec`` and
                ``action_spec`` as its arguments and return an actor network.
                The constructed network will be called with ``forward(observation, state)``.
            critic_network_ctor (Callable): Function to construct the critic
                network. ``critic_netwrok_ctor`` needs to accept ``input_tensor_spec``
                which is a tuple of ``(observation_spec, action_spec)``. The
                constructed network will be called with
                ``forward((observation, action), state)``.
            reward_weights (list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor.
            reward_noise_scale (None|float): If not None, randomized rewards will be
                used for bootstrapped critic training. Denotes the scale of the 
                gaussian noise added to the rollout rewards.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            calculate_priority (bool): whether to calculate priority. This is
                only useful if priority replay is enabled.
            num_rollout_sampled_actions (int): number of sampled actions in rollout.
                The one with the highest Q_value + epistemic_std will be executed.
            num_sampled_target_q_actions (int): number of sampled actions for target
                critics, default is 0, indicating no sampling, i.e., using the mean
                of the policy output.
            target_q_from_sampled_actions (str): the method to generate target q
                values from sampled actions, options are ["max", "mean"]. Only
                effective when num_sampled_target_q_actions is greater than zero.
            num_bootstrap_critics (int): a positive number of bootstrapped critics 
                for uncertainty estimation. Default is 1.
            critic_replicas_deepcopy (bool): whether to deepcopy the critic_network
                for replicas. Default is False, meaning that each critic_replica
                will have different independently instantiated parameters.
            bootstrap_mask_prob (float): the parameter of the Binomial distribution
                for independently masking out a transition to simulate bootstrapping.
            opt_ptb_single_data (bool): whether to perturb each training data
                during optimization.
            opt_ptb_dist (str): the distribution for sampling optimization perturbation.
                Options are ["exponential", "uniform"].
            correct_optimization_noise (bool): whether to correct the optimization
                variability of critic training by using auxiliary critics.
            align_optimization_noise (bool): whether to align the optimization noise
                correction with the estimated optimization variability.
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. ``env`` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            beta_ub (float): parameter for computing the upperbound of Q value:
                :math:`Q_ub(s,a) = \mu_Q(s,a) + \beta_ub * \sigma_Q(s,a)`    
            beta_lb
            output_target_critic (bool): whether to use the target critic output
                whenever critic values are needed, such as explorative rollout 
                and actor training.
            use_target_actor (bool): whether to use target actor for actor.
            std_for_overestimate (str): std type used for std for overestimation,
                options are ['tot', 'opt'].
            initial_uniform_rollout (bool): whether to use uniform rollout instead
                of random sampling actor for initial_collect_steps. 
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between ``[-dqda_clipping, dqda_clipping]``.
                Does not perform clipping if ``dqda_clipping == 0``.
            action_l2 (float): weight of squared action l2-norm on actor loss.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        assert num_bootstrap_critics >= 1, (
            "OaecAlgorithm requires a positive num_bootstrap_critics.")
        assert opt_ptb_dist in ["exponential", "uniform"], (
            "optimization perturbation distribution must be 'exponential' or 'uniform'.")
        assert std_for_overestimate in ["tot", "opt"], (
            "type of std for overestimation must be 'tot' or 'opt'.")
        assert target_q_from_sampled_actions in ["max", "mean"], (
            "target_q_from_sampled_actions must be 'max' or 'mean'.")

        self._calculate_priority = calculate_priority
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

        critic_network = critic_network_cls(
            input_tensor_spec=(observation_spec, action_spec),
            output_tensor_spec=reward_spec,
            use_naive_parallel_network=True)  # enable deepcopy when make_parallel
        actor_network = actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        self._num_bootstrap_critics = num_bootstrap_critics
        self._num_opt_ptb_critics = num_bootstrap_critics
        self._total_num_critics = 1 + self._num_bootstrap_critics \
            + self._num_opt_ptb_critics
        critic_networks = critic_network.make_parallel(
            self._total_num_critics, deepcopy=critic_replicas_deepcopy)

        self._action_l2 = action_l2
        self._reward_noise_scale = reward_noise_scale
        self._beta_ub = beta_ub
        self._beta_lb = beta_lb
        if output_target_critic:
            self._output_critic_name = 'q'
        else:
            self._output_critic_name = 'target_q'
        self._output_target_critic = output_target_critic
        self._use_target_actor = use_target_actor
        self._std_for_overestimate = std_for_overestimate
        self._num_rollout_sampled_actions = num_rollout_sampled_actions
        self._num_sampled_target_q_actions = num_sampled_target_q_actions
        self._target_q_from_sampled_actions = target_q_from_sampled_actions
        self._bootstrap_mask_prob = bootstrap_mask_prob
        self._opt_ptb_single_data = opt_ptb_single_data
        self._opt_ptb_dist = opt_ptb_dist
        # self._opt_ptb_dist = torch.distributions.Exponential(1.0)
        self._device = alf.get_default_device()
        # self._correct_optimization_noise = correct_optimization_noise
        # self._align_optimization_noise = align_optimization_noise

        action_state_spec = OaecActionState(
            actor_network=actor_network.state_spec,
            critics=critic_networks.state_spec)
        train_state_spec = OaecState(
            action=action_state_spec,
            actor=critic_networks.state_spec,
            target_actor=actor_network.state_spec,
            critics=OaecCriticState(
                critics=critic_networks.state_spec,
                target_critics=critic_networks.state_spec))

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            predict_state_spec=OaecState(action=action_state_spec),
            reward_weights=reward_weights,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])

        self._actor_network = actor_network
        self._critic_networks = critic_networks

        self._target_critic_networks = critic_networks.copy(
            name='target_critic_networks')
        original_models = [self._critic_networks]
        target_models = [self._target_critic_networks]

        if use_target_actor:
            self._target_actor_network = actor_network.copy(
                name='target_actor_networks')
            original_models.append(self._actor_network)
            target_models.append(self._target_actor_network)

        self._initial_uniform_rollout = initial_uniform_rollout

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        self._critic_losses = [None] * self._total_num_critics
        for i in range(self._total_num_critics):
            self._critic_losses[i] = critic_loss_ctor(
                name=("critic_loss" + str(i)))

        self._update_target = common.TargetUpdater(
            # models=[self._actor_network, self._critic_networks],
            # target_models=[
            #     self._target_actor_network, self._target_critic_networks
            # ],
            models=original_models,
            target_models=target_models,
            tau=target_update_tau,
            period=target_update_period)

        self._training_started = False
        self._dqda_clipping = dqda_clipping
        self._mini_batch_size = alf.get_config_value(
            'TrainerConfig.mini_batch_size')
        self._mini_batch_length = alf.get_config_value(
            'TrainerConfig.mini_batch_length')
        if opt_ptb_single_data:
            self._opt_ptb_weights = torch.empty((self._mini_batch_length, 
                                                 self._mini_batch_size, 
                                                 self._num_opt_ptb_critics))
        else:
            self._opt_ptb_weights = torch.empty((self._num_opt_ptb_critics,))

    def _predict_action(self, 
                        observation, 
                        state: OaecActionState, 
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        explore=False):
        action_dist, actor_state = self._actor_network(
            observation, state=state.actor_network)
        critic_states = state.critics

        if explore:
            if self._training_started or not self._initial_uniform_rollout:
                ## Step 1: sample multiple candidate actions from action_dist
                # [n_sampled, n_env, ...]
                actions = action_dist.sample(
                    sample_shape=(self._num_rollout_sampled_actions,))  
                # [n_sampled * n_env, ...]
                critic_actions = actions.reshape(
                    actions.shape[0] * actions.shape[1], *actions.shape[2:])
                # [n_sampled * n_env, ...]
                critic_observations = observation.repeat(
                    [self._num_rollout_sampled_actions,] + [1] * self.observation_spec.ndim)

                ## Step 2: forward critic_network to get the Q_values
                # [n_sampled * n_env, n_opt_ptb + n_bootstrap + 1]
                if self._output_target_critic:
                    q_values, critic_states = self._target_critic_networks(
                        (critic_observations, critic_actions), state=state.critics)
                else:
                    q_values, critic_states = self._critic_networks(
                        (critic_observations, critic_actions), state=state.critics)
                # [n_sampled * n_env, n_bootstrap]
                q_bootstrap = q_values[:, 1:1 + self._num_bootstrap_critics]
                # [n_sampled * n_env, n_opt_ptb]
                q_opt_ptb = q_values[:, -self._num_opt_ptb_critics:]

                ## Step 3: compute epistemic_std for each (s, a)
                # [n_sampled * n_env, n_bootstrap]
                q_bootstrap_diff = q_bootstrap - q_values[:, :1]
                # [n_sampled, n_env, n_bootstrap]
                q_bootstrap_diff = q_bootstrap_diff.reshape(
                    actions.shape[0], -1, *q_bootstrap_diff.shape[1:])
                # [n_sampled, n_env]
                q_tot_std = (q_bootstrap_diff ** 2).mean(dim=2).sqrt()

                # [n_sampled * n_env, n_bootstrap]
                q_opt_ptb_diff = q_opt_ptb - q_values[:, :1]
                # [n_sampled, n_env, n_bootstrap]
                q_opt_ptb_diff = q_opt_ptb_diff.reshape(
                    actions.shape[0], -1, *q_opt_ptb_diff.shape[1:])
                # [n_sampled, n_env]
                q_opt_std = (q_opt_ptb_diff ** 2).mean(dim=2).sqrt()

                # get a lower bound of the epistemic_std
                q_epi_std = (q_tot_std - q_opt_std).clamp_(min=0.0)

                ## Step 4: use Q_value + epistemic_std to select action for exploration
                q_mean_values = q_values[:, :1 + self._num_bootstrap_critics].mean(-1)
                # [n_sampled, n_env]
                q_values_ub = q_mean_values.reshape(
                    actions.shape[0], -1) + self._beta_ub * q_epi_std
                action_idx = q_values_ub.max(dim=0)[1]  # [n_env]
                batch_idx = torch.arange(
                    actions.shape[1]).type_as(action_idx)
                # [n_env, ...]
                action = actions[action_idx, batch_idx, ...]

                if self._debug_summaries and alf.summary.should_record_summaries():
                    with alf.summary.scope(self._name):
                        safe_mean_hist_summary(
                            f"explore/{self._output_critic_name}_tot_std", q_tot_std)
                        safe_mean_hist_summary(
                            f"explore/{self._output_critic_name}_opt_std", q_opt_std)
                        safe_mean_hist_summary(
                            f"explore/{self._output_critic_name}_epi_std", q_epi_std)

            else:
                # This uniform sampling during initial collect stage is
                # important since current explore_network is deterministic
                action = alf.nest.map_structure(
                    lambda spec: spec.sample(outer_dims=observation.shape[:1]),
                    self._action_spec)
        else:
            if eps_greedy_sampling:
                action = dist_utils.epsilon_greedy_sample(action_dist, epsilon_greedy)
            else:
                action = dist_utils.rsample_action_distribution(action_dist)

        return action_dist, action, OaecActionState(actor_network=actor_state, 
                                                    critics=critic_states)

    def predict_step(self, inputs: TimeStep, state: OaecState):
        action_dist, action, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=self._epsilon_greedy,
            eps_greedy_sampling=True)

        return AlgStep(
            output=action,
            state=OaecState(action=action_state),
            info=OaecInfo(action_distribution=action_dist))

    def rollout_step(self, inputs: TimeStep, state=None):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by OaecAlgorithm")

        _, action, action_state = self._predict_action(
            inputs.observation, 
            state.action, 
            explore=True)
        # [n_env, n_bootstrap] masks for bootstrap critics
        prob_t = torch.full((inputs.reward.shape[0], self._num_bootstrap_critics), 
                            self._bootstrap_mask_prob)
        mask = torch.bernoulli(prob_t)

        new_state = OaecState(
            action=action_state,
            actor=state.actor,
            critics=state.critics)
        info = OaecRolloutInfo(action=action,
                               mask=mask)
        if self._reward_noise_scale:
            reward_noise = self._reward_noise_scale * torch.randn(
                inputs.reward.shape + (self._num_bootstrap_critics,))
            info._replace(reward_noise=reward_noise)
        return AlgStep(
            output=action, state=new_state, info=info)

    def _critic_train_step(self, inputs: TimeStep, state: OaecCriticState,
                           rollout_info: OaecRolloutInfo, action, action_dist):
        if self._num_sampled_target_q_actions > 0:
            # [n_sampled, T*B, ...]
            sampled_actions = action_dist.sample(
                sample_shape=(self._num_sampled_target_q_actions,))
            # [n_sampled*T*B, ...]
            target_critic_actions = sampled_actions.reshape(
                sampled_actions.shape[0] * sampled_actions.shape[1], 
                *sampled_actions.shape[2:])
            target_critic_observations = inputs.observation.repeat(
                [self._num_sampled_target_q_actions,] + 
                [1] * self.observation_spec.ndim)
        else:
            target_critic_actions = action
            target_critic_observations = inputs.observation

        # [n_sampled * T*B, n_total_critics] or [T*B, n_total_critics]
        target_q_values, target_critic_states = self._target_critic_networks(
            (target_critic_observations, target_critic_actions), 
            state=state.target_critics)
        if self._num_sampled_target_q_actions > 0:
            # [n_sampled, T*B, n_total_critics]
            target_q_values = target_q_values.reshape(
                sampled_actions.shape[0], -1, self._total_num_critics)
            # [n_sampled, T*B]
            target_q_mean = target_q_values[
                :, :, :1 + self._num_bootstrap_critics].mean(-1)
            if self._std_for_overestimate == 'tot':
                target_q_bootstrap = target_q_values[
                    :, :, 1:1 + self._num_bootstrap_critics]
                target_q_bootstrap_diff = target_q_bootstrap - target_q_values[:, :, :1]
                # [n_sampled, T*B]
                target_q_std = (target_q_bootstrap_diff ** 2).mean(dim=2).sqrt()
            else:
                target_q_opt_ptb = target_q_values[:, :, -self._num_opt_ptb_critics:]
                target_q_opt_ptb_diff = target_q_opt_ptb - target_q_values[:, :, :1]
                target_q_std = (target_q_opt_ptb_diff ** 2).mean(dim=2).sqrt()

            # [n_sampled, T*B]
            target_q_lb = target_q_mean - self._beta_lb * target_q_std
            if self._target_q_from_sampled_actions == 'max':
                action_idx = target_q_lb.max(dim=0)[1]  # [T*B]
                batch_idx = torch.arange(
                    sampled_actions.shape[1]).type_as(action_idx)
                target_q_values = target_q_lb[action_idx, batch_idx]
            else:
                target_q_values = target_q_lb.mean(dim=0)

        q_values, critic_states = self._critic_networks(
            (inputs.observation, rollout_info.action), state=state.critics)

        state = OaecCriticState(
            critics=critic_states,
            target_critics=target_critic_states)

        info = OaecCriticInfo(
            q_values=q_values, target_q_values=target_q_values)

        return state, info

    def _actor_train_step(self, inputs: TimeStep, state, action):
        if self._output_target_critic:
            q_values, critic_states = self._target_critic_networks(
                (inputs.observation, action), state=state)
        else:
            q_values, critic_states = self._critic_networks(
                (inputs.observation, action), state=state)
        if self.has_multidim_reward():
            # Multidimensional reward: [B, replicas, reward_dim]
            q_values = q_values * self.reward_weights
        # use the mean of default and bootstrapped target critics
        q_value = q_values[:, :1 + self._num_bootstrap_critics].mean(-1)

        # This sum() will reduce all dims so q_value can be any rank
        dqda = nest_utils.grad(action, q_value.sum())

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            if self._action_l2 > 0:
                assert action.requires_grad
                loss += self._action_l2 * (action**2)
            loss = loss.sum(list(range(1, loss.ndim)))
            return loss

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss_info = LossInfo(loss=sum(nest.flatten(actor_loss)), extra=actor_loss)
        return critic_states, actor_loss_info

    def train_step(self, inputs: TimeStep, state: OaecState,
                   rollout_info: OaecRolloutInfo):
        self._training_started = True

        # train actor_network
        action_dist, action, action_state = self._predict_action(
            inputs.observation, state=state.action)
        actor_state, actor_loss_info = self._actor_train_step(
            inputs=inputs, state=state.actor, action=action)

        # collect infor for critic_networks training
        target_actor_state = ()
        target_critic_action = action
        target_action_dist = action_dist
        if self._use_target_actor:
            target_action_dist, target_actor_state = self._target_actor_network(
                inputs.observation, state=state.target_actor)
            target_critic_action = dist_utils.rsample_action_distribution(
                target_action_dist)
        critic_states, critic_info = self._critic_train_step(
            inputs=inputs, state=state.critics, rollout_info=rollout_info,
            action=target_critic_action, action_dist=target_action_dist)

        state = OaecState(
            action=action_state, actor=actor_state, 
            target_actor=target_actor_state, critics=critic_states),
        info = OaecInfo(
            reward=inputs.reward,
            reward_noise=rollout_info.reward_noise,
            mask=rollout_info.mask,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action_distribution=action_dist,
            critic=critic_info,
            actor_loss=actor_loss_info)
        return AlgStep(output=action, state=state, info=info)

    def calc_loss(self, info: OaecInfo):
        critic_losses = [None] * self._total_num_critics

        # compute total std of the target critic for estimation of overestimation
        if self._num_sampled_target_q_actions > 0:
            target_value = info.critic.target_q_values
        else:
            target_q = info.critic.target_q_values[:, :, :1, ...]
            target_q_mean = info.critic.target_q_values[
                :, :, :1 + self._num_bootstrap_critics, ...].mean(dim=2)  # [T, B, ...]
            if self._std_for_overestimate == 'tot':
                target_q_bootstrap = info.critic.target_q_values[
                    :, :, 1: 1 + self._num_bootstrap_critics, ...]
                target_q_bootstrap_diff = target_q_bootstrap - target_q
                # [T, B, ...]
                target_overest_std = (target_q_bootstrap_diff ** 2).mean(dim=2).sqrt()
            else:
                target_q_opt_ptb = info.critic.target_q_values[
                    :, :, -self._num_opt_ptb_critics:, ...]
                target_q_opt_ptb_diff = target_q_opt_ptb - target_q
                # [T, B, ...]
                target_overest_std = (target_q_opt_ptb_diff ** 2).mean(dim=2).sqrt()
            target_value = target_q_mean - self._beta_lb * target_overest_std

        # original critic
        critic_losses[0] = self._critic_losses[0](
            info=info,
            value=info.critic.q_values[:, :, 0, ...],
            target_value=target_value).loss

        # bootstrapped critics
        n_start = 1
        for i in range(self._num_bootstrap_critics):
            mask = info.mask[:, :, i] / self._bootstrap_mask_prob
            reward = info.reward
            if self._reward_noise_scale:
                reward += info.reward_noise[:, :, :, i]
            critic_losses[n_start + i] = mask * self._critic_losses[n_start + i](
                info=info._replace(reward=reward),
                value=info.critic.q_values[:, :, n_start + i, ...],
                target_value=target_value).loss

        # optimization perturbed critics
        if self._opt_ptb_dist == 'exponential':
            self._opt_ptb_weights.exponential_(1.0)
        else:
            self._opt_ptb_dist.uniform_(0.5, 1.5)
        n_start = 1 + self._num_bootstrap_critics
        for i in range(self._num_opt_ptb_critics):
            if self._opt_ptb_single_data:
                weights = self._opt_ptb_weights[:, :, i]
            else:
                weights = self._opt_ptb_weights[i]
            critic_losses[n_start + i] = weights * self._critic_losses[n_start + i](
                info=info,
                value=info.critic.q_values[:, :, n_start + i, ...],
                target_value=target_value).loss

        critic_loss = math_ops.add_n(critic_losses)

        if self._calculate_priority:
            valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = (
                (critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        actor_loss = info.actor_loss

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                safe_mean_hist_summary("target_value", target_value)
                if self._num_sampled_target_q_actions == 0:
                    safe_mean_hist_summary("target_q_mean", target_q_mean)
                    safe_mean_hist_summary("target_overest_std", target_overest_std)
                safe_mean_hist_summary("opt_ptb_weights", self._opt_ptb_weights)

        return LossInfo(
            loss=critic_loss + actor_loss.loss,
            priority=priority,
            extra=OaecLossInfo(critic=critic_loss, actor=actor_loss.extra))

    def after_update(self, root_inputs, info: OaecInfo):
        self._update_target()

    def _trainable_attributes_to_ignore(self):
        # return ['_target_actor_network', '_target_critic_networks']
        ignored = ['_target_critic_networks']
        if self._use_target_actor:
            ignored.append('_target_actor_network')
        return ignored
