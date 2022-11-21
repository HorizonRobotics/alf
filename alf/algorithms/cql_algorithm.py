# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Conservative Q-Learning Algorithm."""

import math
import torch
import torch.nn as nn

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.sac_algorithm import (SacAlgorithm, SacActionState,
                                          SacCriticState, SacInfo)
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.nest import nest
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import QNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import math_ops

CqlCriticInfo = namedtuple(
    "CqlCriticInfo",
    ["critics", "target_critic", "min_q_loss", "alpha_prime_loss"])

CqlLossInfo = namedtuple(
    'CqlLossInfo', ('actor', 'critic', 'alpha', "cql", "alpha_prime_loss"))


@alf.configurable
class CqlAlgorithm(SacAlgorithm):
    r"""Cql algorithm, described in:

    ::
        Kumar et al. "Conservative Q-Learning for Offline Reinforcement Learning",
        arXiv:2006.04779

    The main idea is to learn a Q-function with an additional regularizer that
    penalizes the Q-values for out-of-distribution actions. It can be shown that
    the expected value of a policy under this Q-function lower-bounds its true
    value.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 q_network_cls=QNetwork,
                 reward_weights=None,
                 epsilon_greedy=None,
                 use_entropy_reward=False,
                 normalize_entropy_reward=False,
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
                 alpha_prime_optimizer=None,
                 debug_summaries=False,
                 cql_type="H",
                 cql_action_replica=10,
                 cql_temperature=1.0,
                 cql_regularization_weight=1.0,
                 cql_target_value_gap=-1,
                 initial_log_alpha_prime=0,
                 name="CqlAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network will be called
                to sample continuous actions. All of its output specs must be
                continuous. Note that we don't need a discrete actor network
                because a discrete action can simply be sampled from the Q values.
            critic_network_cls (Callable): is used to construct critic network.
                for estimating ``Q(s,a)`` given that the action is continuous.
            q_network (Callable): is used to construct QNetwork for estimating ``Q(s,a)``
                given that the action is discrete. Its output spec must be consistent with
                the discrete action in ``action_spec``.
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the q values is used.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
            use_entropy_reward (bool): whether to include entropy as reward
            normalize_entropy_reward (bool): if True, normalize entropy reward
                to reduce bias in episodic cases. Only used if
                ``use_entropy_reward==True``.
            calculate_priority (bool): whether to calculate priority. This is
                only useful if priority replay is enabled.
            num_critic_replicas (int): number of critics to be used. Default is 2.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            initial_log_alpha (float): initial value for variable ``log_alpha``.
            max_log_alpha (float|None): if not None, ``log_alpha`` will be
                capped at this value.
            target_entropy (float|Callable|None): If a floating value, it's the
                target average policy entropy, for updating ``alpha``. If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. If ``None``, a default entropy will
                be calculated. For the mixed action type, discrete action and
                continuous action will have separate alphas and target entropies,
                so this argument can be a 2-element list/tuple, where the first
                is for discrete action and the second for continuous action.
            prior_actor_ctor (Callable): If provided, it will be called using
                ``prior_actor_ctor(observation_spec, action_spec, debug_summaries=debug_summaries)``
                to constructor a prior actor. The output of the prior actor is
                the distribution of the next action. Two prior actors are implemented:
                ``alf.algorithms.prior_actor.SameActionPriorActor`` and
                ``alf.algorithms.prior_actor.UniformPriorActor``.
            target_kld_per_dim (float): ``alpha`` is dynamically adjusted so that
                the KLD is about ``target_kld_per_dim * dim``.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between
                ``[-dqda_clipping, dqda_clipping]``. Will not perform clipping if
                ``dqda_clipping == 0``.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            alpha_prime_optimizer (torch.optim.optimizer): The optimizer for
                alpha_prime, which is the trad-off weight for the conservative
                regularization term.
            debug_summaries (bool): True if debug summaries should be created.
            cql_type (str): the type of CQL formulation: ``H`` (Eqn.(4) in IQL paper)
                or ``rho``.
            cql_action_replica (int): the number of actions to be generated for a
                single observation.
            cql_temperature (float): the temperature parameter for scaling Q
                before applying the log-sum-exp operator.
            cql_regularization_weight (float): the weight of the cql regularization
                term beforing being added to the total loss
            cql_target_value_gap (float): the target value gap between the softmax
                Q value and average Q value. The ``prime_alpha`` parameter is
                adjusted to match this target gap. A negative value corresponds
                to not enabling auto-adjusting of ``prime_alpha``.
            name (str): The name of this algorithm.
        """

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network_cls,
            critic_network_cls=critic_network_cls,
            q_network_cls=q_network_cls,
            reward_weights=reward_weights,
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
            name=name)

        assert cql_type in {'H', 'rho'}, "unknown cql_type {}".format(cql_type)

        self._cql_type = cql_type
        self._cql_target_value_gap = cql_target_value_gap
        self._cql_action_replica = cql_action_replica
        self._cql_temperature = cql_temperature
        self._cql_regularization_weight = cql_regularization_weight
        self._mini_batch_length = config.mini_batch_length

        if self._cql_target_value_gap > 0:
            self._log_alpha_prime = nn.Parameter(
                torch.tensor(float(initial_log_alpha_prime)))
            if alpha_prime_optimizer is not None:
                self.add_optimizer(alpha_prime_optimizer,
                                   [self._log_alpha_prime])

    def _critic_train_step(self, inputs: TimeStep, state: SacCriticState,
                           rollout_info: SacInfo, action, action_distribution):

        critic_state, critic_info = super()._critic_train_step(
            inputs, state, rollout_info, action, action_distribution)
        critics = critic_info.critics
        target_critic = critic_info.target_critic

        # ---- CQL specific regularizations ------
        # repeat observation and action
        # [B, d] -> [B, replica, d] -> [B * replica, d]
        B = inputs.observation.shape[0]
        rep_obs = inputs.observation.unsqueeze(1).repeat(
            1, self._cql_action_replica, 1).view(B * self._cql_action_replica,
                                                 inputs.observation.shape[1])

        # get random actions
        random_actions = self._action_spec.sample(
            (B * self._cql_action_replica, ))
        critics_random_actions, critics_state = self._compute_critics(
            self._critic_networks,
            rep_obs,
            random_actions,
            state.critics,
            replica_min=False,
            apply_reward_weights=False)

        if self._cql_type == "H":
            random_log_probs = math.log(0.5**random_actions.shape[-1])
            critics_random_actions = critics_random_actions - random_log_probs

        # [B, action_replica, critic_replica]
        critics_random_actions = critics_random_actions.reshape(
            B, self._cql_action_replica, -1)

        current_action_distribution, current_actions, _, _ = self._predict_action(
            rep_obs, SacActionState())

        current_log_pi = nest.map_structure(lambda dist, a: dist.log_prob(a),
                                            current_action_distribution,
                                            current_actions)
        current_log_pi = current_log_pi.unsqueeze(-1)

        critics_current_actions, _ = self._compute_critics(
            self._critic_networks,
            rep_obs,
            current_actions.detach(),
            state.critics,
            replica_min=False,
            apply_reward_weights=False)

        if self._cql_type == "H":
            critics_current_actions = critics_current_actions - current_log_pi.detach(
            )

        critics_current_actions = critics_current_actions.reshape(
            B, self._cql_action_replica, -1)

        # This is not mentioned in the CQL paper. But in the official CQL
        # implementation, an additional reqularization term based on
        # Q(s_current, a_next) is also added. Here we used an approximation
        # of a_next, since this is for regularization purpose.
        next_actions = current_actions.reshape(
            self._mini_batch_length, B // self._mini_batch_length, -1).roll(
                -1, 0).reshape(-1, current_actions.shape[-1])

        critics_next_actions, _ = self._compute_critics(
            self._critic_networks,
            rep_obs,
            next_actions.detach(),
            state.critics,
            replica_min=False,
            apply_reward_weights=False)

        if self._cql_type == "H":
            next_log_pi = current_log_pi.reshape(
                2, B // 2, -1).flip(0).reshape(-1, current_log_pi.shape[-1])
            critics_next_actions = critics_next_actions - next_log_pi.detach()

        critics_next_actions = critics_next_actions.reshape(
            B, self._cql_action_replica, -1)

        if self._cql_type == "H":
            cat_critics = torch.cat(
                (critics_random_actions, critics_current_actions,
                 critics_next_actions),
                dim=1)
        else:
            cat_critics = torch.cat(
                (critics_random_actions, critics_current_actions,
                 critics.unsqueeze(1)),
                dim=1)

        min_q_loss = torch.logsumexp(
            cat_critics / self._cql_temperature,
            dim=1) * self._cql_temperature - critics

        # [B, critic_replica] -> [B]
        min_q_loss = min_q_loss.mean(-1) * self._cql_regularization_weight

        if self._cql_target_value_gap > 0:
            alpha_prime = torch.clamp(
                self._log_alpha_prime.exp(), min=0.0, max=1000000.0)
            q_diff = (min_q_loss - self._cql_target_value_gap)
            min_q_loss = alpha_prime.detach() * q_diff
            alpha_prime_loss = -0.5 * (alpha_prime * q_diff.detach())
        else:
            alpha_prime_loss = torch.zeros(B)

        info = CqlCriticInfo(
            critics=critics,
            target_critic=target_critic,
            min_q_loss=min_q_loss,
            alpha_prime_loss=alpha_prime_loss,
        )

        return critic_state, info

    def calc_loss(self, info: SacInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_loss = info.actor
        min_q_loss = info.critic.min_q_loss
        alpha_prime_loss = info.critic.alpha_prime_loss

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                if self._cql_target_value_gap > 0:
                    alpha_prime = torch.clamp(
                        self._log_alpha_prime.exp(), min=0.0, max=1000000.0)
                    alf.summary.scalar("alpha_prime", alpha_prime)
                alf.summary.scalar("min_q_loss", min_q_loss.mean())

        loss = math_ops.add_ignore_empty(
            actor_loss.loss,
            critic_loss.loss + alpha_loss + min_q_loss + alpha_prime_loss)

        return LossInfo(
            loss=loss,
            priority=critic_loss.priority,
            extra=CqlLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss,
                cql=min_q_loss,
                alpha_prime_loss=alpha_prime_loss))
