# Copyright (c) 2021 Horizon Robotics. All Rights Reserved.
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

from absl import logging
from enum import Enum
import functools

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import Experience, LossInfo, namedtuple, TimeStep
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import (common, dist_utils, losses, math_ops, spec_utils,
                       tensor_utils, value_ops)
from alf.utils.conditional_ops import conditional_update
from alf.utils.summary_utils import safe_mean_hist_summary, summarize_action

TasacState = namedtuple("TasacState", ["repeats"], default_value=())

TasacCriticInfo = namedtuple(
    "TasacCriticInfo", ["critics", "target_critic", "value_loss"],
    default_value=())

TasacActorInfo = namedtuple(
    "TasacActorInfo",
    ["actor_loss", "action_entropy", "beta_entropy", "adv", "value_loss"],
    default_value=())

TasacInfo = namedtuple(
    "TasacInfo",
    ["action_distribution", "action", "actor", "critic", "alpha", "repeats"],
    default_value=())

TasacLossInfo = namedtuple('TasacLossInfo', ('actor', 'critic', 'alpha'))

Mode = Enum('AlgorithmMode', ('predict', 'rollout', 'train'))


def _safe_categorical(logits, alpha):
    r"""A numerically stable implementation of categorical distribution
    :math:`exp(\frac{Q}{\alpha})`.
    """
    logits = logits / torch.clamp(alpha, min=1e-10)
    # logits are equivalent after subtracting a common number
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    return td.Categorical(logits=logits)


def _discounted_return(rewards, values, is_lasts, discounts):
    """Computes discounted return for the first T-1 steps.

    Same with ``tf_agents.utils.value_ops``, this function returns accumulated
    discounted reward for steps that are StepType.LAST.

    Args:
        rewards (Tensor): shape is ``[T,B]`` (or ``[T]``) representing rewards.
        values (Tensor): shape is ``[T,B]`` (or ``[T]``) representing values.
        is_lasts (Tensor): shape is ``[T,B]`` (or ``[T]``) representing last steps.
        discounts (Tensor): shape is ``[T,B]`` (or ``[T]``) representing discounts.

    Returns:
        Tensor: A tensor with shape ``[T-1,B]`` (or ``[T-1]``) representing the
        discounted returns.
    """
    assert values.shape[0] >= 2, ("The sequence length needs to be "
                                  "at least 2. Got {s}".format(
                                      s=values.shape[0]))

    is_lasts = is_lasts.to(dtype=torch.float32)
    is_lasts = common.expand_dims_as(is_lasts, values)
    discounts = common.expand_dims_as(discounts, values)

    rets = torch.zeros_like(values)
    rets[-1] = values[-1]
    acc_values = rets.clone()

    with torch.no_grad():
        for t in reversed(range(rewards.shape[0] - 1)):
            rets[t] = acc_values[t + 1] * discounts[t + 1] + rewards[t + 1]
            acc_values[t] = is_lasts[t] * values[t] + (
                1 - is_lasts[t]) * rets[t]

    rets = rets[:-1]
    return rets.detach()


@alf.configurable
class TASACTDLoss(nn.Module):
    r"""This TD loss implements the unbiased multi-step Q operator
    :math:`\mathcal{T}^{\pi^{\text{ta}}}` proposed in the TASAC paper. For a sampled
    trajectory, it compares the beta action :math:`\tilde{b}_n` sampled from the
    current policy with the historical rollout beta action :math:`b_n` step by step,
    and uses the minimum :math:`n` that has :math:`\tilde{b}_n\lor b_n=1` as the
    target step for boostrapping.
    """

    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=losses.element_wise_squared_loss,
                 debug_summaries=False,
                 name="TASACTDLoss"):
        """
        Args:
            gamma (float|list[float]): A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this loss.
        """
        super().__init__()
        self._name = name
        self._gamma = torch.tensor(gamma)
        self._debug_summaries = debug_summaries
        self._td_error_loss_fn = td_error_loss_fn

    @property
    def gamma(self):
        """Return the :math:`\gamma` value for discounting future rewards.

        Returns:
            Tensor: a rank-0 or rank-1 (multi-dim reward) floating tensor.
        """
        return self._gamma.clone()

    def forward(self, experience, value, target_value, train_b):
        r"""Calculate the TD loss. The first dimension of all the tensors is the
        time dimension and the second dimesion is the batch dimension.

        Args:
            experience (Experience): experience collected from a replay buffer.
            value (torch.Tensor): the tensor for the value at each time
                step. The loss is between this and the calculated return.
            target_value (torch.Tensor): the tensor for the value at each time
                step. This is used to calculate return.
            train_b (torch.Tensor): the :math:`\beta` policy actions :math:`\tilde{b}_n`
                sampled from the current policy. These will be compared to the
                rollout :math:`\beta` policy actions :math:`b_n`
                (``experience.rollout_info.action[0]``) to decide the target
                boostrapping steps.

        Returns:
            LossInfo: TD loss with the ``extra`` field same as the loss.
        """
        if experience.reward.ndim == 3:
            # [T, B, D] or [T, B, 1]
            discounts = experience.discount.unsqueeze(-1) * self._gamma
        else:
            # [T, B]
            discounts = experience.discount * self._gamma

        rollout_b = experience.rollout_info.action[0]
        # td return till the first action switching
        b = (rollout_b | train_b).to(torch.bool)
        # b at step 0 doesn't affect the bootstrapping of any step
        b[0, :] = False

        # combine is_last and b
        is_lasts = (experience.step_type == StepType.LAST)
        is_lasts |= b

        returns = _discounted_return(
            rewards=experience.reward,
            values=target_value,
            is_lasts=is_lasts,
            discounts=discounts)

        value = value[:-1]
        loss = self._td_error_loss_fn(returns.detach(), value)
        loss = tensor_utils.tensor_extend_zero(loss)

        if loss.ndim == 3:
            # Multidimensional reward. Average over the critic loss for all
            # dimensions.
            loss = loss.mean(dim=-1)

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = experience.step_type[:-1] != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('values' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("td_error" + suffix, td, mask)

                td = returns - value
                if value.ndim == 2:
                    _summarize(value, returns, td, '')
                else:
                    for i in range(value.shape[-1]):
                        suffix = '/' + str(i)
                        _summarize(value[..., i], returns[..., i], td[..., i],
                                   suffix)

        return LossInfo(loss=loss, extra=loss)


@alf.configurable
class TasacAlgorithm(OffPolicyAlgorithm):
    r"""Temporally abstract soft actor-critic algorithm.

    In a nutsell, for inference TASAC adds a second stage that chooses between a
    candidate action output by an SAC actor and the action from the previous
    step. For policy evaluation, TASAC uses an unbiased multi-step Q operator
    for TD backup by re-using trajectories that have shared repeated actions
    between rollout and training. For policy improvement, the new actor gradient
    is approximated by multiplying a scaling factor to the
    :math:`\frac{\partial Q}{\partial a}` term in the original SAC’s actor
    gradient, where the scaling factor is the optimal probability of choosing
    the candidate action in the second stage. See

        "TASAC: Temporally Abstract Soft Actor-Critic for Continuous Control",
        Yu et al., arXiv 2021.

    for algorithm details.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 num_critic_replicas=2,
                 env=None,
                 config: TrainerConfig = None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 critic_loss_ctor=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 debug_summaries=False,
                 a1_advantage_clipping=None,
                 target_entropy=None,
                 use_entropy_reward=False,
                 name="TasacAlgorithm"):
        r"""
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (BoundedTensorSpec): representing the continuous action.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network will be called to sample continuous
                actions.
            critic_network_cls (Callable): is used to construct critic network.
                for estimating ``Q(s,a)`` given that the action is continuous.
            num_critic_replicas (int): number of critics to be used. Default is 2.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``TASACTDLoss`` will be used.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            debug_summaries (bool): True if debug summaries should be created.
            a1_advantage_clipping (None|tuple[float]): option for clipping the
                advantage (defined as :math:`Q(s,\hat{a}) - Q(s,a^-)`) when
                computing :math:`\beta_1`. If not ``None``, it should be a pair
                of numbers ``[min_adv, max_adv]``.
            target_entropy (Callable|tuple[Callable]|None): If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. If ``None``, a default entropy will
                be calculated. To set separate entropy targets for the two
                stage policies, this argument can be a tuple of two callables.
            use_entropy_reward (bool): whether to include entropy as reward.
                The default suggestion is not to use entropy reward.
            name (str): name of the algorithm
        """
        assert len(
            nest.flatten(action_spec)) == 1 and action_spec.is_continuous, (
                "Only support a single continuous action!")

        self._num_critic_replicas = num_critic_replicas

        critic_networks, actor_network = self._make_networks(
            observation_spec, action_spec, reward_spec, actor_network_cls,
            critic_network_cls)

        log_alpha = (nn.Parameter(torch.zeros(())),
                     nn.Parameter(torch.zeros(())))

        train_state_spec = TasacState(
            repeats=TensorSpec(shape=(), dtype=torch.int64))
        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            # Just for logging the action repeats statistics
            train_state_spec=train_state_spec,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, list(log_alpha))

        self._log_alpha = log_alpha
        self._log_alpha_paralist = nn.ParameterList(list(log_alpha))
        self._actor_network = actor_network
        self._critic_networks = critic_networks
        self._target_critic_networks = self._critic_networks.copy(
            name='target_critic_networks')

        if critic_loss_ctor is None:
            critic_loss_ctor = TASACTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))
        self._gamma = self._critic_losses[0]._gamma

        self._b_spec = BoundedTensorSpec(shape=(), dtype='int64', maximum=1)

        # separate target entropies for discrete and continuous actions
        if not isinstance(target_entropy, tuple):
            target_entropy = (target_entropy, ) * 2
        self._target_entropy = nest.map_structure(
            lambda spec, t: _set_target_entropy(self.name, t, [spec]),
            (self._b_spec, action_spec), target_entropy)

        self._use_entropy_reward = use_entropy_reward
        self._a1_advantage_clipping = a1_advantage_clipping

        # Create as a buffer so that training from a checkpoint will have
        # the correct flag.
        self.register_buffer("_training_started",
                             torch.zeros((), dtype=torch.bool))

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       actor_network_cls, critic_network_cls):
        def _make_parallel(net):
            return net.make_parallel(
                self._num_critic_replicas * reward_spec.numel)

        input_preprocessors = (None,
                               EmbeddingPreprocessor(
                                   input_tensor_spec=action_spec,
                                   embedding_dim=observation_spec.numel))
        # computes the action probability conditioned on s and a^-
        actor_network = actor_network_cls(
            input_tensor_spec=(observation_spec, action_spec),
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=nest_utils.NestConcat(),
            action_spec=action_spec)
        critic_network = critic_network_cls(
            input_tensor_spec=(observation_spec, action_spec))
        critic_networks = _make_parallel(critic_network)

        return critic_networks, actor_network

    def _predict_action(self,
                        time_step_or_exp,
                        state,
                        epsilon_greedy=None,
                        mode=Mode.rollout):

        # NOTE: when time_step_or_exp is from replay buffer, ``prev_action``
        # represents the previous action *during rollout* at the current step!!
        # Its value is determined by the rollout b. To use the ``prev_action``
        # from ``train_step()``, we need to store it in a train_state.

        observation, b0_action = (time_step_or_exp.observation,
                                  time_step_or_exp.prev_action)

        beta_dist, b, action_dist, action, q_values2 = self._compute_beta_and_action(
            observation, b0_action, epsilon_greedy, mode)

        if not common.is_eval() and not self._training_started:
            b = self._b_spec.sample(observation.shape[:1])
            action = self._action_spec.sample(observation.shape[:1])

        def _b1_action(action, state):
            new_state = TasacState(repeats=torch.zeros_like(state.repeats))
            return action, new_state

        # selectively update with new actions
        new_action, new_state = conditional_update(
            target=(b0_action, state),
            cond=b.to(torch.bool),
            func=_b1_action,
            action=action,
            state=state)

        new_state = new_state._replace(repeats=new_state.repeats + 1)

        return ((beta_dist, action_dist), (b, b0_action, action), new_action,
                new_state, q_values2)

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         replica_min=True,
                         apply_reward_weights=True):
        """Compute Q(s,a)"""
        observation = (observation, action)
        critics, _ = critic_net(observation)  # [B, replicas * reward_dim]
        critics = critics.reshape(  # [B, replicas, reward_dim]
            -1, self._num_critic_replicas, *self._reward_spec.shape)
        if replica_min:
            if self.has_multidim_reward():
                sign = self.reward_weights.sign()
                critics = (critics * sign).min(dim=1)[0] * sign
            else:
                critics = critics.min(dim=1)[0]

        if apply_reward_weights and self.has_multidim_reward():
            critics = critics * self.reward_weights
            critics = critics.sum(dim=-1)
        return critics

    def _alpha_train_step(self, beta_entropy, action_entropy):
        alpha_loss = (self._log_alpha[1] *
                      (action_entropy - self._target_entropy[1]).detach())
        alpha_loss += (self._log_alpha[0] *
                       (beta_entropy - self._target_entropy[0]).detach())
        return alpha_loss

    def _calc_critic_loss(self, experience, train_info: TasacInfo):
        # We need to put entropy reward in ``experience.reward`` instead of
        # ``target_critics`` because in the case of multi-step TD learning,
        # the entropy should also appear in intermediate steps!
        if self._use_entropy_reward:
            with torch.no_grad():
                actor_extra = train_info.actor.extra
                beta_alpha = self._log_alpha[0].exp()
                alpha = self._log_alpha[1].exp()
                entropy_reward = (beta_alpha * actor_extra.beta_entropy +
                                  alpha * actor_extra.action_entropy)
                gamma = self._critic_losses[0].gamma
                experience = experience._replace(
                    reward=experience.reward + entropy_reward * gamma)

        critic_info = train_info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            kwargs = dict(
                experience=experience,
                value=critic_info.critics[:, :, i, ...],
                target_value=critic_info.target_critic)
            if isinstance(l, TASACTDLoss):
                kwargs["train_b"] = train_info.action[0]

            critic_losses.append(l(**kwargs).loss)

        critic_loss = math_ops.add_n(critic_losses)
        return LossInfo(
            loss=critic_loss,
            extra=critic_loss / float(self._num_critic_replicas))

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']

    def _compute_beta_and_action(self, observation, b0_action, epsilon_greedy,
                                 mode):
        # compute resampling action dist
        action_dist, _ = self._actor_network((observation.detach(), b0_action))
        # resample a new attempting action
        if mode == Mode.predict:
            action = dist_utils.epsilon_greedy_sample(action_dist,
                                                      epsilon_greedy)
        else:
            action = dist_utils.rsample_action_distribution(action_dist)

        # compute Q(s, b0_action) and Q(s, action)
        with torch.no_grad():
            q_0 = self._compute_critics(self._critic_networks, observation,
                                        b0_action)
        q_1 = self._compute_critics(self._critic_networks, observation, action)

        q_values2 = torch.stack([q_0, q_1], dim=-1)

        # compute beta dist *conditioned* on ``action``
        with torch.no_grad():
            beta_alpha = self._log_alpha[0].exp().detach()
            if self._a1_advantage_clipping is None:
                beta_dist = _safe_categorical(q_values2, beta_alpha)
            else:
                clip_min, clip_max = self._a1_advantage_clipping
                clipped_q_values2 = (q_values2 - q_0.unsqueeze(-1)).clamp(
                    min=clip_min, max=clip_max)
                beta_dist = _safe_categorical(clipped_q_values2, beta_alpha)

        if mode == Mode.predict:
            b = dist_utils.epsilon_greedy_sample(beta_dist, epsilon_greedy)
        else:
            b = dist_utils.sample_action_distribution(beta_dist)

        return beta_dist, b, action_dist, action, q_values2

    def _actor_train_step(self, action, action_entropy, beta_dist,
                          beta_entropy, q_values2):
        alpha = self._log_alpha[1].exp().detach()
        q_a = beta_dist.probs[:, 1].detach() * q_values2[:, 1]

        dqda = nest_utils.grad(action, q_a.sum())

        def actor_loss_fn(dqda, action):
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))
        actor_loss -= alpha * action_entropy

        return LossInfo(
            loss=actor_loss,
            extra=TasacActorInfo(
                actor_loss=actor_loss,
                adv=q_values2[:, 1] - q_values2[:, 0],
                action_entropy=action_entropy,
                beta_entropy=beta_entropy))

    def _critic_train_step(self, exp: Experience, b0_action, action, beta_dist,
                           action_dist, beta_entropy, action_entropy):

        with torch.no_grad():
            target_q_0 = self._compute_critics(
                self._target_critic_networks,
                exp.observation,
                b0_action,
                apply_reward_weights=False)
            target_q_1 = self._compute_critics(
                self._target_critic_networks,
                exp.observation,
                action,
                apply_reward_weights=False)

            beta_probs = beta_dist.probs
            if self.has_multidim_reward():
                beta_probs = beta_probs.unsqueeze(1)

            target_critic = (beta_probs[..., 0] * target_q_0 +
                             beta_probs[..., 1] * target_q_1)

        critics = self._compute_critics(
            self._critic_networks,
            exp.observation,
            exp.action,
            replica_min=False,
            apply_reward_weights=False)
        return TasacCriticInfo(critics=critics, target_critic=target_critic)

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy=0.):
        action_dists, actions, new_action, new_state, _ = self._predict_action(
            time_step, state, epsilon_greedy=epsilon_greedy, mode=Mode.predict)
        return AlgStep(
            output=new_action,
            state=new_state,
            info=TasacInfo(action_distribution=action_dists, action=actions))

    def rollout_step(self, time_step: TimeStep, state):
        action_dists, actions, new_action, new_state, _ = self._predict_action(
            time_step, state, mode=Mode.rollout)
        return AlgStep(
            output=new_action,
            state=new_state,
            info=TasacInfo(
                action_distribution=action_dists,
                action=actions,
                repeats=state.repeats))

    def summarize_rollout(self, experience):
        repeats = experience.rollout_info.repeats.reshape(-1)
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                # if rollout batch size=1, hist won't show
                alf.summary.histogram("rollout_repeats/value", repeats)
                alf.summary.scalar("rollout_repeats/mean",
                                   torch.mean(repeats.to(torch.float32)))

    def train_step(self, exp: Experience, state):
        self._training_started.fill_(True)

        (action_distributions, actions, new_action, new_state,
         q_values2) = self._predict_action(
             exp, state=state, mode=Mode.train)

        beta_dist, action_dist = action_distributions
        b0_action, action = actions[1:]

        action_entropy = -dist_utils.compute_log_probability(
            action_dist, action)
        beta_entropy = beta_dist.entropy()

        actor_loss = self._actor_train_step(action, action_entropy, beta_dist,
                                            beta_entropy, q_values2)
        critic_info = self._critic_train_step(exp, b0_action, action,
                                              beta_dist, action_dist,
                                              beta_entropy, action_entropy)
        alpha_loss = self._alpha_train_step(beta_entropy, action_entropy)

        info = TasacInfo(
            action_distribution=action_distributions,
            actor=actor_loss,
            critic=critic_info,
            action=actions,
            alpha=alpha_loss,
            repeats=state.repeats)
        return AlgStep(output=new_action, state=new_state, info=info)

    def after_update(self, experience, train_info: TasacInfo):
        self._update_target()

    def calc_loss(self, experience, train_info: TasacInfo):
        critic_loss = self._calc_critic_loss(experience, train_info)
        alpha_loss = train_info.alpha
        actor_loss = train_info.actor
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha/beta", self._log_alpha[0].exp())
                alf.summary.scalar("alpha/action", self._log_alpha[1].exp())
                alf.summary.scalar("resample_advantage",
                                   torch.mean(actor_loss.extra.adv))
                p_beta0 = train_info.action_distribution[0].probs[..., 0]
                alf.summary.histogram("P_beta_0/value", p_beta0)
                alf.summary.scalar("P_beta_0/mean", p_beta0.mean())
                alf.summary.scalar("P_beta_0/std", p_beta0.std())
                repeats = train_info.repeats
                alf.summary.scalar("train_repeats/mean",
                                   torch.mean(repeats.to(torch.float32)))
                alf.summary.histogram("train_repeats/value",
                                      repeats.to(torch.float32))

        return LossInfo(
            loss=actor_loss.loss + alpha_loss + critic_loss.loss,
            extra=TasacLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss))
