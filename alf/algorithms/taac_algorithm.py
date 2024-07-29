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

from enum import Enum
import functools
import numpy as np
from typing import Callable

import torch
import torch.nn as nn
import torch.distributions as td

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import LossInfo, namedtuple, TimeStep
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, dist_utils, losses, math_ops, tensor_utils
from alf.utils.conditional_ops import conditional_update
from alf.utils.summary_utils import safe_mean_hist_summary

Tau = namedtuple(
    "Tau",
    [
        "a",  # The current action value
        "v",  # The current first derivative of action (not used by action repetition)
        "u"  # The current second derivative of action (not used by action repetition)
    ],
    default_value=())

TaacState = namedtuple("TaacState", ["tau", "repeats"], default_value=())

TaacCriticInfo = namedtuple(
    "TaacCriticInfo", ["critics", "target_critic", "value_loss"],
    default_value=())

TaacActorInfo = namedtuple(
    "TaacActorInfo",
    ["actor_loss", "b1_a_entropy", "beta_entropy", "adv", "value_loss"],
    default_value=())

TaacInfo = namedtuple(
    "TaacInfo", [
        "reward", "step_type", "tau", "prev_tau", "discount",
        "action_distribution", "rollout_b", "b", "actor", "critic", "alpha",
        "repeats"
    ],
    default_value=())

TaacLossInfo = namedtuple('TaacLossInfo', ('actor', 'critic', 'alpha'))

Distributions = namedtuple("Distributions", ["beta_dist", "b1_a_dist"])

ActPredOutput = namedtuple(
    "ActPredOutput", ["dists", "b", "actor_a", "taus", "q_values2"],
    default_value=())

Mode = Enum('AlgorithmMode', ('predict', 'rollout', 'train'))


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
class TAACTDLoss(nn.Module):
    r"""This TD loss implements the compare-through multi-step Q operator
    :math:`\mathcal{T}^{\pi^{\text{ta}}}` proposed in the TAAC paper. For a sampled
    trajectory, it compares the beta action :math:`\tilde{b}_n` sampled from the
    current policy with the historical rollout beta action :math:`b_n` step by step,
    and uses the minimum :math:`n` that has :math:`\tilde{b}_n\lor b_n=1` as the
    target step for bootstrapping.
    """

    def __init__(self,
                 gamma=0.99,
                 td_error_loss_fn=losses.element_wise_squared_loss,
                 debug_summaries=False,
                 name="TAACTDLoss"):
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

    def forward(self, info, value, target_value):
        r"""Calculate the TD loss. The first dimension of all the tensors is the
        time dimension and the second dimension is the batch dimension.

        Args:
            info (TaacInfo): TaacInfo collected from train_step().
            value (torch.Tensor): the tensor for the value at each time
                step. The loss is between this and the calculated return.
            target_value (torch.Tensor): the tensor for the value at each time
                step. This is used to calculate return.

        Returns:
            LossInfo: TD loss with the ``extra`` field same as the loss.
        """
        train_b = info.b
        if info.reward.ndim == 3:
            # [T, B, D] or [T, B, 1]
            discounts = info.discount.unsqueeze(-1) * self._gamma
        else:
            # [T, B]
            discounts = info.discount * self._gamma

        rollout_b = info.rollout_b
        # td return till the first action switching
        b = (rollout_b | train_b).to(torch.bool)
        # b at step 0 doesn't affect the bootstrapping of any step
        b[0, :] = False

        # combine is_last and b
        is_lasts = (info.step_type == StepType.LAST)
        is_lasts |= b

        returns = _discounted_return(
            rewards=info.reward,
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
            mask = info.step_type[:-1] != StepType.LAST
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
class TaacAlgorithmBase(OffPolicyAlgorithm):
    r"""Temporally abstract actor-critic algorithm.

    In a nutsell, for inference TAAC adds a second stage that chooses between a
    candidate trajectory :math:`\hat{\tau}` output by an SAC actor and the previous
    trajectory :math:`\tau^-`. For policy evaluation, TAAC uses a compare-through Q
    operator for TD backup by re-using state-action sequences that have shared
    actions between rollout and training. For policy improvement, the
    new actor gradient is approximated by multiplying a scaling factor to the
    :math:`\frac{\partial Q}{\partial a}` term in the original SAC’s actor
    gradient, where the scaling factor is the optimal probability of choosing
    the :math:`\hat{\tau}` in the second stage.

    Different sub-algorithms implement different forms of the 'trajectory' concept,
    for example, it can be a constant function representing the same action, or
    a quadratic function.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 actor_observation_processors=alf.layers.Detach(),
                 reward_weights=None,
                 num_critic_replicas=2,
                 epsilon_greedy=None,
                 env=None,
                 config: TrainerConfig = None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 critic_loss_ctor=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 initial_alpha=1.,
                 debug_summaries=False,
                 randomize_first_state_tau=False,
                 b1_advantage_clipping=None,
                 max_repeat_steps=None,
                 target_entropy=None,
                 checkpoint=None,
                 name="TaacAlgorithmBase"):
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
            actor_observation_processors (Nest): a nest of observation processors
                applied to the inputs of the actor network. Note that any configured
                ``input_preprocessors`` of ``actor_network_cls`` will be overwritten
                by a tuple of this one and a preprocessor of the prev action, for
                modeling :math:`\pi(a|s,a^-)`.
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the q values is used.
            num_critic_replicas (int): number of critics to be used. Default is 2.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``config.epsilon_greedy`` and then
                ``alf.get_config_value(TrainerConfig.epsilon_greedy)``.
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
                constructor. If ``None``, a default ``TAACTDLoss`` will be used.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            initial_alpha (float): the initial entropy weight for both policies.
            debug_summaries (bool): True if debug summaries should be created.
            randomize_first_state_tau (bool): whether to randomize ``state.tau``
                at the beginning of an episode during rollout and training.
                Potentially this helps exploration. This was turned off in
                Yu et al. 2021.
            b1_advantage_clipping (None|tuple[float]): option for clipping the
                advantage (defined as :math:`Q(s,\hat{\tau}) - Q(s,\tau^-)`) when
                computing :math:`\beta_1`. If not ``None``, it should be a pair
                of numbers ``[min_adv, max_adv]``.
            max_repeat_steps (None|int): the max number of steps to repeat during
                rollout and evaluation. This value doesn't impact the switch
                during training.
            target_entropy (Callable|tuple[Callable]|None): If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. If ``None``, a default entropy will
                be calculated. To set separate entropy targets for the two
                stage policies, this argument can be a tuple of two callables.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            name (str): name of the algorithm
        """
        assert len(
            nest.flatten(action_spec)) == 1 and action_spec.is_continuous, (
                "Only support a single continuous action!")

        self._num_critic_replicas = num_critic_replicas
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

        self._tau_spec, critic_networks, actor_network = self._make_networks(
            observation_spec, action_spec, reward_spec, actor_network_cls,
            actor_observation_processors, critic_network_cls)

        log_alpha = (nn.Parameter(torch.tensor(np.log(initial_alpha))),
                     nn.Parameter(torch.tensor(np.log(initial_alpha))))

        assert (len(alf.nest.flatten(critic_networks.state_spec)) == 0
                and len(alf.nest.flatten(actor_network.state_spec)) == 0), (
                    "Don't support stateful critic or actor network!")

        train_state_spec = TaacState(
            tau=self._tau_spec,
            repeats=TensorSpec(shape=(), dtype=torch.int64))
        super().__init__(
            observation_spec,
            action_spec,
            reward_spec=reward_spec,
            train_state_spec=train_state_spec,
            reward_weights=reward_weights,
            env=env,
            config=config,
            checkpoint=checkpoint,
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
            critic_loss_ctor = TAACTDLoss
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

        self._b1_advantage_clipping = b1_advantage_clipping
        self._max_repeat_steps = max_repeat_steps
        self._randomize_first_state_tau = randomize_first_state_tau

        # Create as a buffer so that training from a checkpoint will have
        # the correct flag.
        self.register_buffer("_training_started",
                             torch.zeros((), dtype=torch.bool))

        self._update_target = common.TargetUpdater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       actor_network_cls, actor_observation_processors,
                       critic_network_cls):
        raise NotImplementedError()

    def _update_tau(self, tau):
        """Update the current trajectory ``tau`` by moving one step ahead."""
        raise NotImplementedError()

    def _action2tau(self, a, tau):
        """Compute a new trajectory given a new action and the current trajectory
        ``tau``."""
        raise NotImplementedError()

    def _make_networks_impl(self, observation_spec, action_spec, reward_spec,
                            actor_network_cls, actor_observation_processors,
                            critic_network_cls, tau_mask):
        def _make_parallel(net):
            return net.make_parallel(
                self._num_critic_replicas * reward_spec.numel)

        tau_spec = nest.map_structure(lambda m: action_spec if m else (),
                                      tau_mask)
        obs_dim = sum([spec.numel for spec in nest.flatten(observation_spec)])
        tau_embedding = nest.map_structure(
            lambda _: torch.nn.Sequential(
                alf.layers.FC(action_spec.numel, obs_dim)), tau_spec)

        actor_network = actor_network_cls(
            input_tensor_spec=(observation_spec, tau_spec),
            input_preprocessors=(actor_observation_processors, tau_embedding),
            preprocessing_combiner=nest_utils.NestConcat(),
            action_spec=action_spec)
        critic_network = critic_network_cls(
            input_tensor_spec=(observation_spec, tau_spec),
            action_preprocessing_combiner=nest_utils.NestConcat())
        critic_networks = _make_parallel(critic_network)

        return tau_spec, critic_networks, actor_network

    def _randomize_first_tau(self, time_step_or_exp, state, rollout_tau=None):
        """Randomize the first ``tau`` (by default always 0) for better
        exploration if ``b=0`` is selected.

        If a ``rollout_tau`` is already provided, then directly use it (during
        training).
        """

        def _randomize(tau):
            return alf.nest.map_structure(
                lambda spec: spec.sample(outer_dims=tau.a.shape[:1]),
                self._tau_spec)

        if rollout_tau is None:
            kwargs = dict(tau=state.tau)
            randomize = _randomize
        else:
            kwargs = dict(r_tau=rollout_tau)
            randomize = lambda r_tau: r_tau

        tau = conditional_update(
            target=state.tau,
            cond=(time_step_or_exp.step_type == StepType.FIRST),
            func=randomize,
            **kwargs)
        return state._replace(tau=tau)

    def _predict_action(self,
                        time_step,
                        state,
                        epsilon_greedy=None,
                        mode=Mode.rollout):

        observation = time_step.observation

        ap_out = self._compute_beta_and_tau(observation, state, epsilon_greedy,
                                            mode)

        if not common.is_eval() and not self._training_started:
            b = self._b_spec.sample(time_step.step_type.shape)
            b1_a = self._action_spec.sample(time_step.step_type.shape)
            b1_tau = self._action2tau(b1_a, state.tau)
            ap_out = ap_out._replace(b=b, taus=(ap_out.taus[0], b1_tau))

        b0_tau, b1_tau = ap_out.taus
        new_state = state._replace(tau=b0_tau)

        def _b1_action(b1_tau, new_state):
            new_state = new_state._replace(
                repeats=torch.zeros_like(new_state.repeats), tau=b1_tau)
            return new_state

        condition = ap_out.b.to(torch.bool)
        if self._max_repeat_steps is not None and mode != Mode.train:
            condition |= (state.repeats >= self._max_repeat_steps)

        # selectively update with new actions
        new_state = conditional_update(
            target=new_state,
            cond=condition,
            func=_b1_action,
            b1_tau=b1_tau,
            new_state=new_state)

        new_state = new_state._replace(repeats=new_state.repeats + 1)
        return ap_out, new_state

    def _compute_critics(self,
                         critic_net,
                         observation,
                         tau,
                         replica_min=True,
                         apply_reward_weights=True):
        """Compute Q(s,a)"""
        observation = (observation, tau)
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

    def _calc_critic_loss(self, info: TaacInfo):
        critic_info = info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            kwargs = dict(
                info=info,
                value=critic_info.critics[:, :, i, ...],
                target_value=critic_info.target_critic)
            critic_losses.append(l(**kwargs).loss)

        critic_loss = math_ops.add_n(critic_losses)
        return LossInfo(
            loss=critic_loss,
            extra=critic_loss / float(self._num_critic_replicas))

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']

    def _build_beta_dist(self, q_values2):
        def _safe_categorical(logits, alpha):
            r"""A numerically stable implementation of categorical distribution
            :math:`exp(\frac{Q}{\alpha})`.
            """
            logits = logits / torch.clamp(alpha, min=1e-10)
            # logits are equivalent after subtracting a common number
            logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
            return td.Categorical(logits=logits)

        # compute beta dist *conditioned* on ``action``
        with torch.no_grad():
            beta_alpha = self._log_alpha[0].exp().detach()
            if self._b1_advantage_clipping is None:
                beta_dist = _safe_categorical(q_values2, beta_alpha)
            else:
                clip_min, clip_max = self._b1_advantage_clipping
                # The first dim [..., 0] is always 0
                q_values2 = q_values2 - q_values2[..., :1]
                q_values2[..., 1] = q_values2[..., 1].clamp(
                    min=clip_min, max=clip_max)
                beta_dist = _safe_categorical(q_values2, beta_alpha)

        return beta_dist

    def _compute_beta_and_tau(self, observation, state, epsilon_greedy, mode):
        # compute resampling action dist
        b1_a_dist, _ = self._actor_network((observation, state.tau))
        # resample a new attempting action
        if mode == Mode.predict:
            b1_a = dist_utils.epsilon_greedy_sample(b1_a_dist, epsilon_greedy)
        else:
            b1_a = dist_utils.rsample_action_distribution(b1_a_dist)

        b0_tau = self._update_tau(state.tau)
        # This should be a deterministic function converting b1_a to b1_tau
        b1_tau = self._action2tau(b1_a, state.tau)

        # compute Q(s, tau^-) and Q(s, \hat{tau})
        with torch.no_grad():
            q_0 = self._compute_critics(self._critic_networks, observation,
                                        b0_tau)
        q_1 = self._compute_critics(self._critic_networks, observation, b1_tau)

        q_values2 = torch.stack([q_0, q_1], dim=-1)
        beta_dist = self._build_beta_dist(q_values2)

        if mode == Mode.predict:
            b = dist_utils.epsilon_greedy_sample(beta_dist, epsilon_greedy)
        else:
            b = dist_utils.sample_action_distribution(beta_dist)

        dists = Distributions(beta_dist=beta_dist, b1_a_dist=b1_a_dist)
        return ActPredOutput(
            dists=dists,
            b=b,
            actor_a=b1_a,
            taus=(b0_tau, b1_tau),
            q_values2=q_values2)

    def _actor_train_step(self, a, b1_a_entropy, beta_dist, beta_entropy,
                          q_values2):
        alpha = self._log_alpha[1].exp().detach()
        q_a = beta_dist.probs[:, 1].detach() * q_values2[:, 1]

        dqda = nest_utils.grad(a, q_a.sum())

        def actor_loss_fn(dqda, action):
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        actor_loss = nest.map_structure(actor_loss_fn, dqda, a)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))
        actor_loss -= alpha * b1_a_entropy

        return LossInfo(
            loss=actor_loss,
            extra=TaacActorInfo(
                actor_loss=actor_loss,
                adv=q_values2[:, 1] - q_values2[:, 0],
                b1_a_entropy=b1_a_entropy,
                beta_entropy=beta_entropy))

    def _critic_train_step(self, inputs: TimeStep, rollout_tau, b0_tau, b1_tau,
                           beta_dist):

        with torch.no_grad():
            target_q_0 = self._compute_critics(
                self._target_critic_networks,
                inputs.observation,
                b0_tau,
                apply_reward_weights=False)
            target_q_1 = self._compute_critics(
                self._target_critic_networks,
                inputs.observation,
                b1_tau,
                apply_reward_weights=False)

            beta_probs = beta_dist.probs
            if self.has_multidim_reward():
                beta_probs = beta_probs.unsqueeze(1)

            target_critic = (beta_probs[..., 0] * target_q_0 +
                             beta_probs[..., 1] * target_q_1)

        critics = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_tau,
            replica_min=False,
            apply_reward_weights=False)
        return TaacCriticInfo(critics=critics, target_critic=target_critic)

    def predict_step(self, inputs: TimeStep, state):
        ap_out, new_state = self._predict_action(
            inputs,
            state,
            epsilon_greedy=self._epsilon_greedy,
            mode=Mode.predict)
        return AlgStep(
            output=new_state.tau.a,
            state=new_state,
            info=TaacInfo(action_distribution=ap_out.dists, b=ap_out.b))

    def rollout_step(self, inputs: TimeStep, state):
        if self._randomize_first_state_tau:
            state = self._randomize_first_tau(inputs, state)
        ap_out, new_state = self._predict_action(
            inputs, state, mode=Mode.rollout)
        return AlgStep(
            output=new_state.tau.a,
            state=new_state,
            info=TaacInfo(
                action_distribution=ap_out.dists,
                prev_tau=state.tau,  # for getting randomized tau in training
                tau=new_state.tau,  # for critic training
                b=ap_out.b,
                repeats=state.repeats))

    def summarize_rollout(self, experience):
        repeats = experience.rollout_info.repeats.reshape(-1)
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                # if rollout batch size=1, hist won't show
                alf.summary.histogram("rollout_repeats/value", repeats)
                alf.summary.scalar("rollout_repeats/mean",
                                   torch.mean(repeats.to(torch.float32)))

    def train_step(self, inputs: TimeStep, state, rollout_info: TaacInfo):
        self._training_started.fill_(True)

        if self._randomize_first_state_tau:
            # Because we called ``self._randomize_first_tau`` in rollout_step()
            # while the random ``tau`` was not stored in the replay buffer, the
            # first step's ``tau`` here is not accurate. So we need to use the
            # rollout ``tau``.
            state = self._randomize_first_tau(inputs, state,
                                              rollout_info.prev_tau)

        ap_out, new_state = self._predict_action(
            inputs, state=state, mode=Mode.train)
        # According to the TAAC formulation, each (s,prev_tau) is sampled from
        # the replay buffer instead of being generated by sequential training steps.
        # So we need to overwrite the generated tau with the rollout tau.
        new_state = new_state._replace(tau=rollout_info.tau)

        beta_dist = ap_out.dists.beta_dist
        b1_a_dist = ap_out.dists.b1_a_dist
        b0_tau, b1_tau = ap_out.taus
        q_values2 = ap_out.q_values2

        b1_a_entropy = -dist_utils.compute_log_probability(
            b1_a_dist, ap_out.actor_a)
        beta_entropy = beta_dist.entropy()

        actor_loss = self._actor_train_step(ap_out.actor_a, b1_a_entropy,
                                            beta_dist, beta_entropy, q_values2)
        critic_info = self._critic_train_step(inputs, rollout_info.tau, b0_tau,
                                              b1_tau, beta_dist)
        alpha_loss = self._alpha_train_step(beta_entropy, b1_a_entropy)

        info = TaacInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            rollout_b=rollout_info.b,
            action_distribution=ap_out.dists,
            actor=actor_loss,
            critic=critic_info,
            b=ap_out.b,
            alpha=alpha_loss,
            repeats=state.repeats)
        return AlgStep(output=new_state.tau.a, state=new_state, info=info)

    def after_update(self, root_inputs, info: TaacInfo):
        self._update_target()

    def calc_loss(self, info: TaacInfo):
        critic_loss = self._calc_critic_loss(info)
        alpha_loss = info.alpha
        actor_loss = info.actor
        if self._debug_summaries:
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha/beta", self._log_alpha[0].exp())
                alf.summary.scalar("alpha/action", self._log_alpha[1].exp())
                alf.summary.scalar("resample_advantage",
                                   torch.mean(actor_loss.extra.adv))
                p_beta0 = info.action_distribution[0].probs[..., 0]
                alf.summary.histogram("P_beta_0/value", p_beta0)
                alf.summary.scalar("P_beta_0/mean", p_beta0.mean())
                alf.summary.scalar("P_beta_0/std", p_beta0.std())
                repeats = info.repeats
                alf.summary.scalar("train_repeats/mean",
                                   torch.mean(repeats.to(torch.float32)))
                alf.summary.histogram("train_repeats/value",
                                      repeats.to(torch.float32))

        return LossInfo(
            loss=actor_loss.loss + alpha_loss + critic_loss.loss,
            extra=TaacLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss))


@alf.configurable
class TaacAlgorithm(TaacAlgorithmBase):
    r"""Model temporal abstraction by action repetition. See

        "TAAC: Temporally Abstract Actor-Critic for Continuous Control",
        Yu et al., arXiv 2021.

    for algorithm details.
    """

    def __init__(self, name="TaacAlgorithm", *args, **kwargs):
        """See ``TaacAlgorithmBase`` for argument description.
        """
        super().__init__(*args, name=name, **kwargs)

    def _make_networks(self, *args):
        tau_mask = Tau(a=True, v=False, u=False)
        args = args + (tau_mask, )
        return self._make_networks_impl(*args)

    def _update_tau(self, tau):
        """Return a constant trajectory."""
        return tau

    def _action2tau(self, a, tau):
        """Return a constant trajectory."""
        return Tau(a=a)


@alf.configurable
class TaacLAlgorithm(TaacAlgorithmBase):
    r"""TaacL: Piecewise linear trajectory policy for continuous control.

    For a linear trajectory, let :math:`a` be the action and :math:`v` the
    first derivative. Its dynamics is:

    .. math::

        \begin{array}{ll}
            v_{t+1} &\leftarrow v_t\\
            a_{t+1} &\leftarrow v_{t+1} + a_t\\
        \end{array}

    TaacL's trajectory is piece-wise linear. Each time the policy decides whether
    to repeat the previous linear traj or generate a new one. Importantly,
    to generate a new one the policy doesn't directly generate the entire set of
    two parameters :math:`(a,v)` because this will result in bad exploration
    in the action space. Instead,

    .. math::

        \begin{array}{ll}
            a_{t+1} &\sim \pi\\
            v_{t+1} &\leftarrow a_{t+1} - a_t\\
        \end{array}

    For :math:`a\in[0,1]` and :math:`v\in[0,1]`, the actual dynamics is
    :math:`a_{t+1}\leftarrow \max(\min(a_t+2v_{t+1},1),-1)`.
    """

    def __init__(self,
                 name="TaacLAlgorithm",
                 inverse_mode=True,
                 *args,
                 **kwargs):
        """See ``TaacAlgorithmBase`` for other argument description.

        Args:
            inverse_mode (bool): this argument decides how the new traj is computed when
                ``b=1``. If it's False, then the new action is treated as the
                new first derivative ``v``; otherwise the new action is treated
                as the new action ``a``, and ``v`` is inversely inferred.
        """
        super().__init__(*args, name=name, **kwargs)

        assert (
            np.all(self._action_spec.minimum == -1)
            and np.all(self._action_spec.maximum == 1)
        ), ("Only support actions in [-1, 1]! Consider using env wrappers to "
            "scale your action space first.")

        self._inverse_mode = inverse_mode

    def _make_networks(self, *args):
        tau_mask = Tau(a=True, v=True, u=False)
        args = args + (tau_mask, )
        return self._make_networks_impl(*args)

    def _update_tau(self, tau):
        """Compute next action on a linear trajectory specified by a pair of
        ('action', 'action derivative').
        """
        a = torch.clamp(tau.a + 2. * tau.v, min=-1., max=1.)
        return tau._replace(a=a)

    def _action2tau(self, a, tau):
        if self._inverse_mode:
            # Given a new action at the next step and the current traj ``tau``,
            # infer the new traj's first derivative.
            v = (a - tau.a) / 2.
            return Tau(a=a, v=v)
        else:
            # Given a new first derivative and the current traj ``tau``, compute
            # the new traj's action
            tau = Tau(a=tau.a, v=a)
            return self._update_tau(tau)


@alf.configurable
class TaacQAlgorithm(TaacLAlgorithm):
    r"""TaacQ: Piecewise quadratic trajectory policy for continuous control.

    For a quadratic trajectory, let :math:`a` be the action, :math:`u` be the
    second derivative, and :math:`v` be the first derivative. Its dynamics is:

    .. math::

        \begin{array}{ll}
            u_{t+1} &\leftarrow u_t\\
            v_{t+1} &\leftarrow u_{t+1} + v_t\\
            a_{t+1} &\leftarrow v_{t+1} + a_t\\
        \end{array}

    TaacQ's trajectory is piece-wise quadratic. Each time the policy decides whether
    to repeat the previous quadratic traj or generate a new one. Importantly,
    to generate a new one the policy doesn't directly generate the entire set of
    three parameters :math:`(a,v,u)` because this will result in bad exploration
    in the action space. Instead,

    .. math::

        \begin{array}{ll}
            a_{t+1} &\sim \pi\\
            v_{t+1} &\leftarrow a_{t+1} - a_t\\
            u_{t+1} &\leftarrow v_{t+1}\\
        \end{array}

    where the last two steps assume resetting :math:`v_t` to zero.

    For :math:`a\in[0,1]`, :math:`v\in[0,1]`, and :math:`u\in[0,1]`, the actual
    dynamics is :math:`v_{t+1}\leftarrow \max(\min(v_t+2u_{t+1},1),-1)` and
    :math:`a_{t+1}\leftarrow \max(\min(a_t+2v_{t+1},1),-1)`.
    """

    def __init__(self,
                 name="TaacQAlgorithm",
                 inverse_mode=True,
                 *args,
                 **kwargs):
        """See ``TaacAlgorithmBase`` for other argument description.

        Args:
            inverse_mode (bool): this argument decides how the new traj is computed
                when ``b=1``. If it's False, then the new action is treated as the
                new second derivative ``u``; otherwise the new action is treated
                as the new action ``a``, and ``u`` is inversely inferred. In either
                case, the current ``v`` is first set to 0, and then a new ``v`` is
                computed.
        """
        super().__init__(*args, name=name, inverse_mode=inverse_mode, **kwargs)

    def _make_networks(self, *args):
        tau_mask = Tau(a=True, v=True, u=True)
        args = args + (tau_mask, )
        return self._make_networks_impl(*args)

    def _update_tau(self, tau):
        """Compute next action on a quadratic trajectory specified by a triplet
        of ('action', 'action derivative', and 'action second derivative').
        """
        v = torch.clamp(tau.v + tau.u * 2., min=-1., max=1.)
        a = torch.clamp(tau.a + v * 2., min=-1., max=1.)
        return Tau(a=a, v=v, u=tau.u)

    def _action2tau(self, a, tau):
        if self._inverse_mode:
            # Given a new action at the next step and the current traj ``tau``,
            # infer the new traj, assuming resetting ``tau.v`` to 0 first.
            v = (a - tau.a) / 2.
            u = v / 2.
            return Tau(a=a, v=v, u=u)
        else:
            # Given a new second derivative at the next step and the current traj
            # ``tau``, compute the new traj, assuming resetting ``tau.v`` to 0 first.
            tau = Tau(a=tau.a, v=0, u=a)
            return self._update_tau(tau)
