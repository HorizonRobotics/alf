# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
"""Soft Actor Critic Algorithm."""

from absl import logging
import numpy as np
import gin
import functools
from enum import Enum

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import QNetwork, QRNNNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops

ActionType = Enum('ActionType', ('Discrete', 'Continuous', 'Mixed'))

SacActionState = namedtuple(
    "SacActionState", ["actor_network", "critic"], default_value=())

SacCriticState = namedtuple("SacCriticState", ["critics", "target_critics"])

SacState = namedtuple("SacState", ["action", "actor", "critic"])

SacCriticInfo = namedtuple("SacCriticInfo", ["critics", "target_critic"])

SacActorInfo = namedtuple(
    "SacActorInfo", ["actor_loss", "neg_entropy"], default_value=())

SacInfo = namedtuple(
    "SacInfo", ["action_distribution", "actor", "critic", "alpha"],
    default_value=())

SacLossInfo = namedtuple('SacLossInfo', ('actor', 'critic', 'alpha'))


def _set_target_entropy(name, target_entropy, flat_action_spec):
    """A helper function for computing the target entropy under different
    scenarios of ``target_entropy``.

    Args:
        name (str): the name of the algorithm that calls this function.
        target_entropy (float|Callable|None): If a floating value, it will return
            as it is. If a callable function, then it will be called on the action
            spec to calculate a target entropy. If ``None``, a default entropy will
            be calculated.
        flat_action_spec (list[TensorSpec]): a flattened list of action specs.
    """
    if target_entropy is None or callable(target_entropy):
        if target_entropy is None:
            target_entropy = dist_utils.calc_default_target_entropy
        target_entropy = np.sum(list(map(target_entropy, flat_action_spec)))
        logging.info("Target entropy is calculated for {}: {}.".format(
            name, target_entropy))
    else:
        logging.info("User-supplied target entropy for {}: {}".format(
            name, target_entropy))
    return target_entropy


@gin.configurable
class SacAlgorithm(OffPolicyAlgorithm):
    r"""Soft Actor Critic algorithm, described in:

    ::

        Haarnoja et al "Soft Actor-Critic Algorithms and Applications", arXiv:1812.05905v2

    There are 3 points different with ``tf_agents.agents.sac.sac_agent``:

    1. To reduce computation, here we sample actions only once for calculating
    actor, critic, and alpha loss while ``tf_agents.agents.sac.sac_agent``
    samples actions for each loss. This difference has little influence on
    the training performance.

    2. We calculate losses for every sampled steps.
    :math:`(s_t, a_t), (s_{t+1}, a_{t+1})` in sampled transition are used
    to calculate actor, critic and alpha loss while
    ``tf_agents.agents.sac.sac_agent`` only uses :math:`(s_t, a_t)` and
    critic loss for :math:`s_{t+1}` is 0. You should handle this carefully,
    it is equivalent to applying a coefficient of 0.5 on the critic loss.

    3. We mask out ``StepType.LAST`` steps when calculating losses but
    ``tf_agents.agents.sac.sac_agent`` does not. We believe the correct
    implementation should mask out ``LAST`` steps. And this may make different
    performance on same tasks.

    In addition to continuous actions addressed by the original paper, this
    algorithm also supports discrete actions and a mixture of discrete and
    continuous actions. The networks for computing Q values :math:`Q(s,a)` and
    sampling acitons can be divided into 3 cases according to action types:

    1. Discrete only: a ``QNetwork`` is used for estimating Q values. There will
       be no actor network to learn because actions can be directly sampled from
       the Q values: :math:`p(a|s) \propto \exp(\frac{Q(s,a)}{\alpha})`.
    2. Continuous only: a ``CriticNetwork`` is used for estimating Q values. An
       ``ActorDistributionNetwork`` for sampling actions will be learned according
       to Q values.
    3. Mixed: a ``QNetwork`` is used for estimating Q values. The input of this
       particular ``QNetwork`` (dubbed as "Universal Q Network") is augmented
       with all continuous actions as ``(observation, continuous_action)``,
       while the output heads correspond to discrete actions. So a Q value
       :math:`Q(s, a_{cont}, a_{disc}=k)` is estimated by the :math:`k`-th output
       head of the network given :math:`a_{cont}` as the augmented input to
       :math:`s`. Still only an ``ActorDistributionNetwork`` is needed for first
       sampling continuous actions, and then a discrete action is sampled from Q
       values conditioned on the continuous actions.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 actor_network: ActorDistributionNetwork = None,
                 critic_network: CriticNetwork = None,
                 q_network: QNetwork = None,
                 use_parallel_network=False,
                 num_critic_replicas=2,
                 env=None,
                 observation_transformer=math_ops.identity,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_entropy=None,
                 initial_log_alpha=0.0,
                 target_update_tau=0.05,
                 target_update_period=1,
                 dqda_clipping=None,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 alpha_optimizer=None,
                 debug_summaries=False,
                 name="SacAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                be a mixture of discrete and continuous actions. The number of
                continuous actions can be arbitrary while only one discrete
                action is allowed currently. If it's a mixture, then it must be
                a tuple/list ``(discrete_action_spec, continuous_action_spec)``.
            actor_network (ActorDistributionNetwork): The network will be called
                to sample continuous actions. All of its output specs must be
                continuous. Note that we don't need a discrete actor network
                because a discrete action can simply be sampled from the Q values.
            critic_network (CriticNetwork): used for estimating ``Q(s,a)`` given
                that the action is continuous. Its input action spec must be
                consistent with the continuous actions in ``action_spec``.
            q_network (QNetwork): used for estimating ``Q(s,a)`` given that the
                action is discrete. Its output spec must be consistent with
                the discrete action in ``action_spec``.
            use_parallel_network (bool): whether to use parallel network for
                calculating critics.
            num_critic_replicas (int): number of critics to be used. Default is 2.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            observation_transformer (Callable or list[Callable]): transformation(s)
                applied to ``time_step.observation``.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            initial_log_alpha (float): initial value for variable ``log_alpha``.
            target_entropy (float|Callable|None): If a floating value, it's the
                target average policy entropy, for updating ``alpha``. If a
                callable function, then it will be called on the action spec to
                calculate a target entropy. If ``None``, a default entropy will
                be calculated.
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
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        self._num_critic_replicas = num_critic_replicas
        self._use_parallel_network = use_parallel_network

        critic_networks, self._act_type = self._make_critic_networks(
            observation_spec, action_spec, actor_network, critic_network,
            q_network)

        log_alpha = nn.Parameter(torch.Tensor([float(initial_log_alpha)]))

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=SacState(
                action=SacActionState(
                    actor_network=(() if self._act_type == ActionType.Discrete
                                   else actor_network.state_spec),
                    critic=(() if self._act_type == ActionType.Continuous else
                            critic_networks.state_spec)),
                actor=(() if self._act_type != ActionType.Continuous else
                       critic_networks.state_spec),
                critic=SacCriticState(
                    critics=critic_networks.state_spec,
                    target_critics=critic_networks.state_spec)),
            observation_transformer=observation_transformer,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, [log_alpha])

        self._log_alpha = log_alpha
        self._actor_network = actor_network
        self._critic_networks = critic_networks
        self._target_critic_networks = self._critic_networks.copy(
            name='target_critic_networks')

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))

        self._target_entropy = _set_target_entropy(
            self.name, target_entropy, nest.flatten(self._action_spec))

        self._dqda_clipping = dqda_clipping

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

    def _make_critic_networks(self, observation_spec, action_spec,
                              continuous_actor_network, critic_network,
                              q_network):
        def _make_parallel(net):
            if self._use_parallel_network:
                nets = net.make_parallel(self._num_critic_replicas)
            else:
                nets = alf.networks.NaiveParallelNetwork(
                    net, self._num_critic_replicas)
            return nets

        def _check_spec_equal(spec1, spec2):
            assert nest.flatten(spec1) == nest.flatten(spec2), (
                "Unmatched action specs: {} vs. {}".format(spec1, spec2))

        discrete_action_spec = [
            spec for spec in nest.flatten(action_spec) if spec.is_discrete
        ]
        continuous_action_spec = [
            spec for spec in nest.flatten(action_spec) if spec.is_continuous
        ]

        if discrete_action_spec and continuous_action_spec:
            # When there are both continuous and discrete actions, we require
            # that acition_spec is a tuple/list ``(discrete, continuous)``.
            assert (isinstance(
                action_spec, (tuple, list)) and len(action_spec) == 2), (
                    "In the mixed case, the action spec must be a tuple/list"
                    " (discrete_action_spec, continuous_action_spec)!")
            _check_spec_equal(action_spec[0], discrete_action_spec)
            _check_spec_equal(action_spec[1], continuous_action_spec)

        if continuous_action_spec:
            assert continuous_actor_network is not None, (
                "If there are continuous actions, then a ActorDistributionNetwork "
                "must be provided for sampling continuous actions!")
            _check_spec_equal(continuous_action_spec,
                              continuous_actor_network._action_spec)
            if not discrete_action_spec:
                act_type = ActionType.Continuous
                assert critic_network is not None, (
                    "If only continuous actions exist, then a CriticNetwork must"
                    " be provided!")
                if q_network is not None:
                    common.warning_once(
                        "The provided QNetwork will be ignored.")
                _check_spec_equal(continuous_action_spec,
                                  critic_network.input_tensor_spec[1])
                critic_networks = _make_parallel(critic_network)

        if discrete_action_spec:
            act_type = ActionType.Discrete
            assert len(discrete_action_spec) == 1, (
                "Only support at most one discrete action currently! "
                "Discrete action spec: {}".format(discrete_action_spec))
            assert q_network is not None, (
                "If there exists a discrete action, then QNetwork must "
                "be provided!")
            if critic_network is not None:
                common.warning_once(
                    "The provided CriticNetwork will be ignored.")
            if continuous_action_spec:
                act_type = ActionType.Mixed
                assert (
                    isinstance(q_network.input_tensor_spec, tuple)
                    and len(q_network.input_tensor_spec) == 2
                ), ("In the mixed case, the input spec of QNetwork must be "
                    "a tuple (observation_spec, continuous_action_spec)!")
                _check_spec_equal(q_network.input_tensor_spec[0],
                                  observation_spec)
                _check_spec_equal(q_network.input_tensor_spec[1],
                                  continuous_action_spec)
            critic_networks = _make_parallel(q_network)

        return critic_networks, act_type

    def _predict_action(self,
                        observation,
                        state: SacActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False):
        """The reason why we want to do action sampling inside this function
        instead of outside is that for the mixed case, once a continuous action
        is sampled here, we should pair it with the discrete action sampled from
        the Q value. If we just return two distributions and sample outside, then
        the actions will not match.
        """
        new_state = SacActionState()
        if self._act_type != ActionType.Discrete:
            continuous_action_dist, actor_network_state = self._actor_network(
                observation, state=state.actor_network)
            new_state._replace(actor_network=actor_network_state)
            if eps_greedy_sampling:
                continuous_action = dist_utils.epsilon_greedy_sample(
                    continuous_action_dist, epsilon_greedy)
            else:
                continuous_action = dist_utils.rsample_action_distribution(
                    continuous_action_dist)

        critic_network_inputs = observation
        if self._act_type == ActionType.Mixed:
            critic_network_inputs = (observation, continuous_action)

        q_values = None
        if self._act_type != ActionType.Continuous:
            q_values, critic_state = self._critic_networks(
                critic_network_inputs, state=state.critic)
            new_state._replace(critic=critic_state)
            alpha = torch.exp(self._log_alpha)
            # p(a|s) = exp(Q(s,a)/alpha) / Z;
            # the discrete distribution will never be trained, so we detach the
            # logits here.
            logits = (q_values.min(dim=1)[0] / alpha[0]).detach()
            discrete_action_dist = td.Categorical(logits=logits)
            if eps_greedy_sampling:
                discrete_action = dist_utils.epsilon_greedy_sample(
                    discrete_action_dist, epsilon_greedy)
            else:
                discrete_action = dist_utils.sample_action_distribution(
                    discrete_action_dist)

        if self._act_type == ActionType.Mixed:
            # Note that in this case ``action_dist`` is not the valid joint
            # action distribution because ``discrete_action_dist`` is conditioned
            # on a particular continuous action sampled above. So DO NOT use this
            # ``action_dist`` to directly sample an action pair anywhere else!
            # However, for computing the log probability of *this* sampled
            # ``action``, it's still valid. It can also be used for summary
            # purpose because of the expectation taken over the continuous action
            # when summarizing.
            action_dist = type(self._action_spec)((discrete_action_dist,
                                                   continuous_action_dist))
            action = type(self._action_spec)((discrete_action,
                                              continuous_action))
        elif self._act_type == ActionType.Discrete:
            action_dist = discrete_action_dist
            action = discrete_action
        else:
            action_dist = continuous_action_dist
            action = continuous_action

        return action_dist, action, q_values, new_state

    def predict_step(self,
                     time_step: TimeStep,
                     state: SacState,
                     epsilon_greedy=1.0):
        action_dist, action, _, action_state = self._predict_action(
            time_step.observation,
            state=state.action,
            epsilon_greedy=epsilon_greedy,
            eps_greedy_sampling=True)
        empty_state = nest.map_structure(lambda x: (), self.train_state_spec)
        new_state = empty_state._replace(action=action_state)
        return AlgStep(
            output=action,
            state=new_state,
            info=SacInfo(action_distribution=action_dist))

    def rollout_step(self, time_step: TimeStep, state: SacState):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by SacAlgorithm")
        return self.predict_step(time_step, state, epsilon_greedy=1.0)

    def _compute_critics(self, critic_net, observation, action, critics_state):
        if self._act_type == ActionType.Continuous:
            observation = (observation, action)
        elif self._act_type == ActionType.Mixed:
            observation = (observation, action[1])  # continuous action
        # discrete/mixed: critics shape [B, replicas, num_actions]
        # continuous: critics shape [B, replicas]
        critics, critics_state = critic_net(observation, state=critics_state)
        return critics, critics_state

    def _actor_train_step(self, exp: Experience, state, action, critics,
                          log_pi):
        if self._act_type == ActionType.Discrete:
            # Pure discrete case doesn't need to learn an actor network
            return (), LossInfo(extra=SacActorInfo(neg_entropy=log_pi))

        if self._act_type == ActionType.Continuous:
            critics, critics_state = self._compute_critics(
                self._critic_networks, exp.observation, action, state)
        else:
            # use the provided critics for discrete/mixed
            critics_state = ()

        if self._act_type == ActionType.Mixed:
            # pick values according to discrete actions
            critics = self._select_q_value(action[0], critics)
            action = action[1]

        target_q_value = critics.min(dim=1)[0]

        dqda = nest_utils.grad(action, target_q_value.sum())

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))
        alpha = torch.exp(self._log_alpha).detach()
        actor_info = LossInfo(
            loss=actor_loss + alpha * log_pi,
            extra=SacActorInfo(actor_loss=actor_loss, neg_entropy=log_pi))
        return critics_state, actor_info

    def _select_q_value(self, action, q_values):
        """Use ``action`` to index and select Q values.
        Args:
            action: discrete actions with shape ``[batch_size]``.
            critics: Q values with shape ``[batch_size, replicas, num_actions]``.
        Returns:
            selected Q values with shape ``[batch_size, replicas]``.
        """
        # action shape: [batch_size] -> [batch_size, n, 1]
        action = action.view(q_values.shape[0], 1, -1).expand(
            -1, q_values.shape[1], -1).long()
        return q_values.gather(-1, action).squeeze(-1)

    def _critic_train_step(self, exp: Experience, state: SacCriticState,
                           action, log_pi):
        critics, critics_state = self._compute_critics(
            self._critic_networks, exp.observation, exp.action, state.critics)

        target_critics, target_critics_state = self._compute_critics(
            self._target_critic_networks, exp.observation, action,
            state.target_critics)

        if self._act_type == ActionType.Discrete:
            critics = self._select_q_value(exp.action, critics)
            target_critics = self._select_q_value(action, target_critics)
        elif self._act_type == ActionType.Mixed:
            critics = self._select_q_value(exp.action[0], critics)
            target_critics = self._select_q_value(action[0], target_critics)

        target_critic = target_critics.min(dim=1)[0].reshape(log_pi.shape) - \
                         (torch.exp(self._log_alpha) * log_pi)

        target_critic = target_critic.detach()

        state = SacCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = SacCriticInfo(critics=critics, target_critic=target_critic)

        return state, info

    def _alpha_train_step(self, log_pi):
        alpha_loss = self._log_alpha * (
            -log_pi - self._target_entropy).detach()
        return alpha_loss

    def train_step(self, exp: Experience, state: SacState):
        (action_distribution, action, critics,
         action_state) = self._predict_action(
             exp.observation, state=state.action)

        log_pi = dist_utils.compute_log_probability(action_distribution,
                                                    action)

        actor_state, actor_loss = self._actor_train_step(
            exp, state.actor, action, critics, log_pi)
        critic_state, critic_info = self._critic_train_step(
            exp, state.critic, action, log_pi)
        alpha_loss = self._alpha_train_step(log_pi)

        state = SacState(
            action=action_state, actor=actor_state, critic=critic_state)
        info = SacInfo(
            action_distribution=action_distribution,
            actor=actor_loss,
            critic=critic_info,
            alpha=alpha_loss)
        return AlgStep(action, state, info)

    def after_update(self, experience, train_info: SacInfo):
        self._update_target()

    def calc_loss(self, experience, train_info: SacInfo):
        critic_loss = self._calc_critic_loss(experience, train_info)
        alpha_loss = train_info.alpha
        actor_loss = train_info.actor

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha", self._log_alpha.exp())

        return LossInfo(
            loss=math_ops.add_ignore_empty(actor_loss.loss,
                                           critic_loss.loss + alpha_loss),
            extra=SacLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss))

    def _calc_critic_loss(self, experience, train_info: SacInfo):
        critic_info = train_info.critic

        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            critic_losses.append(
                l(experience=experience,
                  value=critic_info.critics[..., i],
                  target_value=critic_info.target_critic).loss)

        critic_loss = math_ops.add_n(critic_losses)

        if (experience.batch_info != ()
                and experience.batch_info.importance_weights != ()):
            valid_masks = (experience.step_type != StepType.LAST).to(
                torch.float32)
            priority = (critic_loss * valid_masks).sum(dim=0).sqrt()
        else:
            priority = ()

        return LossInfo(
            loss=critic_loss,
            priority=priority,
            extra=critic_loss / float(self._num_critic_replicas))

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']
