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
from alf.data_structures import AlgStep
from alf.nest import nest
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops

SacShareState = namedtuple("SacShareState", ["actor"])

SacActorState = namedtuple("SacActorState", ["critics"])

SacCriticState = namedtuple("SacCriticState", ["critics", "target_critics"])

SacState = namedtuple("SacState", ["share", "actor", "critic"])

SacActorInfo = namedtuple("SacActorInfo", ["loss"])

SacCriticInfo = namedtuple("SacCriticInfo", ["critics", "target_critic"])

SacAlphaInfo = namedtuple("SacAlphaInfo", ["loss"])

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
    """Soft Actor Critic algorithm, described in:

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
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 actor_network: ActorDistributionNetwork,
                 critic_network: CriticNetwork,
                 use_parallel_network=False,
                 num_critic_replicas=2,
                 env=None,
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
            action_spec (nested BoundedTensorSpec): representing the actions.
            actor_network (Network): The network will be called with
                ``call(observation)``.
            critic_network (Network): This network can be either a ``CriticNetwork``
                or a ``QNetwork``.
            use_parallel_network (bool): whether to use parallel network for
                calculating critics.
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
        if use_parallel_network:
            critic_networks = critic_network.make_parallel(num_critic_replicas)
        else:
            critic_networks = alf.networks.NaiveParallelNetwork(
                critic_network, num_critic_replicas)

        log_alpha = nn.Parameter(torch.Tensor([float(initial_log_alpha)]))

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=SacState(
                share=SacShareState(actor=actor_network.state_spec),
                actor=SacActorState(critics=critic_networks.state_spec),
                critic=SacCriticState(
                    critics=critic_networks.state_spec,
                    target_critics=critic_networks.state_spec)),
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
        self._num_critic_replicas = num_critic_replicas
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

        flat_action_spec = nest.flatten(self._action_spec)
        self._flat_action_spec = flat_action_spec

        self._is_continuous = flat_action_spec[0].is_continuous
        self._target_entropy = _set_target_entropy(self.name, target_entropy,
                                                   flat_action_spec)
        self._dqda_clipping = dqda_clipping

        self._update_target = common.get_target_updater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

    def _predict(self, time_step: TimeStep, state=None, epsilon_greedy=1.):
        action_dist, state = self._actor_network(
            time_step.observation, state=state.share.actor)
        empty_state = nest.map_structure(lambda x: (), self.train_state_spec)
        state = empty_state._replace(share=SacShareState(actor=state))
        action = dist_utils.epsilon_greedy_sample(action_dist, epsilon_greedy)
        return AlgStep(
            output=action,
            state=state,
            info=SacInfo(action_distribution=action_dist))

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        return self._predict(time_step, state, epsilon_greedy)

    def rollout_step(self, time_step: TimeStep, state):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by SacAlgorithm")
        return self._predict(time_step, state, epsilon_greedy=1.0)

    def _actor_train_step(self, exp: Experience, state: SacActorState,
                          action_distribution, action, log_pi):

        if self._is_continuous:
            critic_input = (exp.observation, action)

            critics, critics_state = self._critic_networks(
                critic_input, state=state.critics)

            target_q_value = critics.min(dim=1)[0]
            dqda = nest.pack_sequence_as(
                action,
                list(
                    torch.autograd.grad(
                        target_q_value.sum(),
                        nest.flatten(action),
                    )))

            def actor_loss_fn(dqda, action):
                if self._dqda_clipping:
                    dqda = torch.clamp(dqda, -self._dqda_clipping,
                                       self._dqda_clipping)
                loss = 0.5 * losses.element_wise_squared_loss(
                    (dqda + action).detach(), action)
                loss = loss.sum(list(range(1, loss.ndim)))
                return loss

            actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
            alpha = torch.exp(self._log_alpha).detach()
            actor_loss += alpha * log_pi
        else:
            critics, critics_state = self._critic_networks(
                exp.observation, state=state.critics)

            base_action_dist = dist_utils.get_base_dist(action_distribution)
            assert isinstance(base_action_dist, td.categorical.Categorical),  \
                 ("Only `Categorical` " + "was supported, received:" + str(
                        type(base_action_dist)))

            log_action_probs = base_action_dist.logits.squeeze(1)

            target_q_value = critics.min(dim=1)[0].detach()
            alpha = torch.exp(self._log_alpha)
            actor_loss = torch.exp(log_action_probs) * (
                alpha.detach() * log_action_probs - target_q_value)
            actor_loss = actor_loss.mean(list(range(1, actor_loss.ndim)))

        state = SacActorState(critics=critics_state)
        info = SacActorInfo(loss=LossInfo(loss=actor_loss, extra=actor_loss))

        return state, info

    def _critic_train_step(self, exp: Experience, state: SacCriticState,
                           action, log_pi):
        if self._is_continuous:
            critic_input = (exp.observation, exp.action)
            target_critic_input = (exp.observation, action)
        else:
            critic_input = exp.observation
            target_critic_input = exp.observation

        critics, critics_state = self._critic_networks(
            critic_input, state=state.critics)

        target_critics, target_critics_state = self._target_critic_networks(
            target_critic_input, state=state.target_critics)

        if not self._is_continuous:
            # action shape: [batch_size] -> [batch_size, n, 1]
            exp_action = exp.action.view(critics.shape[0], 1, -1).expand(
                -1, critics.shape[1], -1).long()
            critics = critics.gather(-1, exp_action)

            sampled_action = action.view(critics.shape[0], 1, -1).expand(
                -1, critics.shape[1], -1).long()
            target_critics = target_critics.gather(-1, sampled_action)

        target_critic = target_critics.min(dim=1)[0].reshape(log_pi.shape) - \
                         (torch.exp(self._log_alpha) * log_pi)

        critics = critics.squeeze(-1)
        target_critic = target_critic.squeeze(-1).detach()

        state = SacCriticState(
            critics=critics_state, target_critics=target_critics_state)

        info = SacCriticInfo(critics=critics, target_critic=target_critic)

        return state, info

    def _alpha_train_step(self, log_pi):
        alpha_loss = self._log_alpha * (
            -log_pi - self._target_entropy).detach()
        info = SacAlphaInfo(loss=LossInfo(loss=alpha_loss, extra=alpha_loss))
        return info

    def train_step(self, exp: Experience, state: SacState):
        action_distribution, share_actor_state = self._actor_network(
            exp.observation, state=state.share.actor)
        if self._is_continuous:
            action = dist_utils.rsample_action_distribution(
                action_distribution)
        else:
            action = dist_utils.sample_action_distribution(action_distribution)

        log_pi = dist_utils.compute_log_probability(action_distribution,
                                                    action)
        actor_state, actor_info = self._actor_train_step(
            exp, state.actor, action_distribution, action, log_pi)
        critic_state, critic_info = self._critic_train_step(
            exp, state.critic, action, log_pi)
        alpha_info = self._alpha_train_step(log_pi)

        state = SacState(
            share=SacShareState(actor=share_actor_state),
            actor=actor_state,
            critic=critic_state)
        info = SacInfo(
            action_distribution=action_distribution,
            actor=actor_info,
            critic=critic_info,
            alpha=alpha_info)
        return AlgStep(action, state, info)

    def after_update(self, experience, train_info: SacInfo):
        self._update_target()

    def calc_loss(self, experience, train_info: SacInfo):
        critic_loss = self._calc_critic_loss(experience, train_info)
        alpha_loss = train_info.alpha.loss
        actor_loss = train_info.actor.loss

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("alpha", self._log_alpha.exp())

        return LossInfo(
            loss=actor_loss.loss + critic_loss.loss + alpha_loss.loss,
            extra=SacLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss.extra))

    def _calc_critic_loss(self, experience, train_info: SacInfo):
        critic_info = train_info.critic

        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            critic_losses.append(
                l(experience=experience,
                  value=critic_info.critics[..., i],
                  target_value=critic_info.target_critic).loss)

        critic_loss = math_ops.add_n(critic_losses)
        return LossInfo(
            loss=critic_loss,
            extra=critic_loss / float(self._num_critic_replicas))

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']
