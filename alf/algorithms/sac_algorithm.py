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

import numpy as np
import gin
import functools

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Callable

from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import TimeStep, Experience, LossInfo, namedtuple
from alf.data_structures import AlgStep, TrainingInfo
from alf.nest import nest
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils

SacShareState = namedtuple("SacShareState", ["actor"])

SacActorState = namedtuple("SacActorState", ["critic1", "critic2"])

SacCriticState = namedtuple(
    "SacCriticState",
    ["critic1", "critic2", "target_critic1", "target_critic2"])

SacState = namedtuple("SacState", ["share", "actor", "critic"])

SacActorInfo = namedtuple("SacActorInfo", ["loss"])

SacCriticInfo = namedtuple("SacCriticInfo",
                           ["critic1", "critic2", "target_critic"])

SacAlphaInfo = namedtuple("SacAlphaInfo", ["loss"])

SacInfo = namedtuple(
    "SacInfo", ["action_distribution", "actor", "critic", "alpha"],
    default_value=())

SacLossInfo = namedtuple('SacLossInfo', ('actor', 'critic', 'alpha'))


@gin.configurable
class SacAlgorithm(OffPolicyAlgorithm):
    """Soft Actor Critic

    It's described in:
    Haarnoja et al "Soft Actor-Critic Algorithms and Applications" arXiv:1812.05905v2

    There are 3 points different with `tf_agents.agents.sac.sac_agent`:

    1. To reduce computation, here we sample actions only once for calculating
    actor, critic, and alpha loss while `tf_agents.agents.sac.sac_agent` samples
    actions for each loss. This difference has little influence on the training
    performance.

    2. We calculate losses for every sampled steps.
    (s_t, a_t), (s_{t+1}, a_{t+1}) in sampled transition are used to calculate
    actor, critic and alpha loss while `tf_agents.agents.sac.sac_agent` only
    uses (s_t, a_t) and critic loss for s_{t+1} is 0. You should handle this
    carefully, it is equivalent to applying a coefficient of 0.5 on the critic
    loss.

    3. We mask out `StepType.LAST` steps when calculating losses but
    `tf_agents.agents.sac.sac_agent` does not. We believe the correct
    implementation should mask out LAST steps. And this may make different
    performance on same tasks.
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 actor_network: ActorDistributionNetwork,
                 critic_network: CriticNetwork,
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
                 gradient_clipping=None,
                 clip_by_global_norm=False,
                 debug_summaries=False,
                 name="SacAlgorithm"):
        """Create a SacAlgorithm

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            actor_network (Network): The network will be called with
                call(observation).
            critic_network (Network): The network will be called with
                call(observation, action).
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs `train_iter()` by
                itself.
            critic_loss_ctor (None|OneStepTDLoss): a critic loss constructor.
                If None, a default OneStepTDLoss will be used.
            initial_log_alpha (float): initial value for variable log_alpha
            target_entropy (float|None): The target average policy entropy, for updating alpha.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between [-dqda_clipping, dqda_clipping].
                Does not perform clipping if dqda_clipping == 0.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            alpha_optimizer (torch.optim.optimizer): The optimizer for alpha.
            gradient_clipping (float): Norm length to clip gradients.
            clip_by_global_norm (bool): If True, use `tensor_utils.clip_by_global_norm`
                to clip gradient. If False, use `tensor_utils.clip_by_norms` for
                each grad.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        critic_network1 = critic_network.copy()
        critic_network2 = critic_network.copy()

        log_alpha = nn.Parameter(torch.Tensor([float(initial_log_alpha)]))

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=SacState(
                share=SacShareState(actor=actor_network.state_spec),
                actor=SacActorState(
                    critic1=critic_network.state_spec,
                    critic2=critic_network.state_spec),
                critic=SacCriticState(
                    critic1=critic_network.state_spec,
                    critic2=critic_network.state_spec,
                    target_critic1=critic_network.state_spec,
                    target_critic2=critic_network.state_spec)),
            env=env,
            config=config,
            gradient_clipping=gradient_clipping,
            clip_by_global_norm=clip_by_global_norm,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer,
                               [critic_network1, critic_network2])
        if alpha_optimizer is not None:
            self.add_optimizer(alpha_optimizer, [log_alpha])

        self._log_alpha = log_alpha
        self._actor_network = actor_network
        self._critic_network1 = critic_network1
        self._critic_network2 = critic_network2
        self._target_critic_network1 = self._critic_network1.copy()
        self._target_critic_network2 = self._critic_network2.copy()

        if critic_loss_ctor is None:
            critic_loss_ctor = functools.partial(
                OneStepTDLoss, debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_loss1 = critic_loss_ctor(name="critic_loss1")
        self._critic_loss2 = critic_loss_ctor(name="critic_loss2")

        flat_action_spec = nest.flatten(self._action_spec)
        self._flat_action_spec = flat_action_spec

        self._is_continuous = flat_action_spec[0].is_continuous
        if target_entropy is None:
            target_entropy = np.sum(
                list(
                    map(dist_utils.calc_default_target_entropy,
                        flat_action_spec)))
        self._target_entropy = target_entropy
        self._dqda_clipping = dqda_clipping

        self._update_target = common.get_target_updater(
            models=[self._critic_network1, self._critic_network2],
            target_models=[
                self._target_critic_network1, self._target_critic_network2
            ],
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

            critic1, critic1_state = self._critic_network1(
                critic_input, state=state.critic1)
            critic2, critic2_state = self._critic_network2(
                critic_input, state=state.critic2)

            target_q_value = torch.min(critic1, critic2)
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
            critic1, critic1_state = self._critic_network1(
                exp.observation, state=state.critic1)

            critic2, critic2_state = self._critic_network2(
                exp.observation, state=state.critic2)

            base_action_dist = dist_utils.get_base_dist(action_distribution)
            assert isinstance(base_action_dist, td.categorical.Categorical),  \
                 ("Only `Categorical` " + "was supported, received:" + str(
                        type(base_action_dist)))

            log_action_probs = base_action_dist.logits.squeeze(1)

            target_q_value = torch.min(critic1, critic2).detach()
            alpha = torch.exp(self._log_alpha)
            actor_loss = torch.exp(log_action_probs) * (
                alpha.detach() * log_action_probs - target_q_value)
            actor_loss = actor_loss.mean(list(range(1, actor_loss.ndim)))

        state = SacActorState(critic1=critic1_state, critic2=critic2_state)
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

        critic1, critic1_state = self._critic_network1(
            critic_input, state=state.critic1)

        critic2, critic2_state = self._critic_network2(
            critic_input, state=state.critic2)

        target_critic1, target_critic1_state = self._target_critic_network1(
            target_critic_input, state=state.target_critic1)

        target_critic2, target_critic2_state = self._target_critic_network2(
            target_critic_input, state=state.target_critic2)

        if not self._is_continuous:
            exp_action = exp.action.view(critic1.shape[0], -1).long()
            critic1 = critic1.gather(-1, exp_action)
            critic2 = critic2.gather(-1, exp_action)
            sampled_action = action.view(critic1.shape[0], -1).long()
            target_critic1 = target_critic1.gather(-1, sampled_action)
            target_critic2 = target_critic2.gather(-1, sampled_action)

        target_critic = torch.min(target_critic1, \
                                  target_critic2).reshape(log_pi.shape) - \
                         (torch.exp(self._log_alpha) * log_pi).detach()

        critic1 = critic1.squeeze(-1)
        critic2 = critic2.squeeze(-1)
        target_critic = target_critic.squeeze(-1).detach()

        state = SacCriticState(
            critic1=critic1_state,
            critic2=critic2_state,
            target_critic1=target_critic1_state,
            target_critic2=target_critic2_state)

        info = SacCriticInfo(
            critic1=critic1, critic2=critic2, target_critic=target_critic)

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

    def after_update(self, training_info):
        self._update_target()

    def calc_loss(self, training_info: TrainingInfo):
        critic_loss = self._calc_critic_loss(training_info)
        alpha_loss = training_info.info.alpha.loss
        actor_loss = training_info.info.actor.loss
        return LossInfo(
            loss=actor_loss.loss + critic_loss.loss + alpha_loss.loss,
            extra=SacLossInfo(
                actor=actor_loss.extra,
                critic=critic_loss.extra,
                alpha=alpha_loss.extra))

    def _calc_critic_loss(self, training_info):
        critic_info = training_info.info.critic

        target_critic = critic_info.target_critic

        critic_loss1 = self._critic_loss1(
            training_info=training_info,
            value=critic_info.critic1,
            target_value=target_critic)

        critic_loss2 = self._critic_loss2(
            training_info=training_info,
            value=critic_info.critic2,
            target_value=target_critic)

        critic_loss = critic_loss1.loss + critic_loss2.loss
        return LossInfo(loss=critic_loss, extra=critic_loss)

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_network1', '_target_critic_network2']
