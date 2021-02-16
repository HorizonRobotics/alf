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
"""Deep Deterministic Policy Gradient (DDPG)."""

import functools
import gin
import numpy as np

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
from alf.networks import ActorNetwork, CriticNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import losses, common, dist_utils, math_ops, spec_utils

DdpgCriticState = namedtuple(
    "DdpgCriticState", [
        'critics', 'non_her_critic', 'target_actor', 'target_critics',
        'non_her_target'
    ],
    default_value=())
DdpgCriticInfo = namedtuple(
    "DdpgCriticInfo", [
        "q_values", "non_her_q_values", "goal_values", "target_q_values",
        "non_her_target", "target_goal_values"
    ],
    default_value=())
DdpgActorState = namedtuple("DdpgActorState", ['actor', 'critics'])
DdpgState = namedtuple("DdpgState", ['actor', 'critics'])
DdpgInfo = namedtuple(
    "DdpgInfo", ["action_distribution", "actor_loss", "critic"],
    default_value=())
DdpgLossInfo = namedtuple('DdpgLossInfo', ('actor', 'critic'))


@gin.configurable
class DdpgAlgorithm(OffPolicyAlgorithm):
    """Deep Deterministic Policy Gradient (DDPG).

    Reference:
    Lillicrap et al "Continuous control with deep reinforcement learning"
    https://arxiv.org/abs/1509.02971
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 actor_network_ctor=ActorNetwork,
                 critic_network_ctor=CriticNetwork,
                 goal_value_net_ctor=None,
                 goal_net_position_only=False,
                 use_non_her_critic=False,
                 keep_her_rate=0.,
                 down_sample_high_value=1.,
                 use_parallel_network=False,
                 reward_weights=None,
                 env=None,
                 config: TrainerConfig = None,
                 ou_stddev=0.2,
                 ou_damping=0.15,
                 critic_loss_ctor=None,
                 num_critic_replicas=1,
                 num_critics_for_training=0,
                 split_exp_on_replicas="",
                 target_update_tau=0.05,
                 target_update_period=1,
                 rollout_random_action=0.,
                 dqda_clipping=None,
                 action_l2=0,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 debug_summaries=False,
                 name="DdpgAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            actor_network_ctor (Callable): Function to construct the actor network.
                ``actor_network_ctor`` needs to accept ``input_tensor_spec`` and
                ``action_spec`` as its arguments and return an actor network.
                The constructed network will be called with ``forward(observation, state)``.
            critic_network_ctor (Callable): Function to construct the critic
                network. ``critic_netwrok_ctor`` needs to accept ``input_tensor_spec``
                which is a tuple of ``(observation_spec, action_spec)``. The
                constructed network will be called with
                ``forward((observation, action), state)``.
            goal_value_net_ctor (Callable or None): if not None, it used to construct
                the network to predict value based on goal input only.
            use_non_her_critic (bool): whether to use non-her experience to train a
                separate critic.  This can be useful for goal-based planning.
            use_parallel_network (bool): whether to use parallel network for
                calculating critics.
            reward_weights (list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the q values
                is used for training the actor.
            num_critic_replicas (int): number of critics to be used. Default is 1.
            num_critics_for_training (int): if not zero, only use the first n critics
                for critic training.
            split_exp_on_replicas (str): use "envid" or "pos" in replay buffer for
                hashing content to different critic replicas.  Default is no split.
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. ``env`` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            ou_stddev (float): Standard deviation for the Ornstein-Uhlenbeck
                (OU) noise added in the default collect policy.
            ou_damping (float): Damping factor for the OU noise added in the
                default collect policy.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            rollout_random_action (float): the probability of taking a uniform
                random action during a ``rollout_step()``. 0 means always directly
                taking actions added with OU noises and 1 means always sample
                uniformly random actions. A bigger value results in more
                exploration during rollout.
            dqda_clipping (float): when computing the actor loss, clips the
                gradient dqda element-wise between ``[-dqda_clipping, dqda_clipping]``.
                Does not perform clipping if ``dqda_clipping == 0``.
            action_l2 (float): weight of squared action l2-norm on actor loss.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """

        critic_network = critic_network_ctor(
            input_tensor_spec=(observation_spec, action_spec))
        actor_network = actor_network_ctor(
            input_tensor_spec=observation_spec, action_spec=action_spec)
        goal_net = None
        self._goal_net_position_only = goal_net_position_only
        if goal_value_net_ctor is not None:
            input_tensor_spec = observation_spec.copy()
            self._goal_net_fields = ["desired_goal", "aux_desired"]
            if goal_net_position_only:
                self._goal_net_fields.remove("aux_desired")
                self._goal_net_fields.append("achieved_goal")
            for k, _ in observation_spec.items():
                if k not in self._goal_net_fields:
                    del input_tensor_spec[k]
            goal_net = goal_value_net_ctor(input_tensor_spec=input_tensor_spec)
            if goal_net_position_only:
                if use_parallel_network:
                    goal_net = goal_net.make_parallel(num_critic_replicas)
                else:
                    goal_net = alf.networks.NaiveParallelNetwork(
                        goal_net, num_critic_replicas)
        self._split_exp_on_replicas = split_exp_on_replicas
        if use_parallel_network:
            critic_networks = critic_network.make_parallel(num_critic_replicas)
        else:
            critic_networks = alf.networks.NaiveParallelNetwork(
                critic_network, num_critic_replicas)
        self._action_l2 = action_l2

        train_state_spec = DdpgState(
            actor=DdpgActorState(
                actor=actor_network.state_spec,
                critics=critic_networks.state_spec),
            critics=DdpgCriticState(
                critics=critic_networks.state_spec,
                target_actor=actor_network.state_spec,
                target_critics=critic_networks.state_spec))

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=train_state_spec,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])

        self._actor_network = actor_network
        self._num_critic_replicas = num_critic_replicas
        self._num_critics_for_training = num_critics_for_training
        self._critic_networks = critic_networks
        self._goal_net = goal_net
        self._goal_net_target = None
        if goal_net_position_only:
            self._goal_net_target = self._goal_net.copy(
                name='target_critic_goal_net')

        self._non_her_critic = None
        self._non_her_target = None
        self._keep_her_rate = keep_her_rate
        self._down_sample_high_value = down_sample_high_value
        if use_non_her_critic:
            self._non_her_critic = critic_network_ctor(
                input_tensor_spec=(observation_spec, action_spec),
                name="critic_non_her")
            self._non_her_target = self._non_her_critic.copy(
                name='target_critic_non_her')

        self._reward_weights = None
        if reward_weights:
            self._reward_weights = torch.tensor(
                reward_weights, dtype=torch.float32)

        self._target_actor_network = actor_network.copy(
            name='target_actor_networks')
        self._target_critic_networks = critic_networks.copy(
            name='target_critic_networks')

        self._rollout_random_action = float(rollout_random_action)

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        self._critic_losses = [None] * num_critic_replicas
        for i in range(num_critic_replicas):
            self._critic_losses[i] = critic_loss_ctor(
                name=("critic_loss" + str(i)))

        self._ou_process = common.create_ou_process(action_spec, ou_stddev,
                                                    ou_damping)
        models = [self._actor_network, self._critic_networks]
        target_models = [
            self._target_actor_network, self._target_critic_networks
        ]
        if goal_net:
            if goal_net_position_only:
                models.append(self._goal_net)
                target_models.append(self._goal_net_target)
                self._goal_value_losses = [None] * num_critic_replicas
                for i in range(num_critic_replicas):
                    self._goal_value_losses[i] = critic_loss_ctor(
                        name=("goal_value_losses" + str(i)))
            else:
                self._goal_value_loss = critic_loss_ctor(
                    name="goal_value_loss")
        if self._non_her_critic:
            models.append(self._non_her_critic)
            target_models.append(self._non_her_target)
            self._non_her_loss = critic_loss_ctor(name="critic_loss_non_her")
        self._update_target = common.get_target_updater(
            models=models,
            target_models=target_models,
            tau=target_update_tau,
            period=target_update_period)

        self._dqda_clipping = dqda_clipping

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy=1.):
        action, state = self._actor_network(
            time_step.observation, state=state.actor.actor)
        empty_state = nest.map_structure(lambda x: (), self.train_state_spec)

        def _sample(a, ou):
            if epsilon_greedy == 0:
                return a
            elif epsilon_greedy >= 1.0:
                return a + ou()
            else:
                ind_explore = torch.where(
                    torch.rand(a.shape[:1]) < epsilon_greedy)
                noisy_a = a + ou()
                a[ind_explore[0], :] = noisy_a[ind_explore[0], :]
                return a

        noisy_action = nest.map_structure(_sample, action, self._ou_process)
        noisy_action = nest.map_structure(spec_utils.clip_to_spec,
                                          noisy_action, self._action_spec)
        state = empty_state._replace(
            actor=DdpgActorState(actor=state, critics=()))
        return AlgStep(
            output=noisy_action,
            state=state,
            info=DdpgInfo(action_distribution=action))

    def rollout_step(self, time_step: TimeStep, state=None):
        if self.need_full_rollout_state():
            raise NotImplementedError("Storing RNN state to replay buffer "
                                      "is not supported by DdpgAlgorithm")

        def _update_random_action(spec, noisy_action):
            random_action = spec_utils.scale_to_spec(
                torch.rand_like(noisy_action) * 2 - 1, spec)
            ind = torch.where(
                torch.rand(noisy_action.shape[:1]) < self.
                _rollout_random_action)
            noisy_action[ind[0], :] = random_action[ind[0], :]

        pred_step = self.predict_step(time_step, state, epsilon_greedy=1.0)
        if self._rollout_random_action > 0:
            nest.map_structure(_update_random_action, self._action_spec,
                               pred_step.output)
        return pred_step

    def _critic_train_step(self, exp: Experience, state: DdpgCriticState):
        target_action, target_actor_state = self._target_actor_network(
            exp.observation, state=state.target_actor)
        target_q_values, target_critic_states = self._target_critic_networks(
            (exp.observation, target_action), state=state.target_critics)

        if self._num_critic_replicas > 1:
            if self._num_critics_for_training > 0:
                target_q_values = target_q_values[:, :self.
                                                  _num_critics_for_training,
                                                  ...]
            target_q_values = target_q_values.min(dim=1)[0]
        else:
            target_q_values = target_q_values.squeeze(dim=1)

        q_values, critic_states = self._critic_networks(
            (exp.observation, exp.action), state=state.critics)

        non_her_q_values, non_her_critic = (), ()
        non_her_target_q, non_her_target = (), ()
        if self._non_her_critic:  # assumes no states.
            non_her_q_values, non_her_critic = self._non_her_critic(
                (exp.observation, exp.action), state=state.non_her_critic)
            non_her_target_q, non_her_target = self._non_her_target(
                (exp.observation, exp.action), state=state.non_her_target)

        state = DdpgCriticState(
            critics=critic_states,
            non_her_critic=non_her_critic,
            target_actor=target_actor_state,
            target_critics=target_critic_states,
            non_her_target=non_her_target)

        info = DdpgCriticInfo(
            q_values=q_values,
            non_her_q_values=non_her_q_values,
            target_q_values=target_q_values,
            non_her_target=non_her_target_q)

        if self._goal_net:
            goal_input = exp.observation.copy()
            for k, _ in exp.observation.items():
                if k not in self._goal_net_fields:
                    goal_input.pop(k, None)
            goal_values, _ = self._goal_net(goal_input, state=())
            if self._goal_net_position_only:
                target_goal_values, _ = self._goal_net_target(
                    goal_input, state=())
                if self._num_critic_replicas > 1:
                    if self._num_critics_for_training > 0:
                        target_goal_values = target_goal_values[:, :self.
                                                                _num_critics_for_training,
                                                                ...]
                    target_goal_values = target_goal_values.min(dim=1)[0]
                else:
                    target_goal_values = target_goal_values.squeeze(dim=1)
                info = info._replace(target_goal_values=target_goal_values)
            info = info._replace(goal_values=goal_values)

        return state, info

    def _actor_train_step(self, exp: Experience, state: DdpgActorState):
        action, actor_state = self._actor_network(
            exp.observation, state=state.actor)

        q_values, critic_states = self._critic_networks(
            (exp.observation, action), state=state.critics)
        if q_values.ndim == 3:
            # Multidimensional reward: [B, num_criric_replicas, reward_dim]
            if self._reward_weights is None:
                q_values = q_values.sum(dim=2)
            else:
                q_values = torch.tensordot(
                    q_values, self._reward_weights, dims=1)

        if self._num_critic_replicas > 1:
            if self._num_critics_for_training > 0:
                q_values = q_values[:, :self._num_critics_for_training, ...]
            q_value = q_values.min(dim=1)[0]
        else:
            q_value = q_values.squeeze(dim=1)

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
        state = DdpgActorState(actor=actor_state, critics=critic_states)
        info = LossInfo(loss=sum(nest.flatten(actor_loss)), extra=actor_loss)
        return AlgStep(output=action, state=state, info=info)

    def train_step(self, exp: Experience, state: DdpgState):
        critic_states, critic_info = self._critic_train_step(
            exp=exp, state=state.critics)
        policy_step = self._actor_train_step(exp=exp, state=state.actor)
        return policy_step._replace(
            state=DdpgState(actor=policy_step.state, critics=critic_states),
            info=DdpgInfo(
                action_distribution=policy_step.output,
                critic=critic_info,
                actor_loss=policy_step.info))

    def calc_loss(self, experience, train_info: DdpgInfo):

        critic_losses = [None] * self._num_critic_replicas
        for i in range(self._num_critic_replicas):
            loss_i = self._critic_losses[i](
                experience=experience,
                value=train_info.critic.q_values[:, :, i, ...],
                target_value=train_info.critic.target_q_values).loss

            if not self._non_her_critic and self._down_sample_high_value < 1:
                q = train_info.critic.q_values[0, :, i]
                if q.ndim == 2:
                    q = q[:,
                          0]  # assuming first dim of multi-dim reward is goal reward.
                batch_size = q.shape[0]
                bottom_v, _ = torch.topk(
                    q, int(batch_size * 0.03), largest=False, sorted=True)
                loss_i[:,
                       (torch.rand(batch_size) >= self._down_sample_high_value)
                       & (q > bottom_v[-1])] = 0.

            def _mask(t):
                if experience.batch_info == ():
                    return t
                divr = self._num_critic_replicas
                if self._split_exp_on_replicas == "pos":
                    x = experience.batch_info.positions % divr != i
                    t[:, x] = 0.
                elif self._split_exp_on_replicas == "envid":
                    x = experience.batch_info.env_ids % divr != i
                    t[:, x] = 0.
                return t

            critic_losses[i] = _mask(loss_i)

        critic_loss = math_ops.add_n(critic_losses)

        if self._goal_net:
            if self._goal_net_position_only:
                for i in range(self._num_critic_replicas):
                    critic_loss += self._goal_value_losses[i](
                        experience=experience,
                        value=train_info.critic.goal_values[:, :, i, ...],
                        target_value=train_info.critic.target_goal_values).loss
            else:
                critic_loss += self._goal_value_loss(
                    experience=experience,
                    value=train_info.critic.goal_values,
                    target_value=train_info.critic.goal_values).loss

        if self._non_her_critic and len(self._critic_losses) > 0:
            loss = self._non_her_loss(
                experience=experience,
                value=train_info.critic.non_her_q_values,
                target_value=train_info.critic.non_her_target.detach()).loss
            if experience.batch_info != ():
                discard = torch.rand(loss.shape[1]) >= self._keep_her_rate
                loss[:, experience.batch_info.her & discard] = 0.
            if self._down_sample_high_value < 1:
                q = train_info.critic.non_her_q_values[0, :]
                if q.ndim == 2:
                    q = q[:,
                          0]  # assuming first dim of multi-dim reward is goal reward.
                batch_size = q.shape[0]
                bottom_v, _ = torch.topk(
                    q, int(batch_size * 0.03), largest=False, sorted=True)
                loss[:,
                     (torch.rand(batch_size) >= self._down_sample_high_value) &
                     (q > bottom_v[-1])] = 0.
            critic_loss += loss

        if (experience.batch_info != ()
                and experience.batch_info.importance_weights != ()):
            valid_masks = (experience.step_type != StepType.LAST).to(
                torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = (
                (critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        actor_loss = train_info.actor_loss

        return LossInfo(
            loss=critic_loss + actor_loss.loss,
            priority=priority,
            extra=DdpgLossInfo(critic=critic_loss, actor=actor_loss.extra))

    def after_update(self, experience, train_info: DdpgInfo):
        self._update_target()

    def _trainable_attributes_to_ignore(self):
        return [
            '_target_actor_network', '_target_critic_networks',
            '_non_her_target', '_goal_net_target'
        ]
