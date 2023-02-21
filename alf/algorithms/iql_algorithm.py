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
"""Implicit Q-Learning Algorithm."""

import numpy as np
import functools
import torch

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.data_structures import TimeStep, LossInfo, namedtuple
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks import ValueNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, dist_utils, math_ops

IqlActionState = namedtuple(
    "IqlActionState", ["actor_network", "critic"], default_value=())

IqlCriticState = namedtuple("IqlCriticState", ["critics", "target_critics"])

IqlState = namedtuple(
    "IqlState", ["action", "actor", "critic"], default_value=())

IqlCriticInfo = namedtuple("IqlCriticInfo",
                           ["critics", "target_value", "value"])

IqlActorInfo = namedtuple("IqlActorInfo", ["actor_loss"], default_value=())

IqlInfo = namedtuple(
    "IqlInfo", [
        "reward", "step_type", "discount", "action", "action_distribution",
        "actor", "critic"
    ],
    default_value=())

IqlLossInfo = namedtuple('IqlLossInfo', ('actor', 'critic'))


@alf.configurable
class IqlAlgorithm(OffPolicyAlgorithm):
    r"""Implicit q-learning algorithm (IQL).

    IQL is an offline reinforcement learning method. The idea is
    that instead of constraining the critic network or policy to avoid the
    value function extrapolation issue, IQL conducts learning using only
    in-sample data, thus voiding the issues when querying the critic network
    with out-of-distribution actions, a problem commonly faced in offline RL.

    Reference:
    ::
        Kostrikov, et al. "Offline Reinforcement Learning with Implicit Q-Learning",
        arXiv:2110.06169
    """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 critic_network_cls=CriticNetwork,
                 v_network_cls=ValueNetwork,
                 reward_weights=None,
                 epsilon_greedy=None,
                 calculate_priority=False,
                 num_critic_replicas=2,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 target_update_tau=0.05,
                 target_update_period=1,
                 temperature=1.0,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 value_optimizer=None,
                 expectile=0.8,
                 max_exp_advantage=100,
                 checkpoint=None,
                 debug_summaries=False,
                 name="IqlAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (BoundedTensorSpec): representing the actions. Only
                continuous action is supported currently.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            actor_network_cls (Callable): is used to construct the actor network.
                The constructed actor network will be called
                to sample continuous actions. All of its output specs must be
                continuous. Discrete actor network is not supported.
            critic_network_cls (Callable): is used to construct critic network.
            v_network_cls (Callable): is used to construct a value network.
                for estimating the expectile of q values.
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
            calculate_priority (bool): whether to calculate priority. This is
                only useful if priority replay is enabled.
            num_critic_replicas (int): number of critics to be used. Default is 2.
                This is only applied for critic networks. The value network is
                not replicated.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. It only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            critic_loss_ctor (None|OneStepTDLoss|MultiStepLoss): a critic loss
                constructor. If ``None``, a default ``OneStepTDLoss`` will be used.
            target_update_tau (float): Factor for soft update of the target
                networks.
            target_update_period (int): Period for soft update of the target
                networks.
            temperature (float): the hyper-parameter for scaling the advantages.
                It corresponds to 1/beta in Eqn.(7) of the paper.
            actor_optimizer (torch.optim.optimizer): The optimizer for actor.
            critic_optimizer (torch.optim.optimizer): The optimizer for critic.
            value_optimizer (torch.optim.optimizer): The optimizer for value network.
            expectile (float): the expectile value for value learning.
            max_exp_advantage (float): clamp the exponentiated advantages with
                this value before being applied to weight the actor loss.
            checkpoint (None|str): a string in the format of "prefix@path",
                where the "prefix" is the multi-step path to the contents in the
                checkpoint to be loaded. "path" is the full path to the checkpoint
                file saved by ALF. Refer to ``Algorithm`` for more details.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this algorithm.
        """
        self._num_critic_replicas = num_critic_replicas
        self._calculate_priority = calculate_priority
        if epsilon_greedy is None:
            epsilon_greedy = alf.utils.common.get_epsilon_greedy(config)
        self._epsilon_greedy = epsilon_greedy

        critic_networks, actor_network, v_network = self._make_networks(
            observation_spec, action_spec, reward_spec, actor_network_cls,
            critic_network_cls, v_network_cls)

        action_state_spec = IqlActionState(
            actor_network=actor_network.state_spec, critic=())

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=IqlState(
                action=action_state_spec,
                actor=critic_networks.state_spec,
                critic=IqlCriticState(
                    critics=critic_networks.state_spec,
                    target_critics=critic_networks.state_spec)),
            predict_state_spec=IqlState(action=action_state_spec),
            reward_weights=reward_weights,
            env=env,
            config=config,
            checkpoint=checkpoint,
            debug_summaries=debug_summaries,
            name=name)

        if actor_optimizer is not None and actor_network is not None:
            self.add_optimizer(actor_optimizer, [actor_network])
        if critic_optimizer is not None:
            self.add_optimizer(critic_optimizer, [critic_networks])
        if value_optimizer is not None:
            self.add_optimizer(value_optimizer, [v_network])

        self._temperature = temperature
        self._actor_network = actor_network
        self._critic_networks = critic_networks
        self._target_critic_networks = self._critic_networks.copy(
            name='target_critic_networks')
        self._v_network = v_network

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)
        # Have different names to separate their summary curves
        self._critic_losses = []
        for i in range(num_critic_replicas):
            self._critic_losses.append(
                critic_loss_ctor(name="critic_loss%d" % (i + 1)))

        self._update_target = common.TargetUpdater(
            models=[self._critic_networks],
            target_models=[self._target_critic_networks],
            tau=target_update_tau,
            period=target_update_period)

        self._expectile = expectile
        self._max_exp_advantage = max_exp_advantage

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       continuous_actor_network_cls, critic_network_cls,
                       v_network_cls):
        def _make_parallel(net):
            return net.make_parallel(
                self._num_critic_replicas * reward_spec.numel)

        def _check_spec_equal(spec1, spec2):
            assert nest.flatten(spec1) == nest.flatten(spec2), (
                "Unmatched action specs: {} vs. {}".format(spec1, spec2))

        actor_network = continuous_actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        critic_network = critic_network_cls(
            input_tensor_spec=(observation_spec, action_spec))
        critic_networks = _make_parallel(critic_network)

        v_network = v_network_cls(input_tensor_spec=observation_spec)

        return critic_networks, actor_network, v_network

    def _predict_action(self,
                        observation,
                        state: IqlActionState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        rollout=False):

        new_state = IqlActionState()

        continuous_action_dist, actor_network_state = self._actor_network(
            observation, state=state.actor_network)
        new_state = new_state._replace(actor_network=actor_network_state)
        if eps_greedy_sampling:
            continuous_action = dist_utils.epsilon_greedy_sample(
                continuous_action_dist, epsilon_greedy)
        else:
            continuous_action = dist_utils.rsample_action_distribution(
                continuous_action_dist)

        action_dist = continuous_action_dist
        action = continuous_action

        return action_dist, action, new_state

    def predict_step(self, inputs: TimeStep, state: IqlState):

        _, action, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=self._epsilon_greedy,
            eps_greedy_sampling=True)

        return AlgStep(output=action, state=IqlState(action=action_state))

    def rollout_step(self, inputs: TimeStep, state: IqlState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        action_dist, action, _, action_state = self._predict_action(
            inputs.observation,
            state=state.action,
            epsilon_greedy=1.0,
            eps_greedy_sampling=True,
            rollout=True)

        if self.need_full_rollout_state():
            _, critics_state = self._compute_critics(
                self._critic_networks, inputs.observation, action,
                state.critic.critics)
            _, target_critics_state = self._compute_critics(
                self._target_critic_networks, inputs.observation, action,
                state.critic.target_critics)
            critic_state = IqlCriticState(
                critics=critics_state, target_critics=target_critics_state)

            actor_state = critics_state

        else:
            actor_state = state.actor
            critic_state = state.critic

        new_state = IqlState(
            action=action_state, actor=actor_state, critic=critic_state)
        return AlgStep(
            output=action,
            state=new_state,
            info=IqlInfo(action=action, action_distribution=action_dist))

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         critics_state,
                         replica_min=True,
                         apply_reward_weights=True):
        observation = (observation, action)

        # critics shape [B, replicas]
        critics, critics_state = critic_net(observation, state=critics_state)

        # For multi-dim reward, do
        # [B, replicas * reward_dim] -> [B, replicas, reward_dim]
        # For scalar reward, do nothing
        if self.has_multidim_reward():
            remaining_shape = critics.shape[2:]
            critics = critics.reshape(-1, self._num_critic_replicas,
                                      *self._reward_spec.shape,
                                      *remaining_shape)

        if replica_min:
            if self.has_multidim_reward():
                sign = self.reward_weights.sign()
                critics = (critics * sign).min(dim=1)[0] * sign
            else:
                critics = critics.min(dim=1)[0]

        if apply_reward_weights and self.has_multidim_reward():
            critics = critics * self.reward_weights
            critics = critics.sum(dim=-1)

        return critics, critics_state

    def _actor_train_step(self, inputs: TimeStep, state, action_distribution,
                          v_value, rollout_info):

        # IQL uses target critic network for computing the value learning target
        q_value, critics_state = self._compute_critics(
            self._target_critic_networks, inputs.observation,
            rollout_info.action, state)

        weight = torch.exp((q_value - v_value) / self._temperature)
        weight = torch.clamp(weight, max=self._max_exp_advantage)

        # log_pi_data: the log probability computed with the action from dataset
        log_pi_data = dist_utils.compute_log_probability(
            action_distribution, rollout_info.action)
        weighted_log_pi = -weight.detach() * log_pi_data
        actor_loss = weighted_log_pi

        actor_info = LossInfo(
            loss=actor_loss, extra=IqlActorInfo(actor_loss=actor_loss))
        return critics_state, actor_info

    def _critic_train_step(self, inputs: TimeStep, state: IqlCriticState,
                           rollout_info: IqlInfo):

        # use dataset action for Q learning
        critics, critics_state = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_info.action,
            state.critics,
            replica_min=False,
            apply_reward_weights=False)

        # use value network (there is no target value network), also no replica
        # use an upper quantile
        # calculate the state value, which will be used in two places:
        # 1) used for constructing the the target value for q-learning
        # 2) used for training the value network using expectile loss over the
        # difference with respect to the prediction of target q-network
        value, critics_state = self._v_network(
            inputs.observation, state=critics_state)
        value = value.squeeze(-1)

        # use dataset state action pair for training
        target_value, target_critics_state = self._compute_critics(
            self._target_critic_networks,
            inputs.observation,
            rollout_info.action,
            state.target_critics,
            apply_reward_weights=False)

        state = IqlCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = IqlCriticInfo(
            critics=critics, target_value=target_value, value=value)

        return state, info

    def train_step(self, inputs: TimeStep, state: IqlState,
                   rollout_info: IqlInfo):
        self._training_started = True

        (action_distribution, action, action_state) = self._predict_action(
            inputs.observation, state=state.action)

        critic_state, critic_info = self._critic_train_step(
            inputs, state.critic, rollout_info)

        actor_state, actor_loss = self._actor_train_step(
            inputs, state.actor, action_distribution, critic_info.value,
            rollout_info)

        state = IqlState(
            action=action_state, actor=actor_state, critic=critic_state)
        info = IqlInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            action_distribution=action_distribution,
            actor=actor_loss,
            critic=critic_info)
        return AlgStep(action, state, info)

    def after_update(self, root_inputs, info: IqlInfo):
        self._update_target()

    def calc_loss(self, info: IqlInfo):
        critic_loss = self._calc_critic_loss(info)
        actor_loss = info.actor

        loss = math_ops.add_ignore_empty(actor_loss.loss, critic_loss.loss)

        return LossInfo(
            loss=loss,
            priority=critic_loss.priority,
            extra=IqlLossInfo(
                actor=actor_loss.extra, critic=critic_loss.extra))

    def _calc_critic_loss(self, info: IqlInfo):
        def exp_loss(diff, expectile):
            weight = torch.where(diff > 0, expectile, (1 - expectile))
            return weight * (diff**2)

        critic_info = info.critic
        critic_losses = []
        for i, l in enumerate(self._critic_losses):
            critic_losses.append(
                l(info=info,
                  value=critic_info.critics[:, :, i, ...],
                  target_value=critic_info.value.detach()).loss
            )  # use ``critic_info.value`` for constructing target value

        critic_loss = math_ops.add_n(critic_losses)

        # q_target_critic_min - v
        value_diff = critic_info.target_value.detach() - critic_info.value
        value_loss = exp_loss(value_diff, self._expectile)

        if self._debug_summaries and alf.summary.should_record_summaries():
            with alf.summary.scope(self._name):
                alf.summary.scalar("value_loss", value_loss.mean())

        critic_loss = critic_loss + value_loss

        if self._calculate_priority:
            valid_masks = (info.step_type != StepType.LAST).to(torch.float32)
            valid_n = torch.clamp(valid_masks.sum(dim=0), min=1.0)
            priority = (
                (critic_loss * valid_masks).sum(dim=0) / valid_n).sqrt()
        else:
            priority = ()

        return LossInfo(
            loss=critic_loss,
            priority=priority,
            extra=critic_loss / float(self._num_critic_replicas))

    def _trainable_attributes_to_ignore(self):
        return ['_target_critic_networks']
