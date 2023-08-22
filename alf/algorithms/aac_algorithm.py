# Copyright (c) 2023 Horizon Robotics. All Rights Reserved.
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
"""Aligned Actor critic algorithm."""

import functools
import torch

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.one_step_loss import OneStepTDLoss
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.data_structures import TimeStep, AlgStep, LossInfo, namedtuple, StepType
from alf.utils import common, dist_utils, math_ops, tensor_utils
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from .config import TrainerConfig

# AacActionState = namedtuple(
#     "AacActionState", ["actor_network_1", "actor_network_2"],
#     default_value=())

AacActorState = namedtuple(
    "AacActorState", ["actor_network_1", "actor_network_2"], default_value=())

AacValueState = namedtuple(
    "AacValueState", ["value_network_1", "value_network_2"], default_value=())

# AacCriticState = namedtuple(
#     "AacCriticState", ["actor", "value"], default_value=())

AacState = namedtuple("AacState", ["actor", "value"])  #, default_value=())

AacCriticInfo = namedtuple("AacCriticInfo", ["critics", "target_critics"])

AacInfo = namedtuple(
    "AacInfo", [
        "reward", "step_type", "discount", "action", "action_distribution",
        "critic", "discounted_return"
    ],
    default_value=())


@alf.configurable
class AacAlgorithm(OffPolicyAlgorithm):
    """Aligned Actor-Critic algorithm. """

    def __init__(self,
                 observation_spec,
                 action_spec: BoundedTensorSpec,
                 reward_spec=TensorSpec(()),
                 actor_network_cls=ActorDistributionNetwork,
                 value_network_cls=ValueNetwork,
                 num_mc_samples=10,
                 epsilon_greedy=None,
                 env=None,
                 config: TrainerConfig = None,
                 critic_loss_ctor=None,
                 initial_log_alpha=0.0,
                 optimizer=None,
                 debug_summaries=False,
                 name="AacAlgorithm"):
        """
        Args:
        
        """

        # self._num_replicas = num_replicas
        actor_network_1, actor_network_2, value_network_1, value_network_2 = \
            self._make_networks(observation_spec, action_spec,
                                reward_spec, actor_network_cls,
                                value_network_cls)

        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value(
                'TrainerConfig.epsilon_greedy')

        log_alpha = torch.tensor(float(initial_log_alpha))

        actor_state_spec = AacActorState(
            actor_network_1=actor_network_1.state_spec,
            actor_network_2=actor_network_2.state_spec)
        value_state_spec = AacValueState(
            value_network_1=value_network_1.state_spec,
            value_network_2=value_network_2.state_spec)

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            train_state_spec=AacState(
                actor=actor_state_spec, value=value_state_spec),
            predict_state_spec=AacState(
                actor=actor_state_spec, value=value_state_spec),
            env=env,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._epsilon_greedy = epsilon_greedy
        self._num_mc_samples = num_mc_samples
        self._training_started = False
        self._log_alpha = log_alpha
        # if max_log_alpha is not None:
        #     self._max_log_alpha = torch.tensor(float(max_log_alpha))
        # else:
        #     self._max_log_alpha = None

        self._actor_network_1 = actor_network_1
        self._actor_network_2 = actor_network_2
        self._value_network_1 = value_network_1
        self._value_network_2 = value_network_2

        self._actor = self._actor_network_1
        self._actor_choice = "actor_network_1"

        if critic_loss_ctor is None:
            critic_loss_ctor = OneStepTDLoss
        critic_loss_ctor = functools.partial(
            critic_loss_ctor, debug_summaries=debug_summaries)

        self._critic_loss_1 = critic_loss_ctor(name="critic_loss_1")
        self._critic_loss_2 = critic_loss_ctor(name="critic_loss_2")

    def _make_networks(self, observation_spec, action_spec, reward_spec,
                       actor_network_cls, value_network_cls):
        assert actor_network_cls is not None, (
            "ActorNetwork must be provided!")

        actor_network_1 = actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        actor_network_2 = actor_network_cls(
            input_tensor_spec=observation_spec, action_spec=action_spec)

        value_network_1 = value_network_cls(input_tensor_spec=observation_spec)

        value_network_2 = value_network_cls(input_tensor_spec=observation_spec)

        return actor_network_1, actor_network_2, value_network_1, value_network_2

    def _predict_action(self,
                        observation,
                        state: AacActorState,
                        epsilon_greedy=None,
                        eps_greedy_sampling=False,
                        rollout=False):

        new_state = AacActorState()
        if rollout and not self._training_started:
            # This uniform sampling during initial collect stage is
            # important since current explore_network is deterministic
            action = alf.nest.map_structure(
                lambda spec: spec.sample(outer_dims=observation.shape[:1]),
                self._action_spec)
            action_dist = ()
        else:
            action_dist, actor_network_state = self._actor(
                observation, state=getattr(state, self._actor_choice))
            new_state = new_state._replace(
                **{self._actor_choice: actor_network_state})
            if eps_greedy_sampling:
                action = dist_utils.epsilon_greedy_sample(
                    action_dist, epsilon_greedy)
            else:
                action = dist_utils.rsample_action_distribution(action_dist)

        return action_dist, action, new_state

    def predict_step(self, inputs: TimeStep, state: AacState):
        action_dist, action, actor_state = self._predict_action(
            inputs.observation,
            state=state.actor,
            epsilon_greedy=self._epsilon_greedy,
            eps_greedy_sampling=True)

        new_state = AacState(actor=actor_state, value=AacValueState())

        return AlgStep(
            output=action,
            state=new_state,
            info=AacInfo(action_distribution=action_dist))

    def rollout_step(self, inputs: TimeStep, state: AacState):
        """``rollout_step()`` basically predicts actions like what is done by
        ``predict_step()``. Additionally, if states are to be stored a in replay
        buffer, then this function also call ``_critic_networks`` and
        ``_target_critic_networks`` to maintain their states.
        """
        if inputs.step_type == StepType.FIRST:
            if self._actor_choice == "actor_network_1":
                self._actor_choice = "actor_network_2"
                self._actor = self._actor_network_2
            else:
                self._actor_choice = "actor_network_1"
                self._actor = self._actor_network_1

        _, action, actor_state = self._predict_action(
            inputs.observation,
            state=state.actor,
            epsilon_greedy=1.0,
            eps_greedy_sampling=True,
            rollout=True)

        new_state = AacState(actor=actor_state, value=state.value)
        return AlgStep(
            output=action, state=new_state, info=AacInfo(action=action))

    def _compute_critic_train_info(self, observation, action, state: AacState):
        action_dist_1, actor_network_state_1 = self._actor_network_1(
            observation, state.actor.actor_network_1)
        action_dist_2, actor_network_state_2 = self._actor_network_2(
            observation, state.actor.actor_network_2)
        actor_state = AacActorState(
            actor_network_1=actor_network_state_1,
            actor_network_2=actor_network_state_2)

        log_prob_1 = action_dist_1.log_prob(action)
        log_prob_2 = action_dist_2.log_prob(action)

        value_1, value_network_state_1 = self._value_network_1(
            observation, state.value.value_network_1)
        value_2, value_network_state_2 = self._value_network_2(
            observation, state.value.value_network_2)
        value_state = AacValueState(
            value_network_1=value_network_state_1,
            value_network_2=value_network_state_2)

        const_alpha = torch.exp(self._log_alpha)
        alpha = const_alpha * torch.abs(value_1 - value_2)

        critics_1 = alpha * (log_prob_1 + value_1)
        critics_2 = alpha * (log_prob_2 + value_2)

        # sample a set of a' for each s'
        target_action_1 = action_dist_1.sample((self._num_mc_samples, ))
        target_action_2 = action_dist_2.sample((self._num_mc_samples, ))

        # compute log_prob for a' crossly
        target_log_prob_1 = action_dist_1.log_prob(target_action_2).mean(0)
        target_log_prob_2 = action_dist_2.log_prob(target_action_1).mean(0)

        target_critics_1 = alpha * (target_log_prob_2 + value_2)
        target_critics_2 = alpha * (target_log_prob_1 + value_1)

        critics = (critics_1, critics_2)
        target_critics = (target_critics_1, target_critics_2)
        critic_info = AacCriticInfo(
            critics=critics, target_critics=target_critics)

        critic_state = AacState(actor=actor_state, value=value_state)

        return critic_info, critic_state

    def train_step(self, inputs: TimeStep, state: AacState,
                   rollout_info: AacInfo):

        self._training_started = True

        # train actor and value networks via TD-learning
        critic_info, critic_state = self._compute_critic_train_info(
            inputs.observation, rollout_info.action, state)

        info = AacInfo(
            reward=inputs.reward,
            step_type=inputs.step_type,
            discount=inputs.discount,
            action=rollout_info.action,
            critic=critic_info,
            discounted_return=rollout_info.discounted_return)

        # state = AacState(actor=critic_state.actor,
        #                  value=critic_state.value)

        return AlgStep(state=critic_state, info=info)

    def calc_loss(self, info: AacInfo):
        critics = info.critic.critics
        target_critics = info.critic.target_critics
        critic_loss_1 = self._critic_loss_1(
            info=info, value=critics[0], target_value=target_critics[0]).loss
        critic_loss_2 = self._critic_loss_2(
            info=info, value=critics[1], target_value=target_critics[1]).loss

        critic_loss = math_ops.add_n([critic_loss_1, critic_loss_2])

        return LossInfo(loss=critic_loss)
