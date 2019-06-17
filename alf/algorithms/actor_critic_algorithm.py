# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

from collections import namedtuple
from typing import Callable

import gin.tf

import tensorflow as tf

from tf_agents.agents.tf_agent import LossInfo
from tf_agents.networks.network import Network, DistributionNetwork
from tf_agents.trajectories.policy_step import PolicyStep

from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.on_policy_algorithm import ActionTimeStep
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.on_policy_algorithm import TrainingInfo
from alf.algorithms.algorithm import Algorithm

ActorCriticState = namedtuple("ActorCriticPolicyState",
                              ["actor_state", "value_state", "icm_state"])

ActorCriticInfo = namedtuple("ActorCriticInfo",
                             ["value", "icm_reward", "icm_info"])

ActorCriticAlgorithmLossInfo = namedtuple("ActorCriticAlgorithmLossInfo",
                                          ["ac", "icm"])


@gin.configurable
class ActorCriticAlgorithm(OnPolicyAlgorithm):
    def __init__(self,
                 action_spec,
                 actor_network: DistributionNetwork,
                 value_network: Network,
                 encoding_network: Network = None,
                 intrinsic_curiosity_module=None,
                 intrinsic_reward_coef=1.0,
                 extrinsic_reward_coef=1.0,
                 loss=None,
                 optimizer=None,
                 gradient_clipping=None,
                 reward_shaping_fn: Callable =None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="ActorCriticAlgorithm"):
        """Create an ActorCriticAlgorithm

        Args:
          action_spec: A nest of BoundedTensorSpec representing the actions.
          actor_network (DistributionNetwork): A network that returns nested
            tensor of action distribution for each observation given observation
            and network state.
          value_net (Network): A function that returns value tensor from neural
            net predictions for each observation given observation and nwtwork
            state.
          encoding_network (Network): A function that encodes the observation
          intrinsic_curiosity_module (Algorithm): an algorithm whose outputs
            is a scalar intrinsid reward
          intrinsic_reward_coef: Coefficient for intrinsic reward
          extrinsic_reward_coef: Coefficient for extrinsic reward
          loss (None|ActorCriticLoss): an object for calculating loss. If None,
            a default ActorCriticLoss will be used.
          optimizer (tf.optimizers.Optimizer): The optimizer for training
          gradient_clipping (float): If not None, serve as a positive threshold
            for clipping gradient norms
          reward_shaping_fn (Callable): a function that transforms extrinsic
            immediate rewards
          debug_summaries: True if debug summaries should be created.
        """

        icm_state_spec = ()
        if intrinsic_curiosity_module is not None:
            icm_state_spec = intrinsic_curiosity_module.train_state_spec

        super(ActorCriticAlgorithm, self).__init__(
            action_spec=action_spec,
            predict_state_spec=actor_network.state_spec,
            train_state_spec=ActorCriticState(
                actor_state=actor_network.state_spec,
                value_state=value_network.state_spec,
                icm_state=icm_state_spec),
            action_distribution_spec=actor_network.output_spec,
            optimizer=optimizer,
            gradient_clipping=gradient_clipping,
            reward_shaping_fn=reward_shaping_fn,
            train_step_counter=train_step_counter,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._value_network = value_network
        self._encoding_network = encoding_network
        self._intrinsic_reward_coef = intrinsic_reward_coef
        self._extrinsic_reward_coef = extrinsic_reward_coef
        if loss is None:
            loss = ActorCriticLoss(
                action_spec, debug_summaries=debug_summaries)
        self._loss = loss
        self._icm = intrinsic_curiosity_module

    def _encode(self, time_step: ActionTimeStep):
        observation = time_step.observation
        if self._encoding_network is not None:
            observation, _ = self._encoding_network(observation)
        return observation

    def predict(self, time_step: ActionTimeStep, state=None):
        observation = self._encode(time_step)
        action_distribution, actor_state = self._actor_network(
            observation, step_type=time_step.step_type, network_state=state)
        return PolicyStep(
            action=action_distribution, state=actor_state, info=())

    def train_step(self, time_step: ActionTimeStep, state=None):
        observation = self._encode(time_step)

        value, value_state = self._value_network(
            observation,
            step_type=time_step.step_type,
            network_state=state.value_state)

        action_distribution, actor_state = self._actor_network(
            observation,
            step_type=time_step.step_type,
            network_state=state.actor_state)

        if self._icm is not None:
            icm_step = self._icm.train_step((observation, time_step.action),
                                            state=state.icm_state)
            info = ActorCriticInfo(
                value=value,
                icm_reward=icm_step.outputs,
                icm_info=icm_step.info)
            icm_state = icm_step.state
        else:
            info = ActorCriticInfo(value=value, icm_reward=(), icm_info=())
            icm_state = ()

        state = ActorCriticState(
            actor_state=actor_state,
            value_state=value_state,
            icm_state=icm_state)

        return PolicyStep(action=action_distribution, state=state, info=info)

    def calc_loss(self, training_info, final_time_step, final_policy_step):
        if self._icm is not None:
            self.add_reward_summary("training_reward/intrinsic",
                                    training_info.info.icm_reward)

            reward_calc_fn = lambda extrinsic, intrinsic: (
                    self._extrinsic_reward_coef * extrinsic +
                    self._intrinsic_reward_coef * intrinsic)

            training_info = training_info._replace(
                reward=reward_calc_fn(training_info.reward,
                                      training_info.info.icm_reward))
            final_time_step = final_time_step._replace(
                reward=reward_calc_fn(final_time_step.reward,
                                      final_policy_step.info.icm_reward))

            self.add_reward_summary("training_reward/overall",
                                    training_info.reward)


        final_value = final_policy_step.info.value
        ac_loss = self._loss(training_info, training_info.info.value,
                             final_time_step, final_value)

        if self._icm is not None:
            icm_loss = self._icm.calc_loss(training_info.info.icm_info)
            loss = ac_loss.loss + icm_loss.loss
            icm_loss_extra = icm_loss.extra
        else:
            icm_loss_extra = ()
            loss = ac_loss.loss

        return LossInfo(
            loss=loss,
            extra=ActorCriticAlgorithmLossInfo(
                ac=ac_loss.extra, icm=icm_loss_extra))
