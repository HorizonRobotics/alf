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

import gin.tf

import tensorflow as tf

from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.networks.network import Network, DistributionNetwork

from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.policies.policy_training_info import TrainingInfo

ActorCriticState = namedtuple("ActorCriticPolicyState",
                              ["actor_state", "value_state"])


class ActorCriticAlgorithm(OnPolicyAlgorithm):
    def __init__(self,
                 action_spec,
                 actor_network: DistributionNetwork,
                 value_network: Network,
                 loss=None,
                 optimizer=None):
        """Create an ActorCriticAlgorithm

        Args:
          action_spec: A nest of BoundedTensorSpec representing the actions.
          actor_network (DistributionNetwork): A network that returns nested
            tensor of action distribution for each observation given observation
            and network state.
          value_net (Network): A function that returns value tensor from neural
            net predictions for each observation given observation and nwtwork
            state.
          loss (None|ActorCriticLoss): an object for calculating loss. If None,
            a default ActorCriticLoss will be used.
          optimizer (tf.optimizers.Optimizer): The optimizer for training.
        """

        self._actor_network = actor_network
        self._value_network = value_network
        self._loss = ActorCriticLoss(action_spec) if loss is None else loss

        super(ActorCriticAlgorithm, self).__init__(
            action_spec=action_spec,
            predict_state_spec=self._actor_network.state_spec,
            train_state_spec=ActorCriticState(
                actor_state=actor_network.state_spec,
                value_state=value_network.state_spec),
            action_distribution_spec=actor_network.output_spec,
            optimizer=optimizer)

    def _variables(self):
        return self._actor_network.variables + self._value_network.variables

    def predict(self, time_step: TimeStep, state=None):
        action_distribution, actor_state = self._actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=state.actor_state)
        return PolicyStep(
            action=action_distribution, state=actor_state, info=())

    def train_step(self, time_step: TimeStep, state=None):
        value, value_state = self._value_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=state.value_state)

        action_distribution, actor_state = self._actor_network(
            time_step.observation,
            step_type=time_step.step_type,
            network_state=state.actor_state)

        state = ActorCriticState(
            actor_state=actor_state, value_state=value_state)

        return PolicyStep(action=action_distribution, state=state, info=value)

    def calc_loss(self, training_info, final_time_step, final_policy_step):
        final_value = final_policy_step.info
        return self._loss(training_info, training_info.info, final_value)
