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
"""Actor critic algorithm."""

import gin

from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.data_structures import TimeStep, AlgStep, namedtuple
from alf.utils import dist_utils

ActorCriticState = namedtuple(
    "ActorCriticState", ["actor", "value"], default_value=())

ActorCriticInfo = namedtuple("ActorCriticInfo",
                             ["action_distribution", "value"])


@gin.configurable
class ActorCriticAlgorithm(RLAlgorithm):
    """Actor critic algorithm."""

    def __init__(self,
                 observation_spec,
                 action_spec,
                 env,
                 config,
                 actor_network: ActorDistributionNetwork,
                 value_network: ValueNetwork,
                 loss=None,
                 loss_class=ActorCriticLoss,
                 optimizer=None,
                 debug_summaries=False,
                 name="ActorCriticAlgorithm"):
        """Create an ActorCriticAlgorithm.

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            env ():
            config ():
            actor_network : A network that returns nested
                tensor of action distribution for each observation given observation
                and network state.
            value_network: A function that returns value tensor from neural
                net predictions for each observation given observation and network
                state.
            loss (None|ActorCriticLoss): an object for calculating loss. If
                None, a default loss of class loss_class will be used.
            loss_class (type): the class of the loss. The signature of its
                constructor: loss_class(action_spec, debug_summaries)
            optimizer (torch.optim.Optimizer): The optimizer for training
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
            """
        super(ActorCriticAlgorithm, self).__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            predict_state_spec=ActorCriticState(
                actor=actor_network.state_spec),
            train_state_spec=ActorCriticState(
                actor=actor_network.state_spec,
                value=value_network.state_spec),
            env=env,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._actor_network = actor_network
        self._value_network = value_network
        if loss is None:
            loss = loss_class(debug_summaries=debug_summaries)
        self._loss = loss

    def convert_train_state_to_predict_state(self, state):
        return state._replace(value=())

    def is_on_policy(self):
        return True

    def predict(self, time_step: TimeStep, state: ActorCriticState,
                epsilon_greedy):
        """Predict for one step."""
        action_dist, actor_state = self._actor_network(
            time_step.observation, state=state.actor)

        action = dist_utils.epsilon_greedy_sample(action_dist, epsilon_greedy)
        return AlgStep(
            output=action,
            state=ActorCriticState(actor=actor_state),
            info=ActorCriticInfo(action_distribution=action_dist))

    def rollout(self, time_step: TimeStep, state: ActorCriticState, mode):
        """Rollout for one step."""
        value, value_state = self._value_network(
            time_step.observation, state=state.value)

        action_distribution, actor_state = self._actor_network(
            time_step.observation, state=state.actor)

        action = dist_utils.sample_action_distribution(action_distribution)
        return AlgStep(
            output=action,
            state=ActorCriticState(actor=actor_state, value=value_state),
            info=ActorCriticInfo(
                value=value, action_distribution=action_distribution))

    def calc_loss(self, training_info):
        """Calculate loss."""
        return self._loss(training_info, training_info.info.value)
