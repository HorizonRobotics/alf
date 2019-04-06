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

import gin.tf

from tf_agents.policies.boltzmann_policy import BoltzmannPolicy
from tf_agents.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from tf_agents.policies.greedy_policy import GreedyPolicy
from tf_agents.policies.q_policy import QPolicy
from tf_agents.environments.trajectory import Trajectory
from tf_agents.environments.time_step import StepType

from alf.algorithms import SimpleAlgorithm, LossStep
from alf.utils import losses, common


@gin.configurable
class DQNAlgorithm(SimpleAlgorithm):
    def __init__(
            self,
            model,
            optimizer,
            epsilon_greedy=0.1,
            boltzmann_temperature=None,
            # Params for target network updates
            target_update_tau=1.0,
            target_update_period=1,
            # Params for training.
            gamma=1.0,
            nstep_reward=1,
    ):
        """Creates a DQN Algorithm.

        Args:
          model (alf.RLModel): The model used by this algorithm. The model 
            The output of the model should be nested Tensor with each Tensor
            correspond to one action in action_spec.
          optimizer: The optimizer to use for training.
          epsilon_greedy: probability of choosing a random action in the default
            epsilon-greedy collect policy (used only if a wrapper is not
            provided to the collect_policy method).
          boltzmann_temperature: Temperature value to use for Boltzmann sampling
            of the actions during data collection. The closer to 0.0, the higher
            the probability of choosing the best action.
          target_update_tau: Factor for soft update of the target networks.
          target_update_period: Period for soft update of the target networks.
          gamma: A discount factor for future rewards.
          nstep_reward (int): aggregate n steps of future reward to get target
            Q value
        """
        self._target_model = model.copy(name='TargetQModel')
        self._epsilon_greedy = epsilon_greedy
        self._boltzmann_temperature = boltzmann_temperature
        self._optimizer = optimizer
        self._gamma = gamma
        self._nstep_reward = nstep_reward
        self._update_target = common.get_target_updater(
            model, self._target_model, target_update_tau, target_update_period)

        policy = QPolicy(
            model.time_step_spec, model.action_spec, q_network=model)

        if boltzmann_temperature is not None:
            collect_policy = BoltzmannPolicy(
                policy, temperature=self._boltzmann_temperature)
        else:
            collect_policy = EpsilonGreedyPolicy(
                policy, epsilon=self._epsilon_greedy)
        policy = GreedyPolicy(policy)

        super(DQNAlgorithm, self).__init__(model, policy, collect_policy,
                                           optimizer)

    def pre_critic_step(self, next_trajectory: Trajectory, state):
        return self._target_model(next_trajectory.observation, state)

    def critic(self, trajectory: Trajectory, next_trajectory: Trajectory,
               pre_critic):

        # dicount is 0 if next_trajectory.step_type is LAST.
        # see time_step.termination()
        discount = next_trajectory.discount * self._gamma

        if self._nstep_reward > 1:

            def _calc_return(reward, value):
                return losses.calc_nstep_return(reward, value, discount,
                                                next_trajectory.step_type,
                                                self._nstep_reward)

            target_value = tf.nest.map_structure(_calc_return,
                                                 trajectory.reward, pre_critic)
        else:
            target_value = tf.nest.map_structure(lambda r, v: r + discount * v,
                                                 trajectory.reward, pre_critic)

        return target_value

    def loss_step(self, input: LossStep, state=None):
        q_values, next_state = self._model(input.trajectory.observation, state)
        loss = tf.nest.map_structure(losses.calc_q_loss, q_values,
                                     input.trajectory.action, input.critic)
        return loss, next_state
