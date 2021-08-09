# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Parallel Actor critic algorithm."""

import alf
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.networks import ActorDistributionNetwork, ValueNetwork
from alf.algorithms.parallel_actor_critic_loss import ParallelActorCriticLoss
from alf.data_structures import TimeStep, AlgStep, namedtuple
from alf.utils import common, dist_utils
from alf.tensor_specs import TensorSpec
from .config import TrainerConfig


ParallelActorCriticState = namedtuple(
    "ParallelActorCriticState", ["actors", "values"], default_value=())

ActorCriticInfo = namedtuple(
    "ActorCriticInfo", [
        "step_type", "discount", "reward", "action", "action_distribution",
        "value"
    ],
    default_value=())


@alf.configurable
class ParallelActorCriticAlgorithm(OnPolicyAlgorithm):
    """
    Parallel Actor critic algorithm. 
    
    This algorithm provides a way to maintain n (n should be the same as the number of environments.) different agents, 
    which means every agent has its own actor network and value network. This is different from directly running ActorCriticAlgorithm 
    in n batched environments where only one agent is maintained.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 reward_spec=TensorSpec(()),
                 actor_network_ctor=ActorDistributionNetwork,
                 value_network_ctor=ValueNetwork,
                 epsilon_greedy=None,
                 env=None,
                 config: TrainerConfig = None,
                 loss=None,
                 loss_class=ParallelActorCriticLoss,
                 optimizer=None,
                 debug_summaries=False,
                 name="ParallelActorCriticAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``alf.get_config_value(TrainerConfig.epsilon_greedy)``
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            actor_network_ctor (Callable): Function to construct the actor network.
                ``actor_network_ctor`` needs to accept ``input_tensor_spec`` and
                ``action_spec`` as its arguments and return an actor network.
                The constructed network will be called with ``forward(observation, state)``.
            value_network_ctor (Callable): Function to construct the value network.
                ``value_network_ctor`` needs to accept ``input_tensor_spec`` as
                its arguments and return a value netwrok. The contructed network
                will be called with ``forward(observation, state)`` and returns
                value tensor for each observation given observation and network
                state.
            loss (None|ParallelActorCriticLoss): an object for calculating loss. If
                None, a default loss of class loss_class will be used.
            loss_class (type): the class of the loss. The signature of its
                constructor: ``loss_class(debug_summaries)``
            optimizer (torch.optim.Optimizer): The optimizer for training
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """

        if env is not None:
            assert env.batch_size == config.num_parallel_agents, "The number of environments must be the same as the number of agents!"

        if epsilon_greedy is None:
            epsilon_greedy = alf.get_config_value(
                'TrainerConfig.epsilon_greedy')
        self._epsilon_greedy = epsilon_greedy
        self._num_parallel_agents = config.num_parallel_agents

        value_network = value_network_ctor(input_tensor_spec=observation_spec)
        parallel_value_networks = value_network.make_parallel(self._num_parallel_agents)
    
        actor_network = actor_network_ctor(input_tensor_spec=observation_spec, action_spec=action_spec)
        parallel_actor_networks = actor_network.make_parallel(self._num_parallel_agents)

        super(ParallelActorCriticAlgorithm, self).__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            predict_state_spec=ParallelActorCriticState(
                actors=parallel_actor_networks.state_spec),
            train_state_spec=ParallelActorCriticState(
                actors=parallel_actor_networks.state_spec,
                values=parallel_value_networks.state_spec),
            env=env,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)
        self._actor_networks = parallel_actor_networks
        self._value_networks = parallel_value_networks
        
        if loss is None:
            loss = loss_class(debug_summaries=debug_summaries)
        self._loss = loss

    def convert_train_state_to_predict_state(self, state):
        return state._replace(values=())

    def predict_step(self, inputs: TimeStep, state: ParallelActorCriticState):
        """
        Predict for one step.
        
        When the batch_size of the environments is the same as the number of agents, we treat it as the parallel rollout setting, 
        where each environment is predicted by the corresponding (in indices) agent. Otherwise, for every env, we simply use agent 0 to predict.
        """
        if inputs.observation.shape[0] == self._num_parallel_agents:
            value, value_state = self._value_networks(
            inputs.observation.unsqueeze(0), state=state.values)
            value = value.squeeze(0)

            action_distribution, actor_state = self._actor_networks(
                inputs.observation.unsqueeze(0), state=state.actors)

            action = dist_utils.sample_action_distribution(action_distribution)
            action = action.squeeze(0)
        else:
            value, value_state = self._value_networks(
            inputs.observation, state=state.values)
            value = value[:, 0]

            action_distribution, actor_state = self._actor_networks(
                inputs.observation, state=state.actors)

            action = dist_utils.sample_action_distribution(action_distribution)
            action = action[:, 0]
            action_distribution = ()

        return AlgStep(
            output=action,
            state=ParallelActorCriticState(actors=actor_state, values=value_state),
            info=ActorCriticInfo(
                action=common.detach(action),
                value=value,
                step_type=inputs.step_type,
                reward=inputs.reward,
                discount=inputs.discount,
                action_distribution=action_distribution))

    def rollout_step(self, inputs: TimeStep, state: ParallelActorCriticState):
        """ Rollout for one step.
            The input shape of two networks should be [B, n, d], where B is the batch size of the environments,
            n is the number of parallel agents and d is the shape of observation. Since we require B should be 
            the same as the number of parallel agents, we only need to adapt the shape of inputs.observations to 
            be [1, B, d]. After adaptation, each environment is handled by the corresponding (in indices) agent.
        """
        value, value_state = self._value_networks(
            inputs.observation.unsqueeze(0), state=state.values)
        value = value.squeeze(0)

        action_distribution, actor_state = self._actor_networks(
            inputs.observation.unsqueeze(0), state=state.actors)

        action = dist_utils.sample_action_distribution(action_distribution)
        action = action.squeeze(0)

        return AlgStep(
            output=action,
            state=ParallelActorCriticState(actors=actor_state, values=value_state),
            info=ActorCriticInfo(
                action=common.detach(action),
                value=value,
                step_type=inputs.step_type,
                reward=inputs.reward,
                discount=inputs.discount,
                action_distribution=action_distribution))

    def calc_loss(self, info: ActorCriticInfo):
        """Calculate loss."""
        return self._loss(info)