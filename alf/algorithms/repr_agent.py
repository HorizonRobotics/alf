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
"""Agent that explicitly learns a latent state space to be used by RL algorithms.
"""

import gin

import torch

import alf
from alf.data_structures import namedtuple
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.agent_helpers import AgentHelper
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.recurrent_state_space_model import RecurrentStateSpaceModel
from alf.data_structures import TimeStep, AlgStep, Experience

AgentState = namedtuple("AgentState", ["rl", "repr"], default_value=())

AgentInfo = namedtuple("AgentInfo", ["rl", "repr"], default_value=())


@gin.configurable
class RepresentationAgent(OnPolicyAlgorithm):
    """RepresentationAgent is an agent that combines a latent state
    representation learning algorithm with an RL algorithm. The RL algorithm
    will directly compute on the latent state representation.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 env=None,
                 config: TrainerConfig = None,
                 rl_algorithm_cls=ActorCriticAlgorithm,
                 state_representation_model_cls=RecurrentStateSpaceModel,
                 optimizer=None,
                 debug_summaries=False,
                 name="RepresentationAgent"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple
                simulations simultaneously. Running multiple environments in
                parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation. ``env`` only
                needs to be provided to the root ``Algorithm``.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            rl_algorithm_cls (type): The algorithm class for learning the policy.
            state_representation_model_cls (type): The representation model class.
            optimizer (Optimizer): The optimizer for training the agent
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """
        agent_helper = AgentHelper(AgentState)

        # 1. representation model
        repr_model = state_representation_model_cls(
            observation_spec=observation_spec, action_spec=action_spec)
        agent_helper.register_algorithm(repr_model, "repr")

        # 2. rl algorithm
        rl_algorithm = rl_algorithm_cls(
            observation_spec=repr_model.state_spec,
            action_spec=action_spec,
            debug_summaries=debug_summaries)
        agent_helper.register_algorithm(rl_algorithm, "rl")
        # Whether the agent is on-policy or not depends on its rl algorithm.
        self._is_on_policy = rl_algorithm.is_on_policy()

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            optimizer=optimizer,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name,
            **agent_helper.state_specs())

        self._rl_algorithm = rl_algorithm
        self._repr_model = repr_model
        self._agent_helper = agent_helper

    def is_on_policy(self):
        return self._is_on_policy

    def predict_step(self, time_step: TimeStep, state: AgentState,
                     epsilon_greedy):
        """Predict for one step."""
        new_state = AgentState()
        repr_step = self._repr_model.predict_step(
            inputs=(time_step.prev_action, time_step.observation),
            state=state.repr)
        new_state = new_state._replace(repr=repr_step.state)

        rl_step = self._rl_algorithm.predict_step(
            time_step._replace(observation=repr_step.output), state.rl,
            epsilon_greedy)
        new_state = new_state._replace(rl=rl_step.state)

        return AlgStep(output=rl_step.output, state=new_state, info=())

    def rollout_step(self, time_step: TimeStep, state: AgentState):
        """Rollout for one step."""
        new_state = AgentState()
        info = AgentInfo()

        repr_step = self._repr_model.rollout_step(
            inputs=(time_step.prev_action, time_step.observation),
            state=state.repr)
        new_state = new_state._replace(repr=repr_step.state)

        rl_step = self._rl_algorithm.rollout_step(
            time_step._replace(observation=repr_step.output), state.rl)
        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def train_step(self, exp: Experience, state):
        new_state = AgentState()
        info = AgentInfo()

        repr_step = self._repr_model.train_step(
            inputs=(exp.prev_action, exp.observation, exp.reward),
            state=state.repr)
        new_state = new_state._replace(repr=repr_step.state)
        info = info._replace(repr=repr_step.info)

        rl_step = self._rl_algorithm.train_step(
            exp._replace(observation=repr_step.output), state.rl)
        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def calc_loss(self, exp: Experience, train_info: AgentInfo):
        return self._agent_helper.accumulate_loss_info(
            [self._repr_model, self._rl_algorithm], exp, train_info)

    def after_update(self, exp: Experience, train_info: AgentInfo):
        """Call ``after_update()`` of the RL algorithm and goal generator,
        respectively.
        """
        self._agent_helper.after_update([self._repr_model, self._rl_algorithm],
                                        exp, train_info)

    def after_train_iter(self, exp: Experience, train_info: AgentInfo = None):
        """Call ``after_train_iter()`` of the RL algorithm and goal generator,
        respectively.
        """
        self._agent_helper.after_train_iter(
            [self._repr_model, self._rl_algorithm], exp, train_info)

    def preprocess_experience(self, exp: Experience):
        """Call ``preprocess_experience()`` of the rl algorithm."""
        new_exp = self._rl_algorithm.preprocess_experience(
            exp._replace(rollout_info=exp.rollout_info.rl))
        return new_exp._replace(
            rollout_info=exp.rollout_info._replace(rl=new_exp.rollout_info))
