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

import gin
import torch
import copy
import math
import numpy as np

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import TimeStep, Experience, namedtuple, AlgStep
from alf.data_structures import make_experience, LossInfo
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.conditional_ops import conditional_update
from alf.utils import common

ActionRepeatState = namedtuple(
    "PeriodicActionState",
    ["rl", "action", "steps", "rl_discount", "rl_reward", "repr"],
    default_value=())


@gin.configurable
class DynamicActionRepeatAgent(OffPolicyAlgorithm):
    """Create an agent which learns a variable action repetition duration.
    At each decision step, the agent outputs both the action to repeat and
    the number of steps to repeat. These two quantities together constitute the
    action of the agent. We use SAC with mixed action type for training.

    The core idea is similar to `Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning <http://arxiv.org/abs/1702.06054>`_.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 env=None,
                 config: TrainerConfig = None,
                 K=5,
                 rl_algorithm_cls=SacAlgorithm,
                 representation_learner_cls=None,
                 gamma=0.99,
                 optimizer=None,
                 debug_summaries=False,
                 name="DynamicActionRepeatAgent"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                only be continuous actions for now.
            env (Environment): The environment to interact with. ``env`` is a
                batched environment, which means that it runs multiple simulations
                simultateously. ``env` only needs to be provided to the root
                algorithm.
            config (TrainerConfig): config for training. ``config`` only needs to
                be provided to the algorithm which performs a training iteration
                by itself.
            K (int): the maiximal repeating times for an action.
            rl_algorithm_cls (Callable): creates an RL algorithm to be augmented
                by this dynamic action repeating ability.
            representation_learner_cls (type): The algorithm class for learning
                the representation. If provided, the constructed learner will
                calculate the representation from the original observation as
                the observation for downstream algorithms such as ``rl_algorithm``.
                We assume that the representation is trained by ``rl_algorithm``.
            gamma (float): the reward discount to be applied when accumulating
                ``k`` steps' rewards for a repeated action.
            optimizer (None|Optimizer): The default optimizer for
                training. See comments above for detail.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this agent.
        """
        assert action_spec.is_continuous, (
            "Only support continuous actions for now!")

        rl_observation_spec = observation_spec

        repr_learner = None
        if representation_learner_cls is not None:
            repr_learner = representation_learner_cls(
                observation_spec=observation_spec,
                action_spec=action_spec,
                debug_summaries=debug_summaries)
            rl_observation_spec = repr_learner.output_spec

        self._rl_action_spec = (BoundedTensorSpec(
            shape=(), dtype='int64', maximum=K - 1), action_spec)
        rl = rl_algorithm_cls(
            observation_spec=rl_observation_spec,
            action_spec=self._rl_action_spec,
            debug_summaries=debug_summaries)

        self._action_spec = action_spec
        self._observation_spec = observation_spec
        self._gamma = gamma

        predict_state_spec = ActionRepeatState(
            rl=rl.predict_state_spec,
            action=action_spec,
            steps=TensorSpec(shape=(), dtype='int64'))

        rollout_state_spec = predict_state_spec._replace(
            rl=rl.rollout_state_spec,
            rl_discount=TensorSpec(()),
            rl_reward=TensorSpec(()))

        train_state_spec = ActionRepeatState(rl=rl.train_state_spec)

        if repr_learner is not None:
            predict_state_spec = predict_state_spec._replace(
                repr=repr_learner.predict_state_spec)
            rollout_state_spec = rollout_state_spec._replace(
                repr=repr_learner.rollout_state_spec)
            train_state_spec = train_state_spec._replace(
                repr=repr_learner.train_state_spec)

        super().__init__(
            observation_spec,
            action_spec,
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            env=env,
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._repr_learner = repr_learner
        self._rl = rl
        self._original_observe_for_replay = self.observe_for_replay
        # Do not observe data at every time step; customized observing
        self.observe_for_replay = lambda exp: None

    @property
    def train_action_spec(self):
        return self._rl_action_spec

    def _should_switch_action(self, time_step: TimeStep, state):
        repeat_last_step = (state.steps == 0)
        return repeat_last_step | time_step.is_first() | time_step.is_last()

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        switch_action = self._should_switch_action(time_step, state)

        @torch.no_grad()
        def _generate_new_action(time_step, state):
            repr_state = ()
            if self._repr_learner is not None:
                repr_step = self._repr_learner.predict_step(
                    time_step, state.repr)
                time_step = time_step._replace(observation=repr_step.output)
                repr_state = repr_step.state

            rl_step = self._rl.predict_step(time_step, state.rl,
                                            epsilon_greedy)
            steps, action = rl_step.output
            return ActionRepeatState(
                action=action,
                steps=steps + 1,  # [0, K-1] -> [1, K]
                rl=rl_step.state,
                repr=repr_state)

        new_state = conditional_update(
            target=state,
            cond=switch_action,
            func=_generate_new_action,
            time_step=time_step,
            state=state)
        new_state = new_state._replace(steps=new_state.steps - 1)

        return AlgStep(
            output=new_state.action,
            state=new_state,
            # plot steps and action when rendering video
            info=dict(action=(new_state.action, new_state.steps)))

    def rollout_step(self, time_step: TimeStep, state: ActionRepeatState):
        switch_action = self._should_switch_action(time_step, state)
        state = state._replace(
            rl_reward=state.rl_reward + state.rl_discount * time_step.reward,
            rl_discount=state.rl_discount * time_step.discount * self._gamma)

        @torch.no_grad()
        def _generate_new_action(time_step, state):
            rl_time_step = time_step._replace(
                reward=state.rl_reward, discount=state.rl_discount)

            observation, repr_state = rl_time_step.observation, ()
            if self._repr_learner is not None:
                repr_step = self._repr_learner.rollout_step(
                    time_step, state.repr)
                observation = repr_step.output
                repr_state = repr_step.state

            rl_step = self._rl.rollout_step(
                rl_time_step._replace(observation=observation), state.rl)
            # store to replay buffer
            self._original_observe_for_replay(
                make_experience(rl_time_step, rl_step, state))
            steps, action = rl_step.output
            return ActionRepeatState(
                action=action,
                steps=steps + 1,  # [0, K-1] -> [1, K]
                repr=repr_state,
                rl=rl_step.state,
                rl_reward=torch.zeros_like(state.rl_reward),
                rl_discount=torch.ones_like(state.rl_discount))

        new_state = conditional_update(
            target=state,
            cond=switch_action,
            func=_generate_new_action,
            time_step=time_step,
            state=state)

        new_state = new_state._replace(steps=new_state.steps - 1)

        return AlgStep(output=new_state.action, state=new_state)

    def train_step(self, rl_exp: Experience, state: ActionRepeatState):
        """Train the underlying RL algorithm ``self._rl``. Because in
        ``self.rollout_step()`` the replay buffer only stores info related to
        ``self._rl``, here we can directly call ``self._rl.train_step()``.

        Args:
            rl_exp (Experience): experiences that have been transformed to be
                learned by ``self._rl``.
            state (ActionRepeatState):
        """
        repr_state = ()
        if self._repr_learner is not None:
            repr_step = self._repr_learner.train_step(rl_exp, state.repr)
            rl_exp = rl_exp._replace(observation=repr_step.output)
            repr_state = repr_step.state

        rl_step = self._rl.train_step(rl_exp, state.rl)
        new_state = ActionRepeatState(rl=rl_step.state, repr=repr_state)
        return rl_step._replace(state=new_state)

    def calc_loss(self, rl_experience, rl_info):
        """Calculate the loss for training ``self._rl``."""
        return self._rl.calc_loss(rl_experience, rl_info)

    def after_update(self, rl_exp, rl_info):
        """Call ``self._rl.after_update()``."""
        self._rl.after_update(rl_exp, rl_info)
