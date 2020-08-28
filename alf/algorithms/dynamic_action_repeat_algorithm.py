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
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.data_structures import TimeStep, Experience, namedtuple, AlgStep
from alf.data_structures import make_experience, LossInfo
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.conditional_ops import conditional_update
from alf.utils import common

ActionRepeatState = namedtuple(
    "PeriodicActionState",
    ["rl", "action", "steps", "rl_discount", "rl_reward"],
    default_value=())


@gin.configurable
class DynamicActionRepeatAlgorithm(Algorithm):
    """Create an algorithm which learns a variable action repetition duration.
    At each decision step, the algorithm outputs both the action to repeat and
    the number of steps to repeat. These two quantities together constitute the
    action of the algorithm. We use SAC with mixed action type for training.

    The core idea is similar to `Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning <http://arxiv.org/abs/1702.06054>`_.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 config: TrainerConfig,
                 K=5,
                 rl_algorithm_cls=SacAlgorithm,
                 gamma=0.99,
                 optimizer=None,
                 debug_summaries=False,
                 name="DynamicActionRepeatAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions; can
                only be continuous actions for now.
            config (TrainerConfig): config for training. ``config`` only needs to
                be provided to the algorithm which performs a training iteration
                by itself.
            K (int): the maiximal repeating times for an action.
            rl_algorithm_cls (Callable): creates an RL algorithm to be augmented
                by this dynamic action repeating ability.
            gamma (float): the reward discount to be applied when accumulating
                ``k`` steps' rewards for a repeated action.
            optimizer (None|Optimizer): The default optimizer for
                training. See comments above for detail.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): name of this algorithm.
        """
        assert action_spec.is_continuous

        self._policy_action_spec = (BoundedTensorSpec(
            shape=(), dtype='int64', maximum=K - 1), action_spec)

        rl = rl_algorithm_cls(
            observation_spec=observation_spec,
            action_spec=self._policy_action_spec,
            config=config,
            debug_summaries=debug_summaries)
        if config:
            rl.set_exp_replayer("uniform",
                                common.get_env().batch_size,
                                config.replay_buffer_length)

        self._action_spec = action_spec
        self._observation_spec = observation_spec
        self._gamma = gamma

        predict_state_spec = ActionRepeatState(
            rl=rl.predict_state_spec,
            action=action_spec,
            steps=TensorSpec(shape=(), dtype='int64'))

        train_state_spec = predict_state_spec._replace(
            rl=rl.train_state_spec,
            rl_discount=TensorSpec(()),
            rl_reward=TensorSpec(()))

        super().__init__(
            train_state_spec=train_state_spec,
            predict_state_spec=predict_state_spec,
            optimizer=optimizer,
            name=name)

        self._rl = rl

    def _trainable_attributes_to_ignore(self):
        return ["_rl"]

    def is_on_policy(self):
        return self._rl.is_on_policy()

    def _should_switch_action(self, time_step: TimeStep, state):
        repeat_last_step = (state.steps == 0)
        return repeat_last_step | time_step.is_first() | time_step.is_last()

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        switch_action = self._should_switch_action(time_step, state)

        @torch.no_grad()
        def _generate_new_action(time_step, state):
            rl_step = self._rl.predict_step(time_step, state.rl,
                                            epsilon_greedy)
            steps, action = rl_step.output
            return ActionRepeatState(
                action=action,
                steps=steps + 1,  # [0, K-1] -> [1, K]
                rl=rl_step.state)

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

    def rollout_step(self, time_step: TimeStep, state):
        switch_action = self._should_switch_action(time_step, state)
        state = state._replace(
            rl_reward=state.rl_reward + state.rl_discount * time_step.reward,
            rl_discount=state.rl_discount * time_step.discount * self._gamma)

        @torch.no_grad()
        def _generate_new_action(time_step, state):
            rl_time_step = time_step._replace(
                reward=state.rl_reward, discount=state.rl_discount)
            rl_step = self._rl.rollout_step(rl_time_step, state.rl)
            # store to replay buffer
            self._rl.observe_for_replay(
                make_experience(rl_time_step, rl_step, state.rl))
            steps, action = rl_step.output
            return ActionRepeatState(
                action=action,
                steps=steps + 1,  # [0, K-1] -> [1, K]
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

    def train_step(self, exp: Experience, state):
        """Do nothing in the training step because the member rl algorithm
        ``self._rl`` has its own training procedure.
        """
        return AlgStep()

    def calc_loss(self, experience, info):
        """Return an empty loss because the member rl algorithm ``self._rl`` has
        its own training procedure.
        """
        return LossInfo()

    def after_train_iter(self, experience, train_info=None):
        """This function calls the training procedure of the member rl algorithm
        ``self._rl``.
        """
        with alf.summary.scope(self.name):
            self._rl.train_from_replay_buffer()
