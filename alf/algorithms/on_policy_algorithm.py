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
"""Base class for on-policy RL algorithms."""

from abc import abstractmethod

import tensorflow as tf

from alf.algorithms.rl_algorithm import ActionTimeStep, RLAlgorithm, TrainingInfo
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm, Experience


class OnPolicyAlgorithm(RLAlgorithm):
    """
    OnPolicyAlgorithm works with alf.drivers.on_policy_driver.OnPolicyDriver
    to do training at the time of policy rollout.

    User needs to implement train_step() and train_complete().

    train_step() is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    train_complete() is called every `train_interval` steps (specified in
    OnPolicyDriver). All the training information collected at each previous
    train_step() are batched and provided as arguments for train_complete().

    The following is the pseudo code to illustrate how OnPolicyAlgoirhtm is used
    by OnPolicyDriver:

    ```python
    with GradientTape as tape:
        for _ in range(train_interval):
            policy_step = train_step(time_step, policy_step.state)
            action = sample action from policy_step.action
            collect necessary information and policy_step.info into training_info
            time_step = env.step(action)
    final_policy_step = train_step(training_info)
    collect necessary information and final_policy_step.info into training_info
    train_complete(tape, training_info)
    ```
    """

    # Subclass may override predict() to allow more efficient implementation
    def predict(self, time_step: ActionTimeStep, state=None):
        """Predict for one step of observation.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec

        Returns:
            policy_step (PolicyStep):
              policy_step.action is nested tf.distribution which consistent with
                `action_distribution_spec`
              policy_step.state should be consistent with `predict_state_spec`
        """
        policy_step = self.train_step(time_step, state)
        return policy_step._replace(info=())

    @abstractmethod
    def train_step(self, time_step: ActionTimeStep, state):
        """Perform one step of action and training computation.

        It is called to generate actions for every environment step.
        It also needs to generate necessary information for training.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec

        Returns (PolicyStep):
            action (nested tf.distribution): should be consistent with 
                `action_distribution_spec`
            state (nested Tensor): should be consistent with `train_state_spec`
            info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OnPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        pass


class OffPolicyAdapter(OffPolicyAlgorithm):
    """An adapter to make an on-policy algorithm into an off-policy algorithm."""

    def __init__(self, algorithm: OnPolicyAlgorithm):
        super().__init__(
            action_spec=algorithm.action_spec,
            train_state_spec=algorithm.train_state_spec,
            action_distribution_spec=algorithm.action_distribution_spec,
            predict_state_spec=algorithm.predict_state_spec,
            debug_summaries=algorithm._debug_summaries,
            name=algorithm._name)
        self._algorithm = algorithm

    @property
    def action_spec(self):
        return self._algorithm.action_spec

    @property
    def action_distribution_spec(self):
        return self._algorithm.action_distribution_spec

    @property
    def predict_state_spec(self):
        return self._algorithm.predict_state_spec

    @property
    def train_state_spec(self):
        return self._algorithm.train_state_spec

    def greedy_predict(self, time_step: ActionTimeStep, state=None):
        return self._algorithm.greedy_predict(time_step, state)

    def predict(self, time_step: ActionTimeStep, state=None):
        return self._algorithm.predict(time_step, state)

    def train_complete(self,
                       tape: tf.GradientTape,
                       training_info: TrainingInfo,
                       weight=1.0):
        return self._algorithm.train_complete(tape, training_info, weight)

    def train_step(self, exp: Experience, state):
        time_step = ActionTimeStep(
            step_type=exp.step_type,
            reward=exp.reward,
            discount=exp.discount,
            observation=exp.observation,
            prev_action=exp.prev_action)
        return self._algorithm.train_step(time_step, state)
