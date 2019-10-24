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


class OnPolicyAlgorithm(OffPolicyAlgorithm):
    """
    OnPolicyAlgorithm works with alf.drivers.on_policy_driver.OnPolicyDriver
    to do training at the time of policy rollout.

    User needs to implement rollout() and train_complete().

    rollout() is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    train_complete() is called every `train_interval` steps (specified in
    OnPolicyDriver). All the training information collected at each previous
    rollout() are batched and provided as arguments for train_complete().

    The following is the pseudo code to illustrate how OnPolicyAlgorithm is used
    by OnPolicyDriver:

    ```python
    with GradientTape as tape:
        for _ in range(train_interval):
            policy_step = rollout(time_step, policy_step.state)
            action = sample action from policy_step.action
            collect necessary information and policy_step.info into training_info
            time_step = env.step(action)
    final_policy_step = rollout(training_info)
    collect necessary information and final_policy_step.info into training_info
    train_complete(tape, training_info)
    ```
    """

    def predict(self, time_step: ActionTimeStep, state=None):
        """Default implementation of predict.

        Subclass may override.
        """
        policy_step = self.rollout(time_step, state)
        return policy_step._replace(info=())

    # Implement train_step() to allow off-policy training for an
    # OnPolicyAlgorithm
    def train_step(self, exp: Experience, state):
        time_step = ActionTimeStep(
            step_type=exp.step_type,
            reward=exp.reward,
            discount=exp.discount,
            observation=exp.observation,
            prev_action=exp.prev_action)
        return self.rollout(time_step, state, with_experience=True)
