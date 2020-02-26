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

import torch

import alf
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.data_structures import Experience, StepType, TimeStep, TrainingInfo


class OnPolicyAlgorithm(RLAlgorithm):
    """
    OnPolicyAlgorithm works with alf.drivers.on_policy_driver.OnPolicyDriver
    to do training at the time of policy rollout.

    User needs to implement rollout_step() and calc_loss()

    rollout_step() is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    update_with_gradient() is called every `unroll_length` steps (specified in
    config.TrainerConfig). All the training information collected at each previous
    rollout_step() are batched and provided as arguments for calc_loss()

    The following is the pseudo code to illustrate how OnPolicyAlgorithm is used:

    ```python
    for _ in range(unroll_length):
        policy_step = rollout_step(time_step, policy_step.state)
        action = sample action from policy_step.action
        collect necessary information and policy_step.info into training_info
        time_step = env.step(action)
    loss = calc_loss(training_info)
    update_with_gradient(loss)
    ```
    """

    def is_on_policy(self):
        return True

    def _train_iter_on_policy(self):
        """User may override this for their own training procedure."""
        training_info = self.unroll(self._config.unroll_length)
        training_info = training_info._replace(
            rollout_info=(), info=training_info.rollout_info)
        valid_masks = (training_info.step_type != StepType.LAST).to(
            torch.float32)
        loss_info, params = self.update_with_gradient(
            self.calc_loss(training_info), valid_masks)
        self.after_update(training_info)
        self.summarize_train(training_info, loss_info, params)
        self.summarize_metrics()
        alf.summary.get_global_counter().add_(1)
        return torch.tensor(training_info.step_type.shape).prod()
