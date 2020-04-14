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
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.data_structures import Experience, StepType, TimeStep, TrainingInfo
from alf.utils.summary_utils import record_time


class OnPolicyAlgorithm(OffPolicyAlgorithm):
    """OnPolicyAlgorithm implements the basic on-policy training procedure.

    User needs to implement ``rollout_step()`` and ``calc_loss()``.

    ``rollout_step()`` is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    ``update_with_gradient()`` is called every ``unroll_length`` steps (specified in
    ``config.TrainerConfig``). All the training information collected by every
    ``rollout_step()`` are batched and provided as arguments for
    ``calc_loss()``.

    The following is the pseudo code to illustrate how ``OnPolicyAlgorithm`` can
    be used:

    .. code-block:: python

        for _ in range(unroll_length):
            policy_step = rollout_step(time_step, policy_step.state)
            action = sample action from policy_step.action
            collect necessary information and policy_step.info into training_info
            time_step = env.step(action)
        loss = calc_loss(training_info)
        update_with_gradient(loss)
    """

    def is_on_policy(self):
        return True

    # Implement train_step() to allow off-policy training for an
    # OnPolicyAlgorithm
    def train_step(self, exp: Experience, state):
        time_step = TimeStep(
            step_type=exp.step_type,
            reward=exp.reward,
            discount=exp.discount,
            observation=exp.observation,
            prev_action=exp.prev_action,
            env_id=exp.env_id)
        return self.rollout_step(time_step, state)

    def _train_iter_on_policy(self):
        """User may override this for their own training procedure."""
        alf.summary.get_global_counter().add_(1)

        with record_time("time/unroll"):
            training_info = self.unroll(self._config.unroll_length)
            self.summarize_metrics()

        with record_time("time/train"):
            training_info = training_info._replace(
                rollout_info=(), info=training_info.rollout_info)
            steps = self.train_from_unroll(training_info)

        with record_time("time/after_train_iter"):
            self.after_train_iter(training_info)

        return steps

    def train_from_unroll(self, training_info):
        """Train given the info collected from ``unroll()``. This function can
        be called by any child algorithm that doesn't have the unroll logic but
        has a different training logic with its parent (e.g., off-policy).

        Args:
            training_info (TrainingInfo):

        Returns:
            int: number of steps that have been trained
        """
        valid_masks = (training_info.step_type != StepType.LAST).to(
            torch.float32)
        loss_info, params = self.update_with_gradient(
            self.calc_loss(training_info), valid_masks)
        self.after_update(training_info)
        self.summarize_train(training_info, loss_info, params)
        return torch.tensor(training_info.step_type.shape).prod()
