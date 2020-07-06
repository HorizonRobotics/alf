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
"""Base class for off policy algorithms."""

import torch

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.utils.summary_utils import record_time


class OffPolicyAlgorithm(RLAlgorithm):
    """``OffPolicyAlgorithm`` implements basic off-policy training pipeline. User
    needs to implement ``rollout_step()`` and ``train_step()``.
    - ``rollout_step()`` is called to generate actions at every environment step.
    - ``train_step()`` is called to generate necessary information for training.

    The following is the pseudo code to illustrate how ``OffPolicyAlgorithm``
    is used:

    .. code-block:: python

        # (1) collect stage
        for _ in range(steps_per_collection):
            # collect experience and store to replay buffer
            policy_step = rollout_step(time_step, policy_step.state)
            experience = make_experience(time_step, policy_step)
            store experience to replay buffer
            action = sample action from policy_step.action
            time_step = env.step(action)

        # (2) train stage
        for _ in range(training_steps_per_collection):
            # sample experiences and perform training
            experiences = sample batch from replay_buffer
            batched_train_info = []
            for experience in experiences:
                policy_step = train_step(experience, state)
                add policy_step.info to batched_train_info
            loss = calc_loss(experiences, batched_train_info)
            update_with_gradient(loss)
    """

    def is_on_policy(self):
        return False

    def _train_iter_off_policy(self):
        """User may override this for their own training procedure."""
        config: TrainerConfig = self._config

        if not config.update_counter_every_mini_batch:
            alf.summary.increment_global_counter()

        with torch.set_grad_enabled(config.unroll_with_grad):
            with record_time("time/unroll"):
                experience = self.unroll(config.unroll_length)
                self.summarize_rollout(experience)
                self.summarize_metrics()

        steps = self.train_from_replay_buffer(update_global_counter=True)

        with record_time("time/after_train_iter"):
            train_info = experience.rollout_info
            experience = experience._replace(rollout_info=())
            if config.unroll_with_grad:
                self.after_train_iter(experience, train_info)
            else:
                self.after_train_iter(experience)  # only off-policy training

        # For now, we only return the steps of the primary algorithm's training
        return steps
