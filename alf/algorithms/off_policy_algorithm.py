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

from alf.algorithms.rl_algorithm import RLAlgorithm


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

    @property
    def on_policy(self):
        return False
