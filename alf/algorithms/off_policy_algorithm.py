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

import abc
from collections import namedtuple

import tensorflow as tf

from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import StepType

from alf.algorithms.rl_algorithm import ActionTimeStep, RLAlgorithm

Experience = namedtuple("Experience", [
    'step_type', 'reward', 'discount', 'observation', 'prev_action', 'action',
    'info', 'action_distribution'
])


def make_experience(time_step: ActionTimeStep, policy_step: PolicyStep,
                    action_distribution):
    """Make an instance of Experience from ActionTimeStep and PolicyStep."""
    return Experience(
        step_type=time_step.step_type,
        reward=time_step.reward,
        discount=time_step.discount,
        observation=time_step.observation,
        prev_action=time_step.prev_action,
        action=policy_step.action,
        info=policy_step.info,
        action_distribution=action_distribution)


class OffPolicyAlgorithm(RLAlgorithm):
    """
       OffPolicyAlgorithm works with alf.drivers.off_policy_driver to do training

       User needs to implement predict() and train_step().

       predict() is called to generate actions for every environment step.

       train_step() is called to generate necessary information for training.

       The following is the pseudo code to illustrate how OffPolicyAlgorithm is used
       with OffPolicyDriver:

       ```python
        # (1) collect stage
        for _ in range(steps_per_collection):
            # collect experience and store to replay buffer
            policy_step = predict(time_step, policy_step.state)
            experience = make_experience(time_step, policy_step)
            store experience to replay buffer
            action = sample action from policy_step.action
            time_step = env.step(action)

        # (2) train stage
        for _ in range(training_per_collection):
            # sample experiences and perform training
            experiences = sample batch from replay_buffer
            with tf.GradientTape() as tape:
                batched_training_info
                for experience in experiences:
                    policy_step = train_step(experience, state)
                    train_info = make_training_info(info, ...)
                    write train_info to batched_training_info
                train_complete(tape, batched_training_info,...)
    ```
    """

    @abc.abstractmethod
    def train_step(self, experience: Experience, state):
        """Perform one step of action and training computation.
        
        Args:
            experience (Experience):
            state (nested Tensor): should be consistent with train_state_spec

        Returns (PolicyStep):
            action (nested tf.distribution): should be consistent with 
                `action_distribution_spec`
            state (nested Tensor): should be consistent with `train_state_spec`
            info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OffPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        pass

    def preprocess_experience(self, experience: Experience):
        """Preprocess experience.

        The shapes of tensors in expererience are assumed to be (B, T, ...)

        Args:
            experience (Experience): original experience
        Returns:
            processed experience
        """
        return experience
