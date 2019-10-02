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
    'info', 'action_distribution', 'state'
])


def make_experience(time_step: ActionTimeStep, policy_step: PolicyStep,
                    action_distribution, state):
    """Make an instance of Experience from ActionTimeStep and PolicyStep."""
    return Experience(
        step_type=time_step.step_type,
        reward=time_step.reward,
        discount=time_step.discount,
        observation=time_step.observation,
        prev_action=time_step.prev_action,
        action=policy_step.action,
        info=policy_step.info,
        action_distribution=action_distribution,
        state=state)


class OffPolicyAlgorithm(RLAlgorithm):
    """
       OffPolicyAlgorithm works with alf.drivers.off_policy_driver to do training

       User needs to implement rollout() and train_step().

       rollout() is called to generate actions for every environment step.

       train_step() is called to generate necessary information for training.

       The following is the pseudo code to illustrate how OffPolicyAlgorithm is used
       with OffPolicyDriver:

       ```python
        # (1) collect stage
        for _ in range(steps_per_collection):
            # collect experience and store to replay buffer
            policy_step = rollout(time_step, policy_step.state)
            experience = make_experience(time_step, policy_step)
            store experience to replay buffer
            action = sample action from policy_step.action
            time_step = env.step(action)

        # (2) train stage
        for _ in range(training_per_collection):
            # sample experiences and perform training
            experiences = sample batch from replay_buffer
            with tf.GradientTape() as tape:
                batched_training_info = []
                for experience in experiences:
                    policy_step = train_step(experience, state)
                    train_info = make_training_info(info, ...)
                    write train_info to batched_training_info
                train_complete(tape, batched_training_info,...)
    ```
    """

    def predict(self, time_step: ActionTimeStep, state=None):
        """Default implementation of predict.

        Subclass may override.
        """
        policy_step = self._rollout_partial_state(time_step, state)
        return policy_step._replace(info=())

    def rollout(self,
                time_step: ActionTimeStep,
                state=None,
                with_experience=False):
        """Base implementation of rollout for OffPolicyAlgorithm.

        Calls _rollout_full_state or _rollout_partial_state based on
        use_rollout_state.

        Subclass may override.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec
            with_experience (bool): a boolean flag indicating whether the current
                rollout is with sampled experiences or not. By default this flag
                is ignored. See ActorCriticAlgorithm's rollout for an example of
                usage to avoid computing intrinsic rewards if
                `with_experience=True`.
        Returns:
            policy_step (PolicyStep):
              action (nested tf.distribution): should be consistent with
                `action_distribution_spec`
              state (nested Tensor): should be consistent with `train_state_spec`
              info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OnPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        if self._use_rollout_state and self._is_rnn:
            return self._rollout_full_state(time_step, state)
        else:
            return self._rollout_partial_state(time_step, state)

    def _rollout_partial_state(self, time_step: ActionTimeStep, state=None):
        """Rollout without the full state for train_step().

        It is used for non-RNN model or RNN model without computating all states
        in train_state_spec. In the returned PolicyStep.state, you can use an
        empty tuple as a placeholder for those states that are not necessary for
        rollout.

        User needs to override this if _rollout_full_state() is not implemented.
        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec
        Returns:
            policy_step (PolicyStep):
              action (nested tf.distribution): should be consistent with
                `action_distribution_spec`
              state (nested Tensor): should be consistent with `train_state_spec`.
              info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OnPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        return self._rollout_full_state(time_step, state)

    def _rollout_full_state(self, time_step: ActionTimeStep, state=None):
        """Rollout with full state for train_step().

        If you want to use the rollout state for off-policy training (by setting
        TrainerConfig.use_rollout=True), you need to implement this function.
        You need to compute all the states for the returned PolicyStep.state.

        Args:
            time_step (ActionTimeStep):
            state (nested Tensor): should be consistent with train_state_spec
        Returns:
            policy_step (PolicyStep):
              action (nested tf.distribution): should be consistent with
                `action_distribution_spec`
              state (nested Tensor): should be consistent with `train_state_spec`.
              info (nested Tensor): everything necessary for training. Note that
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by OnPolicyDriver. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        raise NotImplementedError("_rollout_full_state is not implemented")

    @abc.abstractmethod
    def train_step(self, experience: Experience, state):
        """Perform one step of training computation.

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

        The shapes of tensors in experience are assumed to be (B, T, ...)

        Args:
            experience (Experience): original experience
        Returns:
            processed experience
        """
        return experience
