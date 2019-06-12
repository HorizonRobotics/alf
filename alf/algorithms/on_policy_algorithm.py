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
from collections import namedtuple

import tensorflow as tf

from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import StepType
from tf_agents.utils import eager_utils

from alf.algorithms.rl_algorithm import ActionTimeStep, RLAlgorithm


class OnPolicyAlgorithm(RLAlgorithm):
    """
    OnPolicyAlgorithm works with alf.policies.TrainingPolicy to do training
    at the time of policy rollout.

    User needs to implement train_step() and train_complete().
    
    train_step() is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    train_complete() is called every `train_interval` steps (specified in
    TrainingPolicy). All the training information collected at each previous
    train_step() are batched and provided as arguments for train_complete().

    The following is the pseudo code to illustrate how OnPolicyAlgoirhtm is used
    by TrainingPolicy:

    ```python
    tape = tf.GradientTape()
    training_info = []
    
    while training not ends:
        if len(training_info) == train_intervel:
            old_tape = tape
            tape = tf.GradientTape()
        with tape:
            policy_step = train_step(time_step, policy_step.state)
        if len(training_info) == train_intervel:
            with old_tape:
                get batched_training_info from training_info
            train_complete(tape, batched_training_info, time_step, policy_step)
            training_info = []
        action = sample action from policy_step.action
        collect necessary information and policy_step.info into training_info
        time_step = env.step(action)
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
            info: everything necessary for training. Note that 
                ("action_distribution", "action", "reward", "discount",
                "is_last") are automatically collected by TrainingPolicy. So
                the user only need to put other stuff (e.g. value estimation)
                into `policy_step.info`
        """
        pass
