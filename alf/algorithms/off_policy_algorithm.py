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
    'info'
])


def make_experience(time_step: ActionTimeStep, policy_step: PolicyStep):
    """Make an instance of Experience from ActionTimeStep and PolicyStep."""
    return Experience(
        step_type=time_step.step_type,
        reward=time_step.reward,
        discount=time_step.discount,
        observation=time_step.observation,
        prev_action=time_step.prev_action,
        action=policy_step.action,
        info=policy_step.info)


class OffPolicyAlgorithm(RLAlgorithm):
    """Base class of off-policy algorithms."""

    @abc.abstractmethod
    def predict(self, time_step: ActionTimeStep, state=None):
        """Predict for one step of observation.

        Returns:
            policy_step (PolicyStep):
              policy_step.action is nested tf.distribution which consistent with
                `action_distribution_spec`
              policy_step.state should be consistent with `predict_state_spec`
        """
        pass

    @abc.abstractmethod
    def train_step(self, experience: Experience, state):
        """Perform one step of action and training computation.
        
        Args:
            experience (Experience):
            state (nested Tensor): should be consistent with train_state_spec

        Returns (tuple):
            state: training RNN state
            info: everything necessary for training. Note that 
                ("action", "reward", "discount", "is_last") are automatically
                collected by OffPolicyDriver. So the user only need to put other
                stuff (e.g. value estimation) into `info`
        """
        pass
