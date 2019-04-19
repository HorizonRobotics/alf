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

from abc import abstractmethod

import tensorflow as tf
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from alf.policies.policy_training_info import TrainingInfo


class OnPolicyAlgorithm(object):
    def __init__(self, action_spec, predict_state_spec, train_state_spec,
                 action_distribution_spec):

        self._action_spec = action_spec
        self._predict_state_spec = predict_state_spec
        self._train_state_spec = train_state_spec
        self._action_distribution_spec = action_distribution_spec

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def predict_state_spec(self):
        return self._predict_state_spec

    @property
    def train_state_spec(self):
        return self._train_state_spec

    @property
    def action_distribution_spec(self):
        return self._action_distribution_spec

    #------------- User need to implement the following functions -------
    @abstractmethod
    def predict(self, time_step: TimeStep, state=None):
        """Predict for one step of observation
        Returns:
            distribution (nested tf.distribution): action distribution
            state (None | nested tf.Tensor): RNN state
        """
        pass

    @abstractmethod
    def train_step(self, time_step: TimeStep, state=None):
        pass

    @abstractmethod
    def train_complete(self, tape: tf.GradientTape,
                       training_info: TrainingInfo, final_time_step: TimeStep,
                       policy_state):
        pass
