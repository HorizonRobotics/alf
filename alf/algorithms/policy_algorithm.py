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
from collections import namedtuple

import numpy as np
import tensorflow as tf

from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import StepType
from tf_agents.utils import eager_utils

import alf.utils.common as common
from alf.drivers.policy_driver import ActionTimeStep


class PolicyAlgorithm(tf.Module):
    def __init__(self,
                 action_spec,
                 train_state_spec,
                 action_distribution_spec,
                 predict_state_spec=None,
                 optimizer=None,
                 gradient_clipping=None,
                 train_step_counter=None,
                 debug_summaries=False,
                 name="PolicyAlgorithm"):
        """

        Args:
            action_spec (nested BoundedTensorSpec): representing the actions.
            train_state_spec (nested TensorSpec): for the network state of
                `train_step()`
            action_distribution_spec (nested DistributionSpec): for the action
                distributions.
            predict_state_spec (nested TensorSpec): for the network state of
                `train_step()`. If None, it's assume to be same as
                 train_state_spec
            optimizer (tf.optimizers.Optimizer): The optimizer for training.
            train_step_counter (tf.Variable): An optional counter to increment
                every time the a new iteration is started. If None, it will use
                tf.summary.experimental.get_step(). If this is still None, a
                counter will be created.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """

        super(PolicyAlgorithm, self).__init__(name=name)

        self._action_spec = action_spec
        self._train_state_spec = train_state_spec
        if predict_state_spec is None:
            predict_state_spec = train_state_spec
        self._predict_state_spec = predict_state_spec
        self._action_distribution_spec = action_distribution_spec
        self._optimizer = optimizer
        self._gradient_clipping = gradient_clipping
        self._train_step_counter = common.get_global_counter(
            train_step_counter)
        self._debug_summaries = debug_summaries
        self._trainable_variables = None
        self._cached_vars = None

    def add_reward_summary(self, name, rewards):
        if self._debug_summaries:
            step = self._train_step_counter
            tf.summary.histogram(name + "/value", rewards, step)
            tf.summary.scalar(name + "/mean", tf.reduce_mean(rewards), step)

    def predict(self, time_step: ActionTimeStep, state=None):
        """Predict for one step of observation.

        Returns:
            policy_step (PolicyStep):
              policy_step.action is nested tf.distribution which consistent with
                `action_distribution_spec`
              policy_step.state should be consistent with `predict_state_spec`

        """
        policy_step = self.train_step(time_step, state)
        return policy_step._replace(info=())

    @abstractmethod
    def train_step(self, time_step, state):
        pass

    @abstractmethod
    def train_complete(self):
        pass

    @property
    def action_spec(self):
        """Return the action spec."""
        return self._action_spec

    @property
    def action_distribution_spec(self):
        """Return the action distribution spec for the action distributions."""
        return self._action_distribution_spec

    @property
    def predict_state_spec(self):
        """Return the RNN state spec for predict()."""
        return self._predict_state_spec

    @property
    def train_state_spec(self):
        """Return the RNN state spec for train_step()."""
        return self._train_state_spec


