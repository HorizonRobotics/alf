# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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

import gin.tf
import tensorflow as tf
import numpy as np
import functools

import tf_agents.specs.tensor_spec as tensor_spec
from tf_agents.trajectories.time_step import StepType

from alf.algorithms.algorithm import Algorithm, AlgorithmStep
from alf.data_structures import TrainingInfo, namedtuple

import alf.utils.common as common

GoalState = namedtuple("GoalState", ["goal"], default_value=())
GoalInfo = namedtuple("GoalInfo", ["goal"], default_value=())


@gin.configurable
class RandomCategoricalGoalGenerator(Algorithm):
    """Random Goal Generation Module

    This module generates a random categorical goal for the agent
    in the beginning of every episode.
    """

    def __init__(self, num_of_goals, name="RandomCategoricalGoalGenerator"):
        """Create a RandomCategoricalGoalGenerator.

        Args:
            num_of_goals (int): total number of goals the agent can sample from
        """
        goal_spec = tf.TensorSpec((num_of_goals, ))
        train_state_spec = GoalState(goal=goal_spec)
        super().__init__(train_state_spec=train_state_spec, name=name)
        self._num_of_goals = num_of_goals
        self._p_goal = tf.ones(self._num_of_goals)

    def _generate_goal(self, observation, state):
        """Generate a goal
        """
        batch_size = tf.shape(tf.nest.flatten(observation)[0])[0]
        # generate goal with the same batch size as observation
        samples = tf.random.categorical(
            tf.math.log([self._p_goal]), batch_size)
        samples_onehot = tf.one_hot(indices=samples, depth=self._num_of_goals)
        samples_onehot = tf.reshape(samples_onehot, [batch_size, -1])
        return samples_onehot

    def _update_goal(self, observation, state, step_type):
        new_goal_mask = (step_type == StepType.FIRST)
        new_goal = common.conditional_update(
            target=state.goal,
            cond=new_goal_mask,
            func=self._generate_goal,
            observation=observation,
            state=state)
        return new_goal

    def predict(self, observation, state, step_type):
        """Predict one step for goal generation."""
        new_goal = self._update_goal(observation, state, step_type)
        return AlgorithmStep(
            outputs=new_goal,
            state=GoalState(goal=new_goal),
            info=GoalInfo(goal=new_goal))

    def train_step(self, inputs, state):
        """Perform one step of predicting and training computation.

        Note that as RandomCategoricalGoalGenerator is a non-trainable module,
        this function just passes the goal from state as outputs and
        the input state as output state.

        Args:
            inputs (nested Tensor): inputs for train
            state (nested Tensor): consistent with train_state_spec
        Returns:
            TrainStep:
                outputs: goal vector; currently just output the one from state
                state: state
                info (GoalInfo):
        """
        return AlgorithmStep(
            outputs=(state.goal), state=state, info=GoalInfo())
