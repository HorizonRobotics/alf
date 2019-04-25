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
import numpy as np

import tensorflow as tf
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.specs.tensor_spec import TensorSpec, BoundedTensorSpec
from tf_agents.trajectories.time_step import TimeStep, StepType


class UnittestEnv(TFEnvironment):
    """Abstract base for unittest environment.

    Every episode ends in `episode_length` steps (including LAST step).
    The observation is one dimensional. The action is binary {0, 1}.
    """

    def __init__(self, batch_size, episode_length):
        """Initializes the environment
        Args:
          batch_size (int): The batch size expected for the actions and
             observations.
          episode_length (int): length of each episode
        """
        self._steps = 0
        self._episode_length = episode_length
        super(UnittestEnv, self).__init__(
            action_spec=BoundedTensorSpec(
                shape=(1, ), dtype=tf.int64, minimum=0, maximum=1),
            time_step_spec=TimeStep(
                step_type=TensorSpec(shape=(), dtype=tf.int32),
                reward=TensorSpec(shape=(), dtype=tf.float32),
                discount=TensorSpec(shape=(), dtype=tf.float32),
                observation=TensorSpec(shape=(1, ), dtype=tf.float32)),
            batch_size=batch_size)

    def _current_time_step(self):
        return self.__current_time_step

    def _reset(self):
        self._steps = 0
        self.__current_time_step = self._gen_time_step(0, None)
        return self.__current_time_step

    def _step(self, action):
        self._steps += 1
        self.__current_time_step = self._gen_time_step(
            self._steps % self._episode_length, action)
        return self.__current_time_step

    @abstractmethod
    def _gen_time_step(self, s, action):
        """
        Args:
          s (int): step count in current episode. It is range from 0 to
            `episode_length` - 1
        """
        pass


class ValueUnittestEnv(UnittestEnv):
    """
    Every episode ends in `episode_length` steps. It always give reward
    1 at each step.
    """

    def _gen_time_step(self, s, action):
        """Returns the current `TimeStep`."""
        step_type = StepType.MID
        discount = 1.0

        if s == 0:
            step_type = StepType.FIRST
        elif s == self._episode_length - 1:
            step_type = StepType.LAST
            discount = 0.0

        return TimeStep(
            step_type=tf.constant([step_type] * self.batch_size),
            reward=tf.constant([1.] * self.batch_size),
            discount=tf.constant([discount] * self.batch_size),
            observation=tf.constant([[1.]] * self.batch_size))


class PolicyUnittestEnv(UnittestEnv):
    """
    The agent receives reward 1 if its action matches the observation.
    """

    def _gen_time_step(self, s, action):
        step_type = StepType.MID
        discount = 1.0

        if s == 0:
            step_type = StepType.FIRST
        elif s == self._episode_length - 1:
            step_type = StepType.LAST
            discount = 0.0

        if s == 0:
            reward = tf.constant([0.] * self.batch_size)
        else:
            prev_observation = self._current_time_step().observation
            reward = tf.cast(
                tf.equal(action, tf.cast(prev_observation, tf.int64)),
                tf.float32)
            reward = tf.reshape(reward, shape=(self.batch_size, ))

        observation = tf.constant(
            np.random.randint(2, size=(self.batch_size, 1)), dtype=tf.float32)

        return TimeStep(
            step_type=tf.constant([step_type] * self.batch_size),
            reward=reward,
            discount=tf.constant([discount] * self.batch_size),
            observation=observation)


class RNNPolicyUnittestEnv(UnittestEnv):
    """
    The agent receives reward 1 after initial `gap` steps if its
    actions action match the observation given at the first step.
    """

    def __init__(self, batch_size, episode_length, gap):
        self._gap = gap
        super(RNNPolicyUnittestEnv, self).__init__(batch_size, episode_length)

    def _gen_time_step(self, s, action):
        step_type = StepType.MID
        discount = 1.0

        if s == 0:
            self._observation0 = tf.constant(
                2 * np.random.randint(2, size=(self.batch_size, 1)) - 1,
                dtype=tf.float32)
            step_type = StepType.FIRST
        elif s == self._episode_length - 1:
            step_type = StepType.LAST
            discount = 0.0

        if s <= self._gap:
            reward = tf.constant([0.] * self.batch_size)
        else:
            reward = tf.cast(
                tf.equal(action * 2 - 1, tf.cast(self._observation0,
                                                 tf.int64)), tf.float32)
            reward = tf.reshape(reward, shape=(self.batch_size, ))

        if s == 0:
            observation = self._observation0
        else:
            observation = tf.zeros((self.batch_size, 1))

        return TimeStep(
            step_type=tf.constant([step_type] * self.batch_size),
            reward=reward,
            discount=tf.constant([discount] * self.batch_size),
            observation=observation)
