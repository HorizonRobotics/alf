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
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.tensor_spec import TensorSpec, BoundedTensorSpec
from tf_agents.trajectories.time_step import TimeStep, StepType

from enum import Enum

ActionType = Enum('ActionType', ('Discrete', 'Continuous'))


class UnittestEnv(PyEnvironment):
    """Abstract base for unittest environment.

    Every episode ends in `episode_length` steps (including LAST step).
    The observation is one dimensional.
    The action is binary {0, 1} when action_type is ActionType.Discrete
        and a float value in range (0.0, 1.0) when action_type is ActionType.Continuous
    """

    def __init__(self,
                 batch_size,
                 episode_length,
                 obs_dim=1,
                 action_type=ActionType.Discrete):
        """Initializes the environment.

        Args:
            batch_size (int): The batch size expected for the actions and
                observations.
            episode_length (int): length of each episode
            action_type: ActionType
        """
        self._steps = 0
        self._episode_length = episode_length
        super(UnittestEnv, self).__init__()
        self._action_type = action_type
        if action_type == ActionType.Discrete:
            self._action_spec = BoundedTensorSpec(
                shape=(1, ), dtype=tf.int64, minimum=0, maximum=1)
        else:
            self._action_spec = BoundedTensorSpec(
                shape=(1, ), dtype=tf.float32, minimum=[0], maximum=[1])

        self._observation_spec = TensorSpec(
            shape=(obs_dim, ), dtype=tf.float32)
        self._batch_size = batch_size
        self.reset()

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._batch_size

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._steps = 0
        self._current_time_step = self._gen_time_step(0, None)
        return self._current_time_step

    def _step(self, action):
        self._steps += 1
        self._current_time_step = self._gen_time_step(
            self._steps % self._episode_length, action)
        return self._current_time_step

    @abstractmethod
    def _gen_time_step(self, s, action):
        """Generate time step.

        Args:
            s (int): step count in current episode. It ranges from 0 to
                `episode_length` - 1.
            action: action from agent.

        Returns:
            time_step (TimeStep)
        """
        pass


class ValueUnittestEnv(UnittestEnv):
    """Environment for testing value estimation.

    Every episode ends in `episode_length` steps. It always give reward
    1 at each step.
    """

    def _gen_time_step(self, s, action):
        """Return the current `TimeStep`."""
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
    """Environment for testing policy.

    The agent receives 1-diff(action, observation) as reward
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
            prev_observation = self._current_time_step.observation
            reward = 1.0 - tf.abs(prev_observation -
                                  tf.cast(action, tf.float32))
            reward = tf.reshape(reward, shape=(self.batch_size, ))

        observation = tf.constant(
            np.random.randint(2, size=(self.batch_size, 1)), dtype=tf.float32)

        return TimeStep(
            step_type=tf.constant([step_type] * self.batch_size),
            reward=reward,
            discount=tf.constant([discount] * self.batch_size),
            observation=observation)


class RNNPolicyUnittestEnv(UnittestEnv):
    """Environment for testing RNN policy.

    The agent receives reward 1 after initial `gap` steps if its
    actions action match the observation given at the first step.
    """

    def __init__(self, batch_size, episode_length, gap, obs_dim=1):
        self._gap = gap
        self._obs_dim = obs_dim
        super(RNNPolicyUnittestEnv, self).__init__(
            batch_size, episode_length, obs_dim=obs_dim)

    def _gen_time_step(self, s, action):
        step_type = StepType.MID
        discount = 1.0
        obs_dim = self._obs_dim

        if s == 0:
            self._observation0 = tf.constant(
                2 * np.random.randint(2, size=(self.batch_size, 1)) - 1,
                dtype=tf.float32)
            if obs_dim > 1:
                self._observation0 = tf.concat([
                    self._observation0,
                    tf.ones((self.batch_size, obs_dim - 1))
                ],
                                               axis=-1)
            step_type = StepType.FIRST
        elif s == self._episode_length - 1:
            step_type = StepType.LAST
            discount = 0.0

        if s <= self._gap:
            reward = tf.constant([0.] * self.batch_size)
        else:
            obs0 = tf.reshape(
                tf.cast(self._observation0[:, 0], tf.int64),
                shape=(self.batch_size, 1))
            reward = tf.cast(tf.equal(action * 2 - 1, obs0), tf.float32)
            reward = tf.reshape(reward, shape=(self.batch_size, ))

        if s == 0:
            observation = self._observation0
        else:
            observation = tf.zeros((self.batch_size, obs_dim))

        return TimeStep(
            step_type=tf.constant([step_type] * self.batch_size),
            reward=reward,
            discount=tf.constant([discount] * self.batch_size),
            observation=observation)
