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
"""Environments for unittest."""

from abc import abstractmethod
from enum import Enum
import numpy as np
import torch

import alf
from alf.data_structures import StepType, TimeStep
from alf.tensor_specs import BoundedTensorSpec, TensorSpec

from .alf_environment import AlfEnvironment

ActionType = Enum('ActionType', ('Discrete', 'Continuous'))


class UnittestEnv(AlfEnvironment):
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
                 action_type=ActionType.Discrete,
                 nested_observation=False,
                 reward_dim=1):
        """Initializes the environment.

        Args:
            batch_size (int): The batch size expected for the actions and
                observations.
            episode_length (int): length of each episode
            action_type (nest): ActionType
            nested_observation (bool): whether observation is a tensor
        """
        self._steps = 0
        self._episode_length = episode_length
        super(UnittestEnv, self).__init__()
        self._action_type = action_type

        def _create_action_spec(act_type):
            if act_type == ActionType.Discrete:
                return BoundedTensorSpec(
                    shape=(), dtype=torch.int64, minimum=0, maximum=1)
            else:
                return BoundedTensorSpec(
                    shape=(1, ), dtype=torch.float32, minimum=[0], maximum=[1])

        self._action_spec = alf.nest.map_structure(_create_action_spec,
                                                   action_type)

        self._nested_observation = nested_observation
        observation_spec = TensorSpec(shape=(obs_dim, ), dtype=torch.float32)
        if nested_observation:
            self._observation_spec = (observation_spec, observation_spec)
        else:
            self._observation_spec = observation_spec
        self._batch_size = batch_size
        self._reward_dim = reward_dim
        if reward_dim == 1:
            self._reward_spec = TensorSpec(())
        else:
            self._reward_spec = TensorSpec((reward_dim, ))

        self.reset()

    @property
    def is_tensor_based(self):
        return True

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

    def reward_spec(self):
        return self._reward_spec

    def env_info_spec(self):
        return {}

    def _reset(self):
        self._steps = 0
        time_step = self._gen_time_step(0, None)
        self._current_time_step = time_step._replace(
            prev_action=alf.nest.map_structure(
                lambda spec: spec.zeros([self.batch_size]), self._action_spec),
            env_id=torch.arange(self.batch_size, dtype=torch.int32))
        return self._current_time_step

    def _step(self, action):
        self._steps += 1
        time_step = self._gen_time_step(self._steps % self._episode_length,
                                        action)
        self._current_time_step = time_step._replace(
            prev_action=action,
            env_id=torch.arange(self.batch_size, dtype=torch.int32))
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
            step_type=torch.full([self.batch_size],
                                 step_type,
                                 dtype=torch.int32),
            reward=torch.ones(self.batch_size),
            discount=torch.full([
                self.batch_size,
            ], discount),
            observation=torch.ones(self.batch_size))


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
            reward = torch.zeros(self.batch_size)
        else:
            prev_observation = self._current_time_step.observation
            if self._nested_observation:
                prev_observation = prev_observation[0]
            reward = 1.0 - torch.abs(prev_observation -
                                     action.reshape(prev_observation.shape))
            reward = reward.reshape(self.batch_size)

        if self._reward_dim != 1:
            reward = reward.unsqueeze(-1).expand((-1, self._reward_dim))

        observation = torch.randint(
            0, 2, size=(self.batch_size, 1), dtype=torch.float32)
        if self._nested_observation:
            observation = (observation, torch.randn_like(observation))

        return TimeStep(
            step_type=torch.full([self.batch_size],
                                 step_type,
                                 dtype=torch.int32),
            reward=reward,
            discount=torch.full([self.batch_size], discount),
            observation=observation)


class MixedPolicyUnittestEnv(UnittestEnv):
    """Environment for testing a mixed policy.

    Given the agent's `(discrete, continuous)` action pair ``(a_d, a_c)``, if
    ``'a_d == (a_c > 0.5)``, the agent receives a reward of 1; otherwise it
    receives 0.
    """

    def __init__(self, batch_size, episode_length, obs_dim=1):
        """Initializes the environment.

        Args:
            batch_size (int): The batch size expected for the actions and
                observations.
            episode_length (int): length of each episode
        """
        super().__init__(
            batch_size=batch_size,
            episode_length=episode_length,
            obs_dim=obs_dim,
            action_type=[ActionType.Discrete, ActionType.Continuous])

    def _gen_time_step(self, s, action):
        step_type = StepType.MID
        discount = 1.0
        reward = torch.zeros(self.batch_size)

        if s == 0:
            step_type = StepType.FIRST
        elif s == self._episode_length - 1:
            step_type = StepType.LAST
            discount = 0.0

        if s > 0:
            reward = (action[0] == (action[1].squeeze(-1) > 0.5).to(
                torch.int64)).to(torch.float32)

        observation = self._observation_spec.randn(
            outer_dims=(self.batch_size, ))

        return TimeStep(
            step_type=torch.full([self.batch_size],
                                 step_type,
                                 dtype=torch.int32),
            reward=reward,
            discount=torch.full([self.batch_size], discount),
            observation=observation)


class RNNPolicyUnittestEnv(UnittestEnv):
    """Environment for testing RNN policy.

    The agent receives reward 1 after initial `gap` steps if its
    actions action match the observation given at the first step.
    """

    def __init__(self,
                 batch_size,
                 episode_length,
                 gap=3,
                 action_type=ActionType.Discrete,
                 obs_dim=1):
        self._gap = gap
        self._obs_dim = obs_dim
        super(RNNPolicyUnittestEnv, self).__init__(
            batch_size,
            episode_length,
            action_type=action_type,
            obs_dim=obs_dim)

    def _gen_time_step(self, s, action):
        step_type = StepType.MID
        discount = 1.0
        obs_dim = self._obs_dim

        if s == 0:
            self._observation0 = 2. * torch.randint(
                0, 2, size=(self.batch_size, 1)) - 1.
            if obs_dim > 1:
                self._observation0 = torch.cat([
                    self._observation0,
                    torch.ones(self.batch_size, obs_dim - 1)
                ],
                                               dim=-1)
            step_type = StepType.FIRST
        elif s == self._episode_length - 1:
            step_type = StepType.LAST
            discount = 0.0

        if s <= self._gap:
            reward = torch.zeros(self.batch_size)
        else:
            obs0 = self._observation0[:, 0].reshape(self.batch_size, 1)
            reward = 1.0 - 0.5 * torch.abs(2 * action.reshape(obs0.shape) - 1 -
                                           obs0)
            reward = reward.reshape(self.batch_size)

        if s == 0:
            observation = self._observation0
        else:
            observation = torch.zeros(self.batch_size, obs_dim)

        return TimeStep(
            step_type=torch.full([self.batch_size],
                                 step_type,
                                 dtype=torch.int32),
            reward=reward,
            discount=torch.full([self.batch_size], discount),
            observation=observation)
