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
"""Pytorch environment implementation that generates random observations.

Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/random_py_environment.py
"""
import numpy as np
import torch

import alf.data_structures as ds
from alf.environments import torch_environment
from alf.nest import nest
import alf.tensor_specs as ts


class RandomTorchEnvironment(torch_environment.TorchEnvironment):
    """Randomly generates observations following the given observation_spec.

    If an action_spec is provided it validates that the actions used to step the
    environment fall within the defined spec.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 env_id=None,
                 episode_end_probability=0.1,
                 discount=1.0,
                 reward_fn=None,
                 batch_size=None,
                 seed=42,
                 render_size=(2, 2, 3),
                 min_duration=0,
                 max_duration=None):
        """Initializes the environment.

        Args:
            observation_spec (nested TensorSpec): tensor spec for observations 
            action_spec (nested TensorSpec): tensor spec for actions.
            env_id (int): (optional) ID of the environment.
            episode_end_probability (float): Probability an episode will end when the
                environment is stepped.
            discount (float): Discount to set in time_steps.
            reward_fn (Callable): Callable that takes in step_type, action, an observation(s),
                and returns a tensor of rewards.
            batch_size (int): (Optional) Number of observations generated per call.
                If this value is not `None`, then all actions are expected to
                have an additional major axis of size `batch_size`, and all outputs
                will have an additional major axis of size `batch_size`.
            seed (int): Seed to use for rng used in observation generation.
            render_size (tuple of ints): Size of the random render image to return when calling 
                render.
            min_duration (int): Number of steps at the beginning of the
                episode during which the episode can not terminate.
            max_duration (int): Optional number of steps after which the episode
                terminates regarless of the termination probability.

        Raises:
            ValueError: If batch_size argument is not None and does not match the
            shapes of discount or reward.
        """
        self._batch_size = batch_size
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._time_step_spec = ds.time_step_spec(self._observation_spec,
                                                 action_spec)
        self._episode_end_probability = episode_end_probability
        discount = np.asarray(discount, dtype=np.float32)
        if env_id is None:
            self._env_id = torch.as_tensor(0, dtype=torch.int32)
        else:
            self._env_id = torch.as_tensor(env_id, dtype=torch.int32)

        if self._batch_size:
            if not discount.shape:
                discount = np.tile(discount, self._batch_size)
            if self._batch_size != len(discount):
                raise ValueError(
                    'Size of discounts must equal the batch size.')
        self._discount = discount

        if reward_fn is None:
            # Return a reward whose size matches the batch size
            if self._batch_size is None:
                self._reward_fn = lambda *_: torch.tensor(
                    0.0, dtype=torch.float32)
            else:
                self._reward_fn = (lambda *_: torch.zeros(
                    self._batch_size, dtype=torch.float32))
        else:
            self._reward_fn = reward_fn

        self._done = True
        self._num_steps = 0
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._rng = np.random.RandomState(seed)
        self._render_size = render_size
        super(RandomTorchEnvironment, self).__init__()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batched(self):
        return False if self._batch_size is None else True

    def _get_observation(self):
        batch_size = (self._batch_size, ) if self._batch_size else ()
        return nest.map_structure(
            lambda spec: self._sample_spec(spec, batch_size),
            self._observation_spec)

    def _reset(self):
        self._done = False
        batched = self._batch_size is not None
        return ds.restart(self._get_observation(), self._action_spec,
                          self._env_id, batched)

    def _sample_spec(self, spec, outer_dims):
        """Sample the given TensorSpec."""
        shape = spec.shape
        if not isinstance(spec, ts.BoundedTensorSpec):
            spec = ts.BoundedTensorSpec(shape, spec.dtype)
        return spec.sample(outer_dims=outer_dims)

    def _check_reward_shape(self, reward):
        expected_shape = () if self._batch_size is None else (
            self._batch_size, )
        if reward.shape != expected_shape:
            raise ValueError(
                '%r != %r. Size of reward must equal the batch size.' %
                (np.asarray(reward.cpu()).shape, self._batch_size))

    def _step(self, action):
        if self._done:
            return self.reset()

        if self._action_spec:
            nest.assert_same_structure(self._action_spec, action)

        self._num_steps += 1

        observation = self._get_observation()
        if self._num_steps < self._min_duration:
            self._done = False
        elif self._max_duration and self._num_steps >= self._max_duration:
            self._done = True
        else:
            self._done = self._rng.uniform() < self._episode_end_probability

        if self._batch_size:
            action = nest.map_structure(
                lambda t: torch.cat([t.unsqueeze(0)] * self._batch_size),
                action)

        if self._done:
            reward = self._reward_fn(ds.StepType.LAST, action, observation)
            self._check_reward_shape(reward)
            time_step = ds.termination(observation, action, reward,
                                       self._env_id)
            self._num_steps = 0
        else:
            reward = self._reward_fn(ds.StepType.MID, action, observation)
            self._check_reward_shape(reward)
            time_step = ds.transition(observation, action, reward,
                                      self._discount, self._env_id)

        return time_step

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise ValueError(
                "Only rendering mode supported is 'rgb_array', got {} instead."
                .format(mode))

        return self._rng.randint(
            0, 256, size=self._render_size, dtype=np.uint8)

    def seed(self, seed):
        self._rng.seed(seed)
