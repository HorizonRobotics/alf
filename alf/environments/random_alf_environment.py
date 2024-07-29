# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""An environment that generates random observations.

Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/random_py_environment.py
"""
from absl import logging
import numpy as np
import torch

import alf.data_structures as ds
from alf.environments import alf_environment
from alf.nest import nest
import alf.tensor_specs as ts


class RandomAlfEnvironment(alf_environment.AlfEnvironment):
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
                 max_duration=None,
                 use_tensor_time_step=False):
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
                terminates regardless of the termination probability.
            use_tensor_time_step (bool): convert all quantities in time_step
                to torch.tensor if True. Otherwise use numpy data types.

        Raises:
            ValueError: If batch_size argument is not None and does not match the
            shapes of discount or reward.
        """
        self._batch_size = batch_size
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._time_step_spec = ds.time_step_spec(
            self._observation_spec, action_spec, ts.TensorSpec(()))
        self._episode_end_probability = episode_end_probability
        discount = np.asarray(discount, dtype=np.float32)
        if env_id is None:
            self._env_id = np.int32(0)
        else:
            self._env_id = np.int32(env_id)

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
                self._reward_fn = lambda *_: np.float32(0)
            else:
                self._reward_fn = (
                    lambda *_: np.zeros(self._batch_size, dtype=np.float32))
        else:
            self._reward_fn = reward_fn

        self._done = True
        self._num_steps = 0
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._rng = np.random.RandomState(seed)
        self._render_size = render_size
        self._use_tensor_time_step = use_tensor_time_step

        super(RandomAlfEnvironment, self).__init__()

    def env_info_spec(self):
        return {}

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    @property
    def is_tensor_based(self):
        return self._use_tensor_time_step

    @property
    def batch_size(self):
        return self._batch_size if self.batched else 1

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
        time_step = ds.restart(
            self._get_observation(),
            self._action_spec,
            env_id=self._env_id,
            batched=batched)
        if self._use_tensor_time_step:
            time_step = nest.map_structure(torch.as_tensor, time_step)
        return time_step

    def _sample_spec(self, spec, outer_dims):
        """Sample the given TensorSpec."""
        shape = spec.shape
        if not isinstance(spec, ts.BoundedTensorSpec):
            spec = ts.BoundedTensorSpec(shape, spec.dtype)
        return spec.numpy_sample(outer_dims=outer_dims, rng=self._rng)

    def _check_reward_shape(self, reward):
        expected_shape = () if self._batch_size is None else (
            self._batch_size, )
        if reward.shape != expected_shape:
            raise ValueError(
                '%r != %r. Size of reward must equal the batch size.' %
                (np.asarray(reward).shape, self._batch_size))

    def _step(self, action):
        if self._done:
            time_step = self.reset()
            if self._use_tensor_time_step:
                time_step = nest.map_structure(torch.as_tensor, time_step)
            return time_step

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
                lambda t: np.concatenate([np.expand_dims(t, 0)] * self.
                                         _batch_size), action)

        if self._done:
            reward = self._reward_fn(ds.StepType.LAST, action, observation)
            self._check_reward_shape(reward)
            time_step = ds.termination(
                observation, action, reward, env_id=self._env_id)
            self._num_steps = 0
        else:
            reward = self._reward_fn(ds.StepType.MID, action, observation)
            self._check_reward_shape(reward)
            time_step = ds.transition(
                observation,
                action,
                reward,
                discount=self._discount,
                env_id=self._env_id)

        if self._use_tensor_time_step:
            time_step = nest.map_structure(torch.as_tensor, time_step)

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
