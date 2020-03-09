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
"""Wrapper providing a TorchEnvironment adapter for GYM environments.
    
Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/suite_gym.py
"""

import collections
import gym
import gym.spaces
import numpy as np
import torch

import alf.data_structures as ds
from alf.environments import torch_environment
import alf.nest as nest
from alf.tensor_specs import TensorSpec, BoundedTensorSpec


def tensor_spec_from_gym_space(space, simplify_box_bounds=True):
    """
    Mostly adapted from `spec_from_gym_space` in
    `tf_agents.environments.gym_wrapper`. Instead of using a `dtype_map`
    as default data types, it always uses dtypes of gym spaces since gym is now
    updated to support this.
    """

    # We try to simplify redundant arrays to make logging and debugging less
    # verbose and easier to read since the printed spec bounds may be large.
    def try_simplify_array_to_value(np_array):
        """If given numpy array has all the same values, returns that value."""
        first_value = np_array.item(0)
        if np.all(np_array == first_value):
            return np.array(first_value, dtype=np_array.dtype)
        else:
            return np_array

    def torch_dtype(data):
        return getattr(torch, space.dtype.name)

    if isinstance(space, gym.spaces.Discrete):
        # Discrete spaces span the set {0, 1, ... , n-1} while Bounded Array specs
        # are inclusive on their bounds.
        maximum = space.n - 1
        return BoundedTensorSpec(
            shape=(), dtype=torch_dtype(space), minimum=0, maximum=maximum)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        maximum = try_simplify_array_to_value(
            np.asarray(space.nvec - 1, dtype=space.dtype))
        return BoundedTensorSpec(
            shape=space.shape,
            dtype=torch_dtype(space),
            minimum=0,
            maximum=maximum)
    elif isinstance(space, gym.spaces.MultiBinary):
        shape = (space.n, )
        return BoundedTensorSpec(
            shape=shape, dtype=torch_dtype(space), minimum=0, maximum=1)
    elif isinstance(space, gym.spaces.Box):
        minimum = np.asarray(space.low, dtype=space.dtype)
        maximum = np.asarray(space.high, dtype=space.dtype)
        if simplify_box_bounds:
            minimum = try_simplify_array_to_value(minimum)
            maximum = try_simplify_array_to_value(maximum)
        return BoundedTensorSpec(
            shape=space.shape,
            dtype=torch_dtype(space),
            minimum=minimum,
            maximum=maximum)
    elif isinstance(space, gym.spaces.Tuple):
        return tuple([tensor_spec_from_gym_space(s) for s in space.spaces])
    elif isinstance(space, gym.spaces.Dict):
        return collections.OrderedDict([(key, tensor_spec_from_gym_space(s))
                                        for key, s in space.spaces.items()])
    else:
        raise ValueError(
            'The gym space {} is currently not supported.'.format(space))


class TorchGymWrapper(torch_environment.TorchEnvironment):
    """Base wrapper implementing TorchEnvironmentBaseWrapper interface for Gym envs.

    Action and observation specs are automatically generated from the action and
    observation spaces. See base class for TorchEnvironment details.
    """

    def __init__(self,
                 gym_env,
                 env_id=None,
                 discount=1.0,
                 auto_reset=True,
                 simplify_box_bounds=True):
        """Create a TorchEnvironment.

        Args:
            gym_env (gym.Env): An instance of OpenAI gym environment.
            env_id (int): (optional) ID of the environment.
            discount (float): Discount to use for the environment.
            auto_reset (bool): whether or not to reset the environment when done.
            simplify_box_bounds (bool): whether or not to simplify redundant
                arrays to values for spec bounds.

        """
        super(TorchGymWrapper, self).__init__()

        self._gym_env = gym_env
        self._discount = discount
        if env_id is None:
            self._env_id = torch.as_tensor(0, dtype=torch.int32)
        else:
            self._env_id = torch.as_tensor(env_id, dtype=torch.int32)
        self._action_is_discrete = isinstance(self._gym_env.action_space,
                                              gym.spaces.Discrete)
        # TODO: Add test for auto_reset param.
        self._auto_reset = auto_reset
        self._observation_spec = tensor_spec_from_gym_space(
            self._gym_env.observation_space, simplify_box_bounds)
        self._action_spec = tensor_spec_from_gym_space(
            self._gym_env.action_space, simplify_box_bounds)
        self._flat_obs_spec = nest.flatten(self._observation_spec)
        self._time_step_spec = ds.time_step_spec(self._observation_spec,
                                                 self._action_spec)
        self._info = None
        self._done = True

    @property
    def gym(self):
        """Return the gym environment. """
        return self._gym_env

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self._gym_env, name)

    def get_info(self):
        """Returns the gym environment info returned on the last step."""
        return self._info

    def _convert_action(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().data.cpu().numpy()
        return action

    def _reset(self):
        # TODO: Upcoming update on gym adds **kwargs on reset. Update this to
        # support that.
        observation = self._gym_env.reset()
        self._info = None
        self._done = False

        observation = self._to_tensor_observation(observation)
        return ds.restart(
            observation=observation,
            action_spec=self._action_spec,
            env_id=self._env_id)

    @property
    def done(self):
        return self._done

    def _step(self, action):
        # Automatically reset the environments on step if they need to be reset.
        if self._auto_reset and self._done:
            return self.reset()

        py_action = self._convert_action(action)
        # TODO(oars): Figure out how tuple or dict actions will be generated by the
        # agents and if we can pass them through directly to gym.

        observation, reward, self._done, self._info = self._gym_env.step(
            py_action)
        observation = self._to_tensor_observation(observation)

        if self._done:
            return ds.termination(observation, action, reward, self._env_id)
        else:
            return ds.transition(observation, action, reward, self._discount,
                                 self._env_id)

    def _to_tensor_observation(self, observation):
        """Make sure observation from env is converted to (nested) torch tensor.

        Args:
            observation (nested arrays or tensors): observations from env.

        Returns:
            A (nested) tensors of observation
        """
        flat_obs = nest.flatten(observation)
        tensor_observations = [
            torch.as_tensor(obs, dtype=spec.dtype)
            for spec, obs in zip(self._flat_obs_spec, flat_obs)
        ]
        return nest.pack_sequence_as(self._observation_spec,
                                     tensor_observations)

    def time_step_spec(self):
        return self._time_step_spec

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def close(self):
        return self._gym_env.close()

    def seed(self, seed):
        return self._gym_env.seed(seed)

    def render(self, mode='rgb_array'):
        return self._gym_env.render(mode)
