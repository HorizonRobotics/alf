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
"""Wrapper providing an AlfEnvironment adapter for GYM environments.

Adapted from TF-Agents Environment API as seen in:
    https://github.com/tensorflow/agents/blob/master/tf_agents/environments/suite_gym.py
"""

import collections
import gym
import gym.spaces
import numbers
import numpy as np

import alf.data_structures as ds
from alf.environments.alf_environment import AlfEnvironment
import alf.nest as nest
from alf.tensor_specs import TensorSpec, BoundedTensorSpec, torch_dtype_to_str


def tensor_spec_from_gym_space(space,
                               simplify_box_bounds=True,
                               float_dtype=np.float32):
    """
    Construct tensor spec from gym space.

    Args:
        space (gym.Space): An instance of OpenAI gym Space.
        simplify_box_bounds (bool): if True, will try to simplify redundant
            arrays to make logging and debugging less verbose when printed out.
        float_dtype (np.float32 | np.float64 | None): the dtype to be used for
            the floating numbers. If None, it will use dtypes of gym spaces.
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

    if isinstance(space, gym.spaces.Discrete):
        # Discrete spaces span the set {0, 1, ... , n-1} while Bounded Array specs
        # are inclusive on their bounds.
        maximum = space.n - 1
        return BoundedTensorSpec(
            shape=(), dtype=space.dtype.name, minimum=0, maximum=maximum)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        maximum = try_simplify_array_to_value(
            np.asarray(space.nvec - 1, dtype=space.dtype))
        return BoundedTensorSpec(
            shape=space.shape,
            dtype=space.dtype.name,
            minimum=0,
            maximum=maximum)
    elif isinstance(space, gym.spaces.MultiBinary):
        shape = (space.n, )
        return BoundedTensorSpec(
            shape=shape, dtype=space.dtype.name, minimum=0, maximum=1)
    elif isinstance(space, gym.spaces.Box):

        if float_dtype is not None and "float" in space.dtype.name:
            dtype = np.dtype(float_dtype)
        else:
            dtype = space.dtype

        minimum = np.asarray(space.low, dtype=dtype)
        maximum = np.asarray(space.high, dtype=dtype)
        if simplify_box_bounds:
            minimum = try_simplify_array_to_value(minimum)
            maximum = try_simplify_array_to_value(maximum)
        return BoundedTensorSpec(
            shape=space.shape,
            dtype=dtype.name,
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


def _as_array(nested):
    """Convert scalars in ``nested`` to np.ndarray."""

    def __as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x

    return nest.map_structure(__as_array, nested)


class AlfGymWrapper(AlfEnvironment):
    """Base wrapper implementing AlfEnvironmentBaseWrapper interface for Gym envs.

    Action and observation specs are automatically generated from the action and
    observation spaces. See base class for ``AlfEnvironment`` details.
    """

    def __init__(self,
                 gym_env,
                 env_id=None,
                 discount=1.0,
                 auto_reset=True,
                 simplify_box_bounds=True):
        """

        Args:
            gym_env (gym.Env): An instance of OpenAI gym environment.
            env_id (int): (optional) ID of the environment.
            discount (float): Discount to use for the environment.
            auto_reset (bool): whether or not to reset the environment when done.
            simplify_box_bounds (bool): whether or not to simplify redundant
                arrays to values for spec bounds.

        """
        super(AlfGymWrapper, self).__init__()

        self._gym_env = gym_env
        self._discount = discount
        if env_id is None:
            env_id = 0
        self._env_id = np.int32(env_id)
        self._action_is_discrete = isinstance(self._gym_env.action_space,
                                              gym.spaces.Discrete)
        # TODO: Add test for auto_reset param.
        self._auto_reset = auto_reset
        self._observation_spec = tensor_spec_from_gym_space(
            self._gym_env.observation_space, simplify_box_bounds)
        self._action_spec = tensor_spec_from_gym_space(
            self._gym_env.action_space, simplify_box_bounds)
        if hasattr(self._gym_env, "reward_space"):
            self._reward_spec = tensor_spec_from_gym_space(
                self._gym_env.reward_space, simplify_box_bounds)
        else:
            self._reward_spec = TensorSpec(())
        self._time_step_spec = ds.time_step_spec(
            self._observation_spec, self._action_spec, self._reward_spec)
        self._info = None
        self._done = True
        self._zero_info = self._obtain_zero_info()

        self._env_info_spec = nest.map_structure(TensorSpec.from_array,
                                                 self._zero_info)

    @property
    def gym(self):
        """Return the gym environment. """
        return self._gym_env

    @property
    def is_tensor_based(self):
        return False

    def _obtain_zero_info(self):
        """Get an env info of zeros only once when the env is created.
        This info will be filled in each ``FIRST`` time step as a placeholder.
        """
        self._gym_env.reset()
        action = nest.map_structure(lambda spec: spec.numpy_zeros(),
                                    self._action_spec)
        _, _, _, info = self._gym_env.step(action)
        self._gym_env.reset()
        info = _as_array(info)
        return nest.map_structure(lambda a: np.zeros_like(a), info)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self._gym_env, name)

    def get_info(self):
        """Returns the gym environment info returned on the last step."""
        return self._info

    def _reset(self):
        # TODO: Upcoming update on gym adds **kwargs on reset. Update this to
        # support that.
        observation = self._gym_env.reset()
        self._info = None
        self._done = False

        observation = self._to_spec_dtype_observation(observation)
        return ds.restart(
            observation=observation,
            action_spec=self._action_spec,
            reward_spec=self._reward_spec,
            env_id=self._env_id,
            env_info=self._zero_info)

    @property
    def done(self):
        return self._done

    def _step(self, action):
        # Automatically reset the environments on step if they need to be reset.
        if self._auto_reset and self._done:
            return self.reset()

        observation, reward, self._done, self._info = self._gym_env.step(
            action)
        # NOTE: In recent version of gym, the environment info may have
        # "TimeLimit.truncated" to indicate that the env has run beyond the time
        # limit. If so, it will removed to avoid having conflict with our env
        # info spec.
        self._info.pop("TimeLimit.truncated", None)
        observation = self._to_spec_dtype_observation(observation)
        self._info = _as_array(self._info)

        if self._done:
            return ds.termination(
                observation,
                action,
                reward,
                self._reward_spec,
                self._env_id,
                env_info=self._info)
        else:
            return ds.transition(
                observation,
                action,
                reward,
                self._reward_spec,
                self._discount,
                self._env_id,
                env_info=self._info)

    def _to_spec_dtype_observation(self, observation):
        """Make sure observation from env is converted to the correct dtype.

        Args:
            observation (nested arrays or tensors): observations from env.

        Returns:
            A (nested) arrays of observation
        """

        def _as_spec_dtype(arr, spec):
            dtype = torch_dtype_to_str(spec.dtype)
            if str(arr.dtype) == dtype:
                return arr
            else:
                return arr.astype(dtype)

        return nest.map_structure(_as_spec_dtype, observation,
                                  self._observation_spec)

    def env_info_spec(self):
        return self._env_info_spec

    def time_step_spec(self):
        return self._time_step_spec

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def close(self):
        return self._gym_env.close()

    def seed(self, seed):
        return self._gym_env.seed(seed)

    def render(self, mode='rgb_array'):
        return self._gym_env.render(mode)
