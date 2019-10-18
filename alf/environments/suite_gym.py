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

import collections
import gym
import gym.spaces
import numpy as np
from tf_agents import specs
from tf_agents.environments import gym_wrapper


def _spec_from_gym_space(space, dtype_map, simplify_box_bounds=True):
    """
    Mostly the same with `_spec_from_gym_space` in
    `tf_agents.environments.gym_wrapper`. This function no longer use `dtype_map`
    to set data types; instead it always uses dtypes of gym spaces as gym is now
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

    if isinstance(space, gym.spaces.Discrete):
        # Discrete spaces span the set {0, 1, ... , n-1} while Bounded Array specs
        # are inclusive on their bounds.
        maximum = space.n - 1
        return specs.BoundedArraySpec(
            shape=(), dtype=space.dtype, minimum=0, maximum=maximum)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        maximum = try_simplify_array_to_value(
            np.asarray(space.nvec - 1, dtype=space.dtype))
        return specs.BoundedArraySpec(
            shape=space.shape, dtype=space.dtype, minimum=0, maximum=maximum)
    elif isinstance(space, gym.spaces.MultiBinary):
        shape = (space.n, )
        return specs.BoundedArraySpec(
            shape=shape, dtype=space.dtype, minimum=0, maximum=1)
    elif isinstance(space, gym.spaces.Box):
        dtype = space.dtype
        minimum = np.asarray(space.low, dtype=dtype)
        maximum = np.asarray(space.high, dtype=dtype)
        if simplify_box_bounds:
            minimum = try_simplify_array_to_value(minimum)
            maximum = try_simplify_array_to_value(maximum)
        return specs.BoundedArraySpec(
            shape=space.shape, dtype=dtype, minimum=minimum, maximum=maximum)
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(
            [_spec_from_gym_space(s, dtype_map) for s in space.spaces])
    elif isinstance(space, gym.spaces.Dict):
        return collections.OrderedDict([(key, _spec_from_gym_space(
            s, dtype_map)) for key, s in space.spaces.items()])
    else:
        raise ValueError(
            'The gym space {} is currently not supported.'.format(space))


gym_wrapper._spec_from_gym_space = _spec_from_gym_space

from tf_agents.environments import suite_gym

# forward tf_agents suite_gym's functions
load = suite_gym.load
wrap_env = suite_gym.wrap_env
