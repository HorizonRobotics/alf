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
"""Suite for loading highway environments.
    Installation:
    pip install git+https://github.com/eleurent/highway-env
"""

import collections
import gym
import gym.spaces
import numpy as np

import alf
from alf.environments import suite_gym, alf_wrappers, gym_wrappers, process_environment

try:
    import highway_env
except ImportError:
    highway_env = None


def is_available():
    return highway_env is not None


class FlattenObservation(gym_wrappers.BaseObservationWrapper):
    """Flatten the 2D observations into a 1D vector
    """

    def transform_space(self, observation_space):
        return gym.spaces.Box(
            low=-observation_space.low.ravel(),
            high=observation_space.high.ravel())

    def transform_observation(self, observation):
        return observation.ravel()


class RemoveActionEnvInfo(gym.Wrapper):
    """Remove action from EnvInfo if exist
    """

    def step(self, action):
        obs, reward, done, env_info = self.env.step(action)
        env_info.pop('action', None)
        return obs, reward, done, env_info


class ActionScalarization(gym.Wrapper):
    """Convert action to scalar if the current action space is MetaDiscreteAction
        and type of the input action is ``np.ndarray``
    """

    def __init__(self, env):
        super().__init__(env)
        self._is_discrete = isinstance(self.action_space, gym.spaces.Discrete)

    def step(self, action):
        if self._is_discrete and isinstance(action, np.ndarray):
            action = action.item()
        return self.env.step(action)


@alf.configurable
def load(environment_name,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         env_config=None):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a TimeLimit wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name (str): Name for the environment to load.
        env_id (int): (optional) ID of the environment.
        discount (float): Discount to use for the environment.
        max_episode_steps (int): If None or 0 the ``max_episode_steps`` will be
            set to the default step limit defined in the environment. Otherwise
            ``max_episode_steps`` will be set to the smaller value of the two.
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers
            classes to use directly on the gym environment.
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.
        env_config (dict|None): a dictionary for configuring some aspects of the
            environment. If is None, the default configuration will be used.
            Please refer to the ``default_env_config`` below for
            an example config and the doc for more details:
            https://highway-env.readthedocs.io/en/latest/user_guide.html

    Returns:
        An AlfEnvironment instance.
    """
    assert environment_name in {
        "highway-v0", "merge-v0", "roundabout-v0", "intersection-v0",
        "parking-v0"
    }, "wrong highway environment name"

    gym_spec = gym.spec(environment_name)
    gym_env = gym_spec.make()

    if env_config is None:
        default_env_config = {
            "observation": {
                "type":
                    "Kinematics",
                "vehicles_count":
                    5,
                "features": [
                    "presence", "x", "y", "vx", "vy", "cos_h", "sin_h"
                ],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute":
                    False,
                "order":
                    "sorted"
            },
            "action": {
                "type": "ContinuousAction"
            }
        }
        env_config = default_env_config

    gym_env.configure(env_config)
    gym_env.reset()

    # currently flatten the observations, will support other ways later
    gym_env = FlattenObservation(gym_env)
    gym_env = RemoveActionEnvInfo(gym_env)
    gym_env = ActionScalarization(gym_env)

    # In the original environment, the last step due to time limit is not
    # differentiated from those due to other reasons (e.g. crash):
    # https://github.com/eleurent/highway-env/blob/ede285567a164a58b5bf8a78f1a6792f5a13a3fb/highway_env/envs/highway_env.py#L97-L99
    # Here we -1 on top of the max steps specified by config["duration"] and
    # use the time_limit_wrapper from alf to handle the last step correctly.
    if not max_episode_steps:
        max_episode_steps = gym_env.config["duration"] - 1

    max_episode_steps = min(gym_env.config["duration"] - 1, max_episode_steps)

    return suite_gym.wrap_env(
        gym_env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers)
