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
"""Suite for loading OpenAI Robotics environments.

**NOTE**: Mujoco requires separated installation.

(gym >= 0.10, and mujoco>=1.50)

Follow the instructions at:

https://github.com/openai/mujoco-py

"""

try:
    import mujoco_py
except ImportError:
    mujoco_py = None

import functools
import numpy as np
import gym

import alf
from alf.environments import suite_gym, alf_wrappers, process_environment
from alf.environments.utils import UnwrappedEnvChecker

_unwrapped_env_checker_ = UnwrappedEnvChecker()


def is_available():
    return mujoco_py is not None


class SparseReward(gym.Wrapper):
    """Convert the original :math:`-1/0` rewards to :math:`0/1`.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        # openai Robotics env will always return ``done=False``
        ob, reward, done, info = self.env.step(action)
        if reward == 0:
            done = True
        return ob, reward + 1, done, info


@alf.configurable
class SuccessWrapper(gym.Wrapper):
    """Retrieve the success info from the environment return.
    """

    def __init__(self, env, since_episode_steps):
        super().__init__(env)
        self._since_episode_steps = since_episode_steps

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._steps += 1

        info["success"] = 0.0
        # only count success after a certain amount of steps
        if self._steps >= self._since_episode_steps and info["is_success"] == 1:
            info["success"] = 1.0

        info.pop("is_success")  # from gym, we remove it here
        return obs, reward, done, info


@alf.configurable
class ObservationClipWrapper(gym.ObservationWrapper):
    """Clip observation values according to OpenAI's baselines.
    """

    def __init__(self, env, min_v=-200., max_v=200.):
        super().__init__(env)
        # NOTE: the code assumes that all spaces under the nested observation
        # space is a Box space.
        self.min_v = min_v
        self.max_v = max_v

    def observation(self, observation):
        if isinstance(observation, dict):
            for k, v in observation.items():
                observation[k] = np.clip(v, self.min_v, self.max_v)
            return observation
        else:
            return np.clip(observation, self.min_v, self.max_v)


@alf.configurable
def load(environment_name,
         env_id=None,
         concat_desired_goal=True,
         discount=1.0,
         max_episode_steps=None,
         sparse_reward=False,
         use_success_wrapper=True,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         wrap_with_process=False):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a ``TimeLimit`` wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name: Name for the environment to load.
        env_id: A scalar ``Tensor`` of the environment ID of the time step.
        discount: Discount to use for the environment.
        max_episode_steps: If None the ``max_episode_steps`` will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no ``timestep_limit`` set in the environment's spec.
        sparse_reward (bool): If True, the game ends once the goal is achieved.
            Rewards will be added by 1, changed from -1/0 to 0/1.
        use_success_wrapper (bool): If True, wraps the environment with the
            SuccessWrapper which will record Success info after a specified
            amount of timesteps.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        alf_env_wrappers: Iterable with references to wrapper classes to use on
            the torch environment.

    Returns:
        An AlfEnvironment instance.
    """
    assert (environment_name.startswith("Fetch")
            or environment_name.startswith("HandManipulate")), (
                "This suite only supports OpenAI's Fetch and ShadowHand envs!")

    _unwrapped_env_checker_.check_and_update(wrap_with_process)

    gym_spec = gym.spec(environment_name)
    env = gym_spec.make()

    if max_episode_steps is None:
        if gym_spec.max_episode_steps is not None:
            max_episode_steps = gym_spec.max_episode_steps
        else:
            max_episode_steps = 0

    def env_ctor(env_id=None):
        return suite_gym.wrap_env(
            env,
            env_id=env_id,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,
            alf_env_wrappers=alf_env_wrappers,
            image_channel_first=False)

    # concat robot's observation and the goal location
    if concat_desired_goal:
        keys = ["observation", "desired_goal"]
        try:  # for modern Gym (>=0.15.3)
            # 0.15.3 has a bug in ``FlattenObservation``, so avoid using it!
            from gym.wrappers import FilterObservation, FlattenObservation
            env = FlattenObservation(FilterObservation(env, keys))
        except ImportError:  # for older gym (<0.15.3)
            from gym.wrappers import FlattenDictWrapper  # pytype:disable=import-error
            env = FlattenDictWrapper(env, keys)
    if use_success_wrapper:
        env = SuccessWrapper(env, max_episode_steps)
    env = ObservationClipWrapper(env)
    if sparse_reward:
        env = SparseReward(env)

    if wrap_with_process:
        process_env = process_environment.ProcessEnvironment(
            functools.partial(env_ctor))
        process_env.start()
        torch_env = alf_wrappers.AlfEnvironmentBaseWrapper(process_env)
    else:
        torch_env = env_ctor(env_id=env_id)

    return torch_env
