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
"""Suite for loading OpenAI `Safety Gym <https://openai.com/blog/safety-gym/>`_ environments.

**NOTE**: Mujoco requires separated installation.

(gym >= 0.10, and mujoco>=1.50)

Follow the instructions at:

https://github.com/openai/mujoco-py


Several general facts about the provided benchmark environments:

1. All have distance-based dense rewards
2. All have continual goals: after reaching a goal, the goal is reset but the
   layout keeps the same until timeout.
3. Layouts are randomized before episodes begin
4. Costs are indicator binaries (0 or 1). Every positive cost will be binarized
   to 1. Thus the total cost will be 1 if any component cost is positive.
5. level 0 has no constraints; level 1 has some unsafe elements; level 2 has
   very dense unsafe elements.

See https://github.com/openai/safety-gym/blob/f31042f2f9ee61b9034dd6a416955972911544f5/safety_gym/envs/engine.py#L97
for a complete list of default configurations.

################## Customized Safety Gym ####################
To use our own customized safet_gym env in the paper
"Towards Safe Reinforcement Learning with a Safety Editor Policy", Yu et al 2022,
please install https://github.com/hnyu/safety-gym.git instead of the official
safety gym.

You might need to install apt packages below when running safety_gym for the
first time:

.. code-block:: python

    apt install patchelf libglew-dev
"""

try:
    import mujoco_py
    import safety_gym
except ImportError:
    mujoco_py = None
    safety_gym = None

import numpy as np
import copy
import gym
from typing import Callable, List

import alf
from alf.environments import suite_gym
from alf.environments.alf_wrappers import NonEpisodicAgent


def is_available():
    """Check if both ``mujoco_py`` and ``safety_gym`` have been installed."""
    return (mujoco_py is not None and safety_gym is not None)


class CompleteEnvInfo(gym.Wrapper):
    """Always set the complete set of information so that the env info has a
    fixed shape (no matter whether some event occurs or not), which is required
    by ALF.

    The current safety gym env only adds a key to env info when the corresponding
    event is triggered, see:
    https://github.com/openai/safety-gym/blob/f31042f2f9ee61b9034dd6a416955972911544f5/safety_gym/envs/engine.py#L1242
    """

    def __init__(self, env, env_name):
        super().__init__(env)
        # env info keys are retrieved from:
        # https://github.com/openai/safety-gym/blob/master/safety_gym/envs/engine.py
        self._env_info_keys = [
            'cost_exception',
            'goal_met',
            'cost'  # this is the summed overall cost
        ]
        if not self._is_level0_env(env_name):
            # for level 1 and 2 envs, there are constraints cost info
            self._env_info_keys += [
                'cost_vases_contact', 'cost_pillars', 'cost_buttons',
                'cost_gremlins', 'cost_vases_displace', 'cost_vases_velocity',
                'cost_hazards'
            ]
        self._default_env_info = self._generate_default_env_info()

    def _is_level0_env(self, env_name):
        return "0-v" in env_name

    def _generate_default_env_info(self):
        env_info = {}
        for key in self._env_info_keys:
            if key == "goal_met":
                env_info[key] = False
            else:
                env_info[key] = np.float32(0.)
        return env_info

    def step(self, action):
        """Take a step through the environment the returns the complete set of
        env info, regardless of whether the corresponding event is enabled or not.
        """
        env_info = copy.copy(self._default_env_info)
        obs, reward, done, info = self.env.step(action)
        env_info.update(info)
        return obs, reward, done, env_info


class VectorReward(gym.Wrapper):
    """This wrapper makes the env returns a reward vector of length 3. The three
    dimensions are:

    1. distance-improvement reward indicating the delta smaller distances of
       agent<->box and box<->goal for "push" tasks, or agent<->goal for
       "goal"/"button" tasks.
    2. negative binary cost where -1 means that at least one constraint has been
       violated at the current time step (constraints vary depending on env
       configurations).

    All rewards are the higher the better.
    """

    REWARD_DIMENSION = 2

    def __init__(self, env, sparse_reward):
        """
        Args:
            env: env being wrapped
            sparse_reward: if True, then the first reward dim will only be a
                binary value indicating a success.
        """
        super().__init__(env)
        self._reward_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=[self.REWARD_DIMENSION])
        self._sparse_reward = sparse_reward

    def step(self, action):
        """Take one step through the environment and obtains several rewards.

        Args:
            action (np.array):

        Returns:
            tuple:
            - obs (np.array): a flattened observation vector that contains
              all enabled sensors' data
            - rewards (np.array): a reward vector of length ``REWARD_DIMENSION``.
              See the class docstring for their meanings.
            - done (bool): whether the episode has ended
            - info (dict): a dict of additional env information
        """
        obs, reward, done, info = self.env.step(action)
        # Get the second and third reward from ``info``
        cost_reward = -info["cost"]
        success_reward = float(info["goal_met"])
        if self._sparse_reward:
            reward = success_reward
        return obs, np.array([reward, cost_reward],
                             dtype=np.float32), done, info

    @property
    def reward_space(self):
        return self._reward_space


@alf.configurable(blacklist=['env'])
class RGBRenderWrapper(gym.Wrapper):
    """A ``metadata`` field should've been defined in the original safety gym env;
    otherwise video recording will be disabled. See
    https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/video_recorder.py#L41

    Also the original env needs a ``camera_id`` if "rgb_array" mode is used for
    rendering, which is incompatible with our ``ALFEnvironment`` interfaces.
    Here we wrap ``render()`` with a customizable camera mode.
    """
    _metadata = {'render.modes': ["rgb_array", "human"]}

    def __init__(self, env, width=None, height=None, camera_mode="fixedfar"):
        """
        Args:
            width (int): the width of rgb image
            height (int): the height of rbg image
            camera_mode (str): one of ('fixednear', 'fixedfar', 'vision', 'track')
        """
        super().__init__(env)
        # self.metadata will first inherit subclass's metadata
        self.metadata.update(self._metadata)
        self._width = width
        self._height = height
        self._camera_mode = camera_mode

    def render(self, mode="human"):
        camera_id = self.unwrapped.model.camera_name2id(self._camera_mode)
        render_kwargs = dict(mode=mode, camera_id=camera_id)
        if self._width is not None:
            render_kwargs["width"] = self._width
        if self._height is not None:
            render_kwargs["height"] = self._height
        return self.env.render(**render_kwargs)


@alf.configurable
class EpisodicWrapper(gym.Wrapper):
    """The original safety gym is non-episodic: a new goal will be re-spawned
    after a goal is achieved. This wrapper makes the env episodic.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info["goal_met"]:
            done = True
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


@alf.configurable
def load(environment_name: str,
         env_id: int = None,
         discount: float = 1.0,
         max_episode_steps: int = None,
         unconstrained: bool = False,
         sparse_reward: bool = False,
         episodic: bool = False,
         gym_env_wrappers: List[Callable] = (),
         alf_env_wrappers: List[Callable] = ()):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a ``TimeLimit`` wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name: Name for the environment to load.
        env_id: A scalar ``Tensor`` of the environment ID of the time step.
        discount: Discount to use for the environment.
        max_episode_steps: If None the ``max_episode_steps`` will be set to
            the default step limit -1 defined in the environment. If 0, no
            ``TimeLimit`` wrapper will be used.
        unconstrained: if True, the suite will be used just as an
            unconstrained environment. The reward will always be scalar without
            including constraints.
        sparse_reward: If True, only give reward when reaching a goal.
        episodic: whether terminate the episode when a goal is achieved.
            Note that if True, both ``EpisodicWrapper`` and ``NonEpisodicAgent``
            wrapper will be used to simulate an infinite horizon even though the
            success rate is computed on per-goal basis. This is for approximating
            an average constraint reward objective. ``EpisodicWrapper`` first
            returns ``done=True`` to signal the end of an episode, and ``NonEpisodicAgent``
            replaces ``discount=0`` with ``discount=1``.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        alf_env_wrappers: Iterable with references to wrapper classes to use on
            the torch environment.

    Returns:
        AlfEnvironment:
    """

    # We can directly make the env here because none of the safety gym tasks
    # is registered with a ``max_episode_steps`` argument (the
    # ``gym.wrappers.time_limit.TimeLimit`` won't be applied). But each task
    # will inherently manage the time limit through ``env.num_steps``.
    env = gym.make(environment_name)

    # fill all env info with default values
    env = CompleteEnvInfo(env, environment_name)

    # make vector reward
    if not unconstrained:
        env = VectorReward(env, sparse_reward)

    env = RGBRenderWrapper(env)

    if episodic:
        env = EpisodicWrapper(env)
        alf_env_wrappers = alf_env_wrappers + (NonEpisodicAgent, )

    # Have to -1 on top of the original env max steps here, because the
    # underlying gym env will output ``done=True`` when reaching the time limit
    # ``env.num_steps`` (before the ``AlfGymWrapper``), which is incorrect:
    # https://github.com/openai/safety-gym/blob/f31042f2f9ee61b9034dd6a416955972911544f5/safety_gym/envs/engine.py#L1302
    if max_episode_steps is None:
        max_episode_steps = env.num_steps - 1
        max_episode_steps = min(env.num_steps - 1, max_episode_steps)

    return suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers)
