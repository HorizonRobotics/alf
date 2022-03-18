# Copyright (c) 2022 Horizon Robotics. All Rights Reserved.
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
"""Suite for loading RLBench::

    "RLBench: The Robot Learning Benchmark \& Learning Environment", James et al., 2020

You need to first follow the `installation instructions <https://github.com/stepjam/RLBench>`_
to install all software dependencies. We use a **forked version** at
`https://github.com/HorizonRobotics/RLBench`_ where some customized changes are made.
So when pip installing RLBench, please use our url.

For headless rendering with VirtualGL if image inputs are trained on a machine
without displays, also follow their instructions on how to set it up.

Qt required by RLBench might have conflicts with ``cv2``. To resolve the this,
we need to switch to a headless version:

.. code-block::python

    pip uninstall opencv-python
    pip install opencv-python-headless

Note that the headless version won't allow us to use GUI with ``cv2``, e.g.,
``cv2.imshow()``.

RLBench also have a conflict with ``matplotlib`` regarding xserver when multiple
envs are run in parallel. Our ``VideoRecorder`` (``alf.summary.render``) uses
``matplotlib`` for rendering, so when recording a video, currently only one
RLBench env can be used.

Tip: when GPU rendering is needed, setting ``CUDA_VISIBLE_DEVICES`` to other gpus
will separate the GPU training and rendering. Also PyRep seems not reactive to
SIGINT (ctrl+c), so to kill the training job, we need SIGQUIT (ctrl+\).
(https://github.com/stepjam/PyRep/issues/12)
"""

import alf

try:
    import rlbench
    import rlbench.gym
    from rlbench.gym.rlbench_env import RLBenchEnv
    from rlbench.observation_config import CameraConfig, ObservationConfig
    import alf.environments.rlbench_custom_tasks
except ImportError:
    rlbench = None
    RLBenchEnv = None
    ObservationConfig = None

from typing import List
import gym
import numpy as np

from alf.environments.alf_wrappers import AlfEnvironmentBaseWrapper
from alf.environments.alf_environment import AlfEnvironment
from alf.environments import suite_gym


def is_available():
    return rlbench is not None


class RLBenchLanguageWrapper(gym.Wrapper):
    """A language command wrapper for RLBench environments.

    For some tasks, the language command can be used to infer the target object.
    Each episode will randomly sample a command from the candidate set returned by
    the wrapped env's ``reset()``. This command will given to the robot one character
    per time step. We assume that the vocabulary is 26 lower-case letters plus ' '.
    """

    VOCAB_SIZE = 128

    def __init__(self, env: RLBenchEnv, language: bool = False):
        """
        Args:
            env: an RLBench gym env instance
            language: whether use language commands
        """
        assert isinstance(env, RLBenchEnv), (
            "This wrapper only accepts an RLBenchEnv instance! "
            "Make sure to put this wrapper as the first gym wrapper.")
        super().__init__(env)

        if language:
            obs_space = {
                'observation': env.observation_space,
                'command': gym.spaces.Discrete(self.VOCAB_SIZE)
            }
            self.observation_space = gym.spaces.Dict(obs_space)

        self._language = language
        self._cmd_chars = []

    def reset(self, **kwargs):
        commands, obs = self.env.reset(**kwargs)
        if not self._language:
            return obs
        else:
            command = np.random.choice(commands)
            self._cmd_chars = [ord(c) for c in command]
            if np.amax(self._cmd_chars) > 127:
                raise ValueError(
                    "Character out of range. The unicode of "
                    "character should be in [0, 127]: %s" % command)
            return self._augment_obs_with_command(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self._language:
            obs = self._augment_obs_with_command(obs)
        return obs, reward, done, info

    def _augment_obs_with_command(self, obs):
        observation = {'observation': obs}
        if self._cmd_chars:
            observation['command'] = self._cmd_chars.pop(0)
        else:
            observation['command'] = np.int64(0)
        return observation


class TaskVariationWrapper(gym.Wrapper):
    """This wrapper enables variations (different goals) for a task (if applicable).
    For example, in the ``reach_target`` task, the target object might have a
    different color for a different variation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(variation=True, **kwargs)


"""
class StackRGBDepth(gym.Wrapper):
    ""For any pair of RGB and depth from the same camera, this wrapper stacks
    them to form a 'rgbd' image, add it to the observation dict, and remove the
    RGB and depth from the observation. This wrapper also converts rgb uint8 to
    float in [0,1] before stacking with depth.
    ""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Dict)
        self.observation_space = gym.spaces.Dict(self._stack_observations(
            self.observation_space.spaces, self._stack_spaces))

    def _stack_observations(self, observation, stack_fn):
        # ``observation`` can be either a dict of arrays or a ``gym.spaces.Dict``
        obs = dict()
        excluded_keys = []
        for k, v in observation.items():
            if k.endswith("_depth"):
                camera = '_'.join(k.split('_')[:-1])
                rgb = observation.get(camera + "_rgb", None)
                if rgb is not None:
                    obs[camera + '_rgbd'] = stack_fn(rgb, v)
                    excluded_keys.append(camera + '_rgb')
                    continue
            obs[k] = v
        for k in excluded_keys:
            obs.pop(k)
        return obs

    def _stack_spaces(self, rgb, d):
        # [H,W,D]
        return gym.spaces.Box(
            low=d.low.min(), high=d.high.max(), shape=d.shape + (4,),
            dtype=d.dtype)

    def _stack_arrays(self, rgb, d):
        # [H,W,D]
        return np.concatenate([rgb, np.expand_dims(d, -1)], axis=-1)

    def reset(self):
        obs = self.env.reset()
        obs = self._stack_observations(obs, self._stack_arrays)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._stack_observations(obs, self._stack_arrays)
        return obs, reward, done, info
"""


@alf.configurable
def load(environment_name: str,
         env_id: int = None,
         discount: float = 1.,
         max_episode_steps: int = None,
         observe_language: bool = False,
         task_variation: bool = False,
         observation_config: ObservationConfig = None,
         gym_env_wrappers: List[gym.Wrapper] = (),
         alf_env_wrappers: List[AlfEnvironmentBaseWrapper] = ()
         ) -> AlfEnvironment:
    """Loads the selected environment and wraps it with the specified wrappers.

    Currently, the fields for the vision-based environment observation are hardcoded.
    Potentially we can customize RLBench to return different vision&depth sensorary
    data.

    Args:
        environment_name: Name for the environment to load. For a complete list of
            100 tasks, take a look at `https://github.com/HorizonRobotics/RLBench/tree/master/rlbench/tasks`_.
            For a particular task file ``<task_name>.py``, we can use one of
            ``["<task_name>-state-v0", "<task_name>-vision-v0", "<task_name>-v0"]`` as
            the environment name. The first two are envs with predefined observation configs,
            where the first returns a low-dim flattend state vector and the second
            returns a dictionary with all five camera rgb images and the state vector.
            When using the third env name, it is required that the user also provides
            ``observation_config`` for customization.
        env_id: A scalar ``Tensor`` of the environment ID of the time step.
        discount: Discount to use for the environment.
        max_episode_steps: If None the ``max_episode_steps`` will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no ``timestep_limit`` set in the environment's spec.
        observe_language: whether the observation contains language commands or not.
        task_variation: whether enables task variations across episodes. If True, then
            ``observe_language`` must also be True to let the agent know the task goal.
            For example, in ``reach_target`` task, the variation is which object to reach,
            and the language command will describe the goal.
        observation_config: configuration object for observation. Using this config,
            we can easily customize which sensors to turn on in the env observation.
            For all options, please see ``rlbench.observation_config.py``. This arg
            is only used when ``environment_name`` is ``"<task_name>-v0"``.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        alf_env_wrappers: Iterable with references to wrapper classes to use on
            the torch environment.
    """

    gym_spec = gym.spec(environment_name)
    env = gym_spec.make(obs_config=observation_config)

    if max_episode_steps is None:
        if gym_spec.max_episode_steps is not None:
            max_episode_steps = gym_spec.max_episode_steps
        else:
            max_episode_steps = 0

    env = RLBenchLanguageWrapper(env, observe_language)

    if task_variation:
        env = TaskVariationWrapper(env)

    return suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        # RLBench returns 'channels_last' rgb, so we always need to transpose it
        image_channel_first=True)
