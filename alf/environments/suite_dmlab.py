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

import functools
import gym
import numpy as np

import alf
from alf.environments import suite_gym, alf_wrappers, process_environment
from alf.environments.utils import UnwrappedEnvChecker

_unwrapped_env_checker_ = UnwrappedEnvChecker()

# `DeepmindLab` are required,
#   see `https://github.com/deepmind/lab` to build `DeepmindLab`

try:
    import deepmind_lab
except ImportError:
    deepmind_lab = None


def is_available():
    return deepmind_lab is not None


@alf.configurable
def action_discretize(action_spec,
                      look_left_right_pixels_per_frame=(-20, 20),
                      look_down_up_pixels_per_frame=(-10, 10),
                      strafe_left_right=(-1, 1),
                      move_back_forward=(-1, 1),
                      fire=(),
                      jump=(1, ),
                      crouch=(1, ),
                      **kwargs):
    """Discretize action from action_spec

    TODO: action combinations

    Mapping all valid action values to discrete action

    original deepmind lab environment action_spec:

    .. code-block:: python

        [{'max': 512, 'min': -512, 'name': 'LOOK_LEFT_RIGHT_PIXELS_PER_FRAME'},
        {'max': 512, 'min': -512, 'name': 'LOOK_DOWN_UP_PIXELS_PER_FRAME'},
        {'max': 1, 'min': -1, 'name': 'STRAFE_LEFT_RIGHT'},
        {'max': 1, 'min': -1, 'name': 'MOVE_BACK_FORWARD'},
        {'max': 1, 'min': 0, 'name': 'FIRE'},
        {'max': 1, 'min': 0, 'name': 'JUMP'},
        {'max': 1, 'min': 0, 'name': 'CROUCH'}]

    and discretized actions:

    .. code-block::

        0  -> [20,0,0,0,0,0,0] (look left 20 pixels),
        1  -> [-20,0,0,0,0,0,0] (look right 20 pixels),
        ...,
        m  -> [0,0,0,-1,0,0,0] (move back),
        m+1-> [0,0,0,1,0,0,0] (move forward) ,
        ...,
        n  -> [0,0,0,0,1,1,0] (jump and fire),
        ...

    see `SuiteDMLabTest.test_action_discretize` in `suite_dmlab_test.py` for examples

    Args:
        action_spec (list(dict)): action spec
        look_left_right_pixels_per_frame (iterable|str): look left or look right pixels
        look_down_up_pixels_per_frame (iterable|str): look down or look up pixels
        strafe_left_right (iterable|str): strafe left or strafe right
        move_back_forward (iterable|str): move back or move forward
        fire (iterable|str): fire values
        jump (iterable|str): jump values
        crouch (iterable|str): crouch values
        kwargs (dict): other config for actions
    Returns:
        actions (list[numpy.array]): discrete actions
    """
    actions = []

    config = dict(
        look_left_right_pixels_per_frame=look_left_right_pixels_per_frame,
        look_down_up_pixels_per_frame=look_down_up_pixels_per_frame,
        strafe_left_right=strafe_left_right,
        move_back_forward=move_back_forward,
        fire=fire,
        jump=jump,
        crouch=crouch)
    config.update(kwargs)
    config = {key.upper(): value for key, value in config.items()}

    for i, spec in enumerate(action_spec):
        val_min = spec['min']
        val_max = spec['max']
        values = config.get(spec['name'], None)

        if values is None:
            values = list(range(val_min, val_max + 1))
        elif isinstance(values, str):
            values = eval(values)

        for value in values:
            if value < val_min or value > val_max or value == 0:
                continue
            action = np.zeros([len(action_spec)], np.intc)
            action[i] = value
            actions.append(action)

    return actions


@alf.configurable
class DeepmindLabEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 scene,
                 action_repeat=4,
                 observation='RGB_INTERLEAVED',
                 config={},
                 renderer='hardware'):
        """Create an deepmind_lab env

        Args:
            scene (str): script for the deepmind_lab env. See available script:
                `<https://github.com/deepmind/lab/tree/master/game_scripts/levels>`_
            action_repeat (int): the interval at which the agent experiences the game
            observation (str):  observation format. See doc about the available observations:
                `<https://github.com/deepmind/lab/blob/master/docs/users/python_api.md>`_
            config (dict): config for env
            renderer (str): 'software' or 'hardware'. If set to 'hardware', EGL or GLX is
                used for rendering. Make sure you have GPU if you use 'hardware'.
        """
        super(DeepmindLabEnv, self).__init__()

        self._action_repeat = action_repeat
        self._observation = observation
        self._lab = deepmind_lab.Lab(
            scene, [self._observation], config=config, renderer=renderer)

        self._lab.reset()
        action_spec = self._lab.action_spec()
        action_list = action_discretize(action_spec)
        self.action_space = gym.spaces.Discrete(len(action_list))
        self._action_list = action_list

        obs = self._lab.observations()[observation]
        self.observation_space = gym.spaces.Box(
            0, 255, obs.shape, dtype=np.uint8)
        self._last_obs = obs

    def step(self, action):
        reward = self._lab.step(
            self._action_list[action], num_steps=self._action_repeat)
        terminal = not self._lab.is_running()
        obs = None if terminal else self._lab.observations()[self._observation]
        self._last_obs = obs if obs is not None else np.copy(self._last_obs)
        return self._last_obs, reward, terminal, dict()

    def reset(self):
        self._lab.reset()
        self._last_obs = self._lab.observations()[self._observation]
        return self._last_obs

    def seed(self, seed=None):
        self._lab.reset(seed=seed)

    def close(self):
        self._lab.close()

    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            return self._last_obs
        else:
            super().render(mode=mode)  # just raise an exception


@alf.configurable
def load(scene,
         env_id=None,
         discount=1.0,
         frame_skip=4,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         wrap_with_process=False,
         max_episode_steps=None):
    """Load deepmind lab envs.
    Args:
        scene (str): script for the deepmind_lab env. See available script:
            `<https://github.com/deepmind/lab/tree/master/game_scripts/levels>`_
        env_id (int): (optional) ID of the environment.
        discount (float): Discount to use for the environment.
        frame_skip (int): the frequency at which the agent experiences the game
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers,
            classes to use directly on the gym environment.
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.
        wrap_with_process (bool): Whether wrap env in a process
        max_episode_steps (int): max episode step limit
    Returns:
        An AlfEnvironment instance.
    """
    _unwrapped_env_checker_.check_and_update(wrap_with_process)

    if max_episode_steps is None:
        max_episode_steps = 0

    def env_ctor(env_id=None):
        return suite_gym.wrap_env(
            DeepmindLabEnv(scene=scene, action_repeat=frame_skip),
            env_id=env_id,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,
            alf_env_wrappers=alf_env_wrappers)

    if wrap_with_process:
        process_env = process_environment.ProcessEnvironment(
            functools.partial(env_ctor))
        process_env.start()
        torch_env = alf_wrappers.AlfEnvironmentBaseWrapper(process_env)
    else:
        torch_env = env_ctor(env_id=env_id)
    return torch_env
