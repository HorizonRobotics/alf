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

import alf
from alf.environments import suite_gym, alf_wrappers, process_environment
from alf.environments.gym_wrappers import FrameSkip
from alf.environments.mario_wrappers import MarioXReward, \
    LimitedDiscreteActions, ProcessFrame84
from alf.environments.utils import UnwrappedEnvChecker

_unwrapped_env_checker_ = UnwrappedEnvChecker()

try:
    import retro
except ImportError:
    retro = None


def is_available():
    if retro is None:
        return False
    try:
        retro.data.get_romfile_path('SuperMarioBros-Nes')
    except FileNotFoundError:
        return False
    return True


@alf.configurable
def load(game,
         env_id=None,
         state=None,
         discount=1.0,
         wrap_with_process=False,
         frame_skip=4,
         record=False,
         crop=True,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         max_episode_steps=4500):
    """Loads the selected mario game and wraps it .
    Args:
        game (str): Name for the environment to load.
        env_id (int): (optional) ID of the environment.
        state (str): game state (level)
        wrap_with_process (bool): Whether wrap env in a process
        discount (float): Discount to use for the environment.
        frame_skip (int): the frequency at which the agent experiences the game
        record (bool): Record the gameplay , see retro.retro_env.RetroEnv.record
               `False` for not record otherwise record to current working directory or
               specified director
        crop (bool): whether to crop frame to fixed size
        gym_env_wrappers (Iterable): Iterable with references to gym_wrappers,
            classes to use directly on the gym environment.
        alf_env_wrappers (Iterable): Iterable with references to alf_wrappers
            classes to use on the ALF environment.
        max_episode_steps (int): max episode step limit

    Returns:
        An AlfEnvironment instance.
    """
    _unwrapped_env_checker_.check_and_update(wrap_with_process)

    if max_episode_steps is None:
        max_episode_steps = 0

    def env_ctor(env_id=None):
        env_args = [game, state] if state else [game]
        env = retro.make(*env_args, record=record)
        buttons = env.buttons
        env = MarioXReward(env)
        if frame_skip:
            env = FrameSkip(env, frame_skip)
        env = ProcessFrame84(env, crop=crop)
        env = LimitedDiscreteActions(env, buttons)
        return suite_gym.wrap_env(
            env,
            env_id=env_id,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,
            alf_env_wrappers=alf_env_wrappers,
            auto_reset=True)

    # wrap each env in a new process when parallel envs are used
    # since it cannot create multiple emulator instances per process
    if wrap_with_process:
        process_env = process_environment.ProcessEnvironment(
            functools.partial(env_ctor))
        process_env.start()
        torch_env = alf_wrappers.AlfEnvironmentBaseWrapper(process_env)
    else:
        torch_env = env_ctor(env_id=env_id)
    return torch_env
