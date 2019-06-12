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

import gym
import numpy as np
import gin

from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from alf.environments.suite_socialbot import ProcessPyEnvironment
from alf.environments.mario_wrappers import MarioXReward, \
    LimitedDiscreteActions, FrameSkip, ProcessFrame84, FrameStack, FrameFormat

try:
    import retro
except ImportError:
    retro = None


def is_available():
    return retro is not None


@gin.configurable
def load(game,
         state=None,
         discount=1.0,
         wrap_with_process=True,
         frame_skip=4,
         frame_stack=4,
         data_format='channels_last',
         record=False,
         crop=True,
         max_episode_steps=4500,
         spec_dtype_map=None):
    """Loads the selected mario game and wraps it .
    Args:
        game: Name for the environment to load.
        state: game state (level)
        wrap_with_process: Whether wrap env in a process
        discount: Discount to use for the environment.
        frame_skip: the frequency at which the agent experiences the game
        frame_stack: Stack k last frames
        data_format：one of `channels_last` (default) or `channels_first`.
                    The ordering of the dimensions in the inputs.
        record: Record the gameplay , see retro.retro_env.RetroEnv.record
               `False` for not record otherwise record to current working directory or
               specified director
        crop: whether to crop frame to fixed size
        max_episode_steps: max episode step limit
        spec_dtype_map: A dict that maps gym specs to tf dtypes to use as the
            default dtype for the tensors. An easy way how to configure a custom
            mapping through Gin is to define a gin-configurable function that returns
            desired mapping and call it in your Gin config file, for example:
            `suite_socialbot.load.spec_dtype_map = @get_custom_mapping()`.

    Returns:
        A PyEnvironmentBase instance.
    """
    if spec_dtype_map is None:
        spec_dtype_map = {gym.spaces.Box: np.float32}

    if max_episode_steps is None:
        max_episode_steps = 0

    def env_ctor():
        env_args = [game, state] if state else [game]
        env = retro.make(*env_args, record=record)
        buttons = env.buttons
        env = MarioXReward(env)
        if frame_skip:
            env = FrameSkip(env, frame_skip)
        env = ProcessFrame84(env, crop=crop)
        if frame_stack:
            env = FrameStack(env, frame_stack)
        env = FrameFormat(env, data_format=data_format)
        env = LimitedDiscreteActions(env, buttons)
        return suite_gym.wrap_env(
            env,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=(),
            env_wrappers=(),
            spec_dtype_map=spec_dtype_map,
            auto_reset=False)

    # wrap each env in a new process when parallel envs are used
    # since it cannot create multiple emulator instances per process
    if wrap_with_process:
        process_env = ProcessPyEnvironment(
            lambda: env_ctor())
        process_env.start()
        py_env = wrappers.PyEnvironmentBaseWrapper(process_env)
    else:
        py_env = env_ctor()
    return py_env
