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
"""Suite for simple environments defined by ALF"""

import gym
import numpy as np

import alf
from alf.environments import suite_gym
from alf.environments.simple.noisy_array import NoisyArray
from alf.environments.simple.stochastic_with_risky_branch import StochasticWithRiskyBranch
from alf.environments.object_centric.bouncing_squares import BouncingSquares
from alf.environments.gym_wrappers import FrameSkip, FrameStack


@alf.configurable
def load(game,
         env_id=None,
         env_args=dict(),
         discount=1.0,
         frame_skip=None,
         frame_stack=None,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         max_episode_steps=0):
    """Loads the specified simple game and wraps it.
    Args:
        game (str): name for the environment to load. The game should have been
            defined in the sub-directory ``./simple/``.
        env_args (dict): extra args for creating the game.
        discount (float): discount to use for the environment.
        frame_skip (int): the time interval at which the agent experiences the
            game.
        frame_stack (int): stack so many latest frames as the observation input.
        gym_env_wrappers (list): list of gym env wrappers.
        alf_env_wrappers (list): list of ALF env wrappers.
        max_episode_steps (int): max number of steps for an episode.

    Returns:
        An AlfEnvironment instance.
    """

    if game == "NoisyArray":
        env = NoisyArray(**env_args)
    elif game == "StochasticWithRiskyBranch":
        env = StochasticWithRiskyBranch(**env_args)
    elif game == "BouncingSquares":
        env = BouncingSquares(**env_args)
    else:
        assert False, "No such simple environment!"
    if frame_skip:
        env = FrameSkip(env, frame_skip)
    if frame_stack:
        env = FrameStack(env, stack_size=frame_stack)
    return suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        auto_reset=True)
