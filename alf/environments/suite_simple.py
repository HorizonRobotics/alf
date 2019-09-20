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
import gin

from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers

from alf.environments.suite_socialbot import ProcessPyEnvironment
from alf.environments.simple.noisy_array import NoisyArray
from alf.environments.wrappers import FrameSkip, FrameStack


@gin.configurable
def load(game,
         env_args=dict(),
         discount=1.0,
         wrap_with_process=True,
         frame_skip=None,
         frame_stack=None,
         max_episode_steps=0,
         spec_dtype_map=None):

    if spec_dtype_map is None:
        spec_dtype_map = {gym.spaces.Box: np.float32}

    def env_ctor():
        if game == "NoisyArray":
            env = NoisyArray(**env_args)
        else:
            assert False, "No such simple environment!"
        if frame_skip:
            env = FrameSkip(env, frame_skip)
        if frame_stack:
            env = FrameStack(env, stack_size=frame_stack)
        return suite_gym.wrap_env(
            env,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=(),
            env_wrappers=(),
            spec_dtype_map=spec_dtype_map,
            auto_reset=True)

    # wrap each env in a new process when parallel envs are used
    # since it cannot create multiple emulator instances per process
    if wrap_with_process:
        process_env = ProcessPyEnvironment(lambda: env_ctor())
        process_env.start()
        py_env = wrappers.PyEnvironmentBaseWrapper(process_env)
    else:
        py_env = env_ctor()
    return py_env
