# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import gym3
from procgen import ProcgenGym3Env

from alf.environments.alf_gym3_wrapper import AlfGym3Wrapper


def _load_procgen(env_name: str, batch_size: int = 1, render: bool = False):
    env = ProcgenGym3Env(
        num=batch_size,
        env_name=env_name,
        render_mode='rgb_array' if render else None)
    # This extracts the 'rgb' field in the observation dictionary
    # every time when step() is called, so that the observation is
    # an image that our algorithm can consume.
    env = gym3.ExtractDictObWrapper(env, 'rgb')
    if render:
        env = gym3.ViewerWrapper(env, info_key='rgb')
    return env


def _extract_frame(env: gym3.Env):
    return env.get_info()[0]['rgb']


def load(env_name: str, batch_size: int = 1):
    """Load the Procgen environment

    Args:
    
        env_name: the name of the procgen environment, such as 'goldrun',
            'bossfight', etc.

        batch_size: the number of parallel environments to run simultaneously.

    """
    return AlfGym3Wrapper(
        _load_procgen(env_name, batch_size, render=False),
        image_channel_first=True,
        ignored_info_keys=['rgb'],
        support_force_reset=True,
        render_activator=lambda: _load_procgen(
            env_name, batch_size, render=True),
        frame_extractor=_extract_frame)


load.batched = True
