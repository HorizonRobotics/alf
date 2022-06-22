# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from gym.envs.registration import registry, register, make, spec

register(
    id='FetchPickAndPlaceAdv-v0',
    entry_point=('alf.environments.adv_fetch_envs.pick_and_place_adv:'
                 'FetchPickAndPlaceAdvEnv'),
    max_episode_steps=100,
)

register(
    id='FetchSlideAdv-v0',
    entry_point=('alf.environments.adv_fetch_envs.slide_adv:'
                 'FetchSlideAdvEnv'),
    max_episode_steps=100,
)

register(
    id='FetchReachAdv-v0',
    entry_point=('alf.environments.adv_fetch_envs.reach_adv:'
                 'FetchReachAdvEnv'),
    max_episode_steps=100,
)

register(
    id='FetchPushAdv-v0',
    entry_point=('alf.environments.adv_fetch_envs.push_adv:'
                 'FetchPushAdvEnv'),
    max_episode_steps=100,
)
