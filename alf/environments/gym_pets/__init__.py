# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from gym.envs.registration import register

register(
    id='MBRLCartpole-v0',
    entry_point='alf.environments.gym_pets.envs:CartpoleEnv')

register(
    id='MBRLReacher3D-v0',
    entry_point='alf.environments.gym_pets.envs:Reacher3DEnv')

register(
    id='MBRLPusher-v0', entry_point='alf.environments.gym_pets.envs:PusherEnv')

register(
    id='MBRLHalfCheetah-v0',
    entry_point='alf.environments.gym_pets.envs:HalfCheetahEnv')
