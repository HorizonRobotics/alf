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

from functools import partial

import alf
from alf.environments import suite_metadrive

# Environment Configuration
alf.config(
    'create_environment',
    env_load_fn=suite_metadrive.load,
    env_name='RandomMap',
    num_parallel_environments=12)

alf.config(
    'suite_metadrive.load',
    scenario_num=5000,
    crash_penalty=50.0,
    success_reward=200.0)
