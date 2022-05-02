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

# default ddpg config

import alf
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm

alf.config(
    'TrainerConfig',
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    algorithm_ctor=DdpgAlgorithm)
