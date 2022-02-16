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

from functools import partial
from alf.environments import suite_carla
from alf.examples import sac_conf

import alf

# This is a example config file for data collection in CARLA.

# the desired replay buffer size for collection
replay_buffer_length = 10000

# the desired environment for data collection
alf.config(
    'create_environment',
    env_name='Town01',
    env_load_fn=suite_carla.load,
    num_parallel_environments=1)

# some additional time for each trajectory to compensate for the time
# on stopping on red light
additional_time = 500
initial_collect_steps = replay_buffer_length
num_env_steps = initial_collect_steps
data_collection_mode = True
overwrite_policy_output = True
enable_buffer_checkpoint = True

# 5.6 m/s = 20 km/h
# alf.config('SimpleNavigationAgent', target_speed=5.6)
alf.config('RLAlgorithm', overwrite_policy_output=overwrite_policy_output)

# config Player for data collection:
# turn on data collection mode and give some additional time
alf.config(
    'suite_carla.Player',
    data_collection_mode=data_collection_mode,
    additional_time=additional_time)

from alf.examples import sac_carla_conf
# training config for data collection:
# initial_collect_steps, num_env_steps, replay_buffer_length
alf.config(
    'TrainerConfig',
    initial_collect_steps=initial_collect_steps,
    num_iterations=0,
    num_env_steps=num_env_steps,
    summary_interval=100,
    replay_buffer_length=replay_buffer_length,
    num_checkpoints=1)

alf.config(
    'ReplayBuffer',
    enable_checkpoint=enable_buffer_checkpoint,
    keep_episodic_info=True)
