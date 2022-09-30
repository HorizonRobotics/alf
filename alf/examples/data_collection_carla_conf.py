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
from alf.environments.carla_env.carla_agents import SimpleNavigationAgent
from alf.algorithms.handcrafted_algorithm import SimpleCarlaAlgorithm
from alf.environments import suite_carla
from alf.environments.alf_wrappers import (ActionObservationWrapper,
                                           ScalarRewardWrapper)

import alf
from alf.algorithms.agent import Agent

from alf.examples import carla_conf

# This is an example config file for data collection in CARLA.

# the desired replay buffer size for collection
# 100 is just an example. Should set it to he actual desired size.
replay_buffer_length = 100

# the desired environment for data collection
alf.config(
    'create_environment',
    env_name='Town01',
    env_load_fn=suite_carla.load,
    num_parallel_environments=1)

# some additional time for each trajectory to compensate for the time
# on stopping on red light
additional_time = 500
initial_collect_steps = replay_buffer_length + 1000
num_env_steps = replay_buffer_length
data_collection_mode = True
overwrite_policy_output = True
enable_buffer_checkpoint = True

# 5.6 m/s = 20 km/h
alf.config('SimpleNavigationAgent', target_speed=5.6)
alf.config('RLAlgorithm', overwrite_policy_output=overwrite_policy_output)

# config Player for data collection:
# turn on data collection mode and give some additional time
alf.config(
    'suite_carla.Player',
    data_collection_mode=data_collection_mode,
    additional_time=additional_time)

alf.config(
    'suite_carla.Player',
    sparse_reward=True,
    sparse_reward_interval=50,
    allow_negative_distance_reward=True,
    max_collision_penalty=100,
    max_red_light_penalty=100.,
    terminate_upon_infraction="all",
    with_gnss_sensor=False,
    with_imu_sensor=True,
    with_camera_sensor=False,
    with_radar_sensor=True,
    with_red_light_sensor=True,
    with_obstacle_sensor=True,
    with_dynamic_object_sensor=True,
    min_speed=3.0,
    additional_time=20)

alf.config(
    'CarlaEnvironment',
    vehicle_filter='vehicle.tesla.model3',
    num_other_vehicles=50,
    num_walkers=0,
    # 1000 second day length means 4.5 days in replay buffer of 90000 length
    day_length=0,
    max_weather_length=0,
    step_time=0.1)
wrappers = [ActionObservationWrapper, ScalarRewardWrapper]
alf.config('suite_carla.load', wrappers=wrappers)

alf.config(
    'ReplayBuffer',
    enable_checkpoint=enable_buffer_checkpoint,
    keep_episodic_info=True)

# skip representation learning and use simple carla algorithm instead of SAC
# to save computation by aovidng NN forward
alf.config(
    'Agent',
    representation_learner_cls=None,
    rl_algorithm_cls=SimpleCarlaAlgorithm,
    optimizer=None)

# Some relevant parameters in the config for data collection:
# initial_collect_steps, num_env_steps, replay_buffer_length.
# Note that since we set the value of num_env_steps as initial_collect_steps,
# we are only leveraging the initial collection phase for data collection,
# and do not enter the actual training mode beyond the initial collection phase.
alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    initial_collect_steps=initial_collect_steps,
    unroll_length=1,
    num_iterations=0,
    num_env_steps=num_env_steps,
    summary_interval=100,
    replay_buffer_length=replay_buffer_length,
    num_checkpoints=1)
