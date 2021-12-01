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

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.handcrafted_algorithm import SimpleCarlaAlgorithm
from alf.environments import suite_carla
from alf.environments.alf_wrappers import ActionObservationWrapper
from alf.environments.carla_controller import VehicleController

alf.config('CameraSensor', image_size_x=200, image_size_y=100, fov=135)

alf.config(
    'create_environment',
    env_name='Town01',
    env_load_fn=suite_carla.load,
    num_parallel_environments=4)

alf.config(
    'suite_carla.Player',
    # camera is on for visualization purpose during play
    with_camera_sensor=True,
    # uncomment to turn on BEV for visualization
    # with_bev_sensor=True,
    with_gnss_sensor=False,
    with_imu_sensor=False,
    with_radar_sensor=False,
    sparse_reward=False,
    allow_negative_distance_reward=True,
    # uncomment to use collision and red light penalty
    # max_collision_penalty=20.,
    # max_red_light_penalty=20.,
    # use PID controller
    controller_ctor=VehicleController)

alf.config('Agent', rl_algorithm_cls=SimpleCarlaAlgorithm)

# the rest of training config is the mostly the same as the typical setting
# to ensure results are comparable when used as a baseline

alf.config(
    'CarlaEnvironment',
    vehicle_filter='vehicle.*',
    # uncomment to add other vehicles and walkers
    # num_other_vehicles=20,
    # num_walkers=20,
    # uncomment to use day length and dynamic weather
    # day_length=1000,
    # max_weather_length=500,
)

alf.config('suite_carla.load', wrappers=[ActionObservationWrapper])

# training config
alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False,
    initial_collect_steps=3000,
    mini_batch_length=4,
    unroll_length=10,
    mini_batch_size=64,
    num_updates_per_train_iter=1,
    num_iterations=100,
    num_checkpoints=20,
    evaluate=False,
    debug_summaries=True,
    summarize_grads_and_vars=False,
    summary_interval=100,
    num_summaries=1000,
    # use a small replay buffer as there is no training
    replay_buffer_length=10,
    summarize_action_distributions=True)
