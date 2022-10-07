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
import torch

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.bc_algorithm import BcAlgorithm

from alf.examples import carla_conf
from alf.environments import suite_carla
from alf.environments.alf_wrappers import ActionObservationWrapper, ScalarRewardWrapper
from alf.environments.carla_env.carla_utils import CarlaMergedActionWrapper

# +++++++++++++++++++ ENV +++++++++++++++++++++++++++
alf.config(
    'suite_carla.Player',
    # Not yet able to successfully train with sparse reward.
    # hybrid_reward_mode=None,
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
    step_time=0.1,
    # no_rendering_mode=True,
)

# for data collection
# from alf.examples import data_collection_carla_conf

alf.config(
    'CarlaMergedActionWrapper', throttle_damping=0.1, brake_damping=0.01)

wrappers = [ActionObservationWrapper, ScalarRewardWrapper]
alf.config(
    'create_environment',
    env_name='Town01',
    env_load_fn=suite_carla.load,
    num_parallel_environments=1)

alf.config('suite_carla.load', wrappers=wrappers)

# +++++++++++++++++++ Agent +++++++++++++++++++++++++++

latest_checkpoint_interval = 10000

alf.config('Agent', rl_algorithm_cls=BcAlgorithm)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False)

lr = 1e-4
num_iterations = 1000000
rl_train_after_update_steps = num_iterations
initial_collect_steps = 0
offline_mini_batch_length = 1
offline_mini_batch_size = 64
offline_buffer_dir = "./hybrid_rl/replay_buffer_data/carla-replay-buffer-mini100"

fc_layers_params = (256, ) * 2
encoding_dim = 256
use_batch_normalization = False
activation = torch.relu_

proj_net = partial(
    alf.networks.NormalProjectionNetwork,
    state_dependent_std=False,
    scale_distribution=False,
    std_transform=alf.math.clipped_exp)

actor_distribution_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    input_preprocessors=carla_conf.create_input_preprocessors(
        encoding_dim, use_batch_normalization),
    preprocessing_combiner=alf.layers.NestConcat(
        nest_mask=carla_conf.create_input_preprocessor_masks()),
    fc_layer_params=fc_layers_params,
    continuous_projection_net_ctor=proj_net)

optimizer_actor = alf.optimizers.AdamTF(lr=lr)

alf.config(
    'BcAlgorithm',
    actor_network_cls=actor_distribution_network_cls,
    actor_optimizer=optimizer_actor)

alf.config(
    "TrainerConfig",
    rl_train_after_update_steps=rl_train_after_update_steps,
    offline_buffer_dir=offline_buffer_dir,
    offline_buffer_length=2,
    initial_collect_steps=initial_collect_steps,
    mini_batch_length=offline_mini_batch_length,
    unroll_length=1,
    mini_batch_size=offline_mini_batch_size,
    num_updates_per_train_iter=1,
    num_iterations=num_iterations,
    num_checkpoints=1,
    evaluate=False,
    eval_interval=10000,
    num_eval_episodes=3,
    debug_summaries=True,
    summarize_grads_and_vars=False,
    summary_interval=1000,
    whole_replay_buffer_training=False)
