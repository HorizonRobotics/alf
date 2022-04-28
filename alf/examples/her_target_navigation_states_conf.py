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

import functools

import alf

from alf.algorithms.agent import Agent
from alf.algorithms.data_transformer import HindsightExperienceTransformer
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm
from alf.environments import suite_gym, suite_socialbot
from alf.nest.utils import NestConcat
from alf.networks import ActorNetwork, CriticNetwork
from alf.optimizers import AdamTF

from alf.examples import ddpg_conf

batch_size = 30
alf.config(
    'create_environment',
    env_load_fn=suite_socialbot.load,
    env_name='SocialBot-PlayGround-v0',
    num_parallel_environments=batch_size)

alf.config('suite_socialbot.load', gym_env_wrappers=[], max_episode_steps=50)
alf.config('suite_gym.wrap_env', image_channel_first=False)

alf.config(
    'GoalTask',
    # Episodic limits
    fail_distance_thresh=1000,

    # Goal & distraction object setup
    random_goal=False,
    goal_name="target_ball",
    distraction_list=['coke_can', 'car_wheel'],
    distraction_penalty=20,
    distraction_penalty_distance_thresh=0.4,
    random_range=5,

    # Observation
    polar_coord=True,

    # Goal conditioned task setup
    success_with_angle_requirement=False,
    goal_conditioned=True,
    end_episode_after_success=0,
    end_on_hitting_distraction=0,
    max_steps=50,
    move_goal_during_episode=0,
    reset_time_limit_on_success=0,
    use_aux_achieved=True,
    use_curriculum_training=0)

alf.config('GazeboAgent', goal_conditioned=True)

alf.config(
    'PlayGround',
    use_image_observation=False,
    with_language=False,
    max_steps=50)

# Networks
alf.config('ActorNetwork', preprocessing_combiner=NestConcat())
alf.config(
    'CriticNetwork',
    observation_preprocessing_combiner=NestConcat(),
    action_preprocessing_combiner=NestConcat())

hidden_layers = (256, 256, 256)
actor_network_cls = functools.partial(
    ActorNetwork, fc_layer_params=hidden_layers)
critic_network_cls = functools.partial(
    CriticNetwork, joint_fc_layer_params=hidden_layers)

optimizer = AdamTF(lr=2e-4, gradient_clipping=0.5)
alf.config('Agent', rl_algorithm_cls=DdpgAlgorithm, optimizer=optimizer)

alf.config(
    'DdpgAlgorithm',
    actor_network_ctor=actor_network_cls,
    critic_network_ctor=critic_network_cls,
    rollout_random_action=0.3,
    target_update_period=8)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    data_transformer_ctor=HindsightExperienceTransformer,
    unroll_length=100,
    initial_collect_steps=10000,
    mini_batch_length=2,
    mini_batch_size=5000,
    num_updates_per_train_iter=40,
    replay_buffer_length=200000,
    summary_interval=20,
    num_iterations=1500,
    num_checkpoints=10,
    evaluate=True,
    eval_interval=500,
    num_eval_episodes=50)

# HER
alf.config('ReplayBuffer', keep_episodic_info=True)
alf.config('HindsightExperienceTransformer', her_proportion=0.8)
alf.config('l2_dist_close_reward_fn', threshold=0.5)

# Finer grain tensorboard summaries plus local action distribution
# TrainerConfig.summarize_action_distributions=True
# TrainerConfig.summary_interval=1
# TrainerConfig.update_counter_every_mini_batch=True
# TrainerConfig.summarize_grads_and_vars=1
# TrainerConfig.summarize_output=True
# summarize_gradients.with_histogram=False
# summarize_variables.with_histogram=False
