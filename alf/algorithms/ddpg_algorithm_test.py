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

from absl import logging
from absl.testing import parameterized
import functools
import torch
import torch.distributions as td
import unittest

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm
from alf.algorithms.rl_algorithm_test import MyEnv
from alf.data_structures import StepType, TimeStep
from alf.environments.suite_unittest import PolicyUnittestEnv, ActionType
from alf.networks import ActorNetwork, CriticNetwork
from alf.algorithms.ppo_algorithm_test import unroll
from alf.utils import common, dist_utils, tensor_utils
from alf.utils.math_ops import clipped_exp


class DDPGAlgorithmTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((1, 1, None), (2, 3, [1, 2, 3]))
    def test_ddpg_algorithm(self, num_critic_replicas, reward_dim,
                            reward_weights):
        num_env = 128
        num_eval_env = 100
        steps_per_episode = 13
        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=steps_per_episode,
            mini_batch_length=2,
            mini_batch_size=128,
            initial_collect_steps=steps_per_episode,
            whole_replay_buffer_training=False,
            clear_replay_buffer=False,
        )
        env_class = PolicyUnittestEnv

        env = env_class(
            num_env,
            steps_per_episode,
            action_type=ActionType.Continuous,
            reward_dim=reward_dim)

        eval_env = env_class(
            num_eval_env,
            steps_per_episode,
            action_type=ActionType.Continuous,
            reward_dim=reward_dim)

        obs_spec = env._observation_spec
        action_spec = env._action_spec

        fc_layer_params = (16, 16)

        actor_network = functools.partial(
            ActorNetwork, fc_layer_params=fc_layer_params)

        critic_network = functools.partial(
            CriticNetwork,
            output_tensor_spec=env.reward_spec(),
            joint_fc_layer_params=fc_layer_params)

        alg = DdpgAlgorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            reward_spec=env.reward_spec(),
            actor_network_ctor=actor_network,
            critic_network_ctor=critic_network,
            reward_weights=reward_weights,
            epsilon_greedy=0.0,
            env=env,
            config=config,
            num_critic_replicas=num_critic_replicas,
            actor_optimizer=alf.optimizers.Adam(lr=1e-2),
            critic_optimizer=alf.optimizers.Adam(lr=1e-2),
            debug_summaries=False,
            name="MyDDPG")

        for _ in range(500):
            alg.train_iter()

        eval_env.reset()
        eval_time_step = unroll(eval_env, alg, steps_per_episode - 1)
        print(eval_time_step.reward.mean())

        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=2e-1)


if __name__ == '__main__':
    alf.test.main()
