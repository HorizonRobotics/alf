# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
import torch
import torch.distributions as td
import unittest

import alf
from alf.utils import common, dist_utils, tensor_utils
from alf.data_structures import StepType, TimeStep
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import ActionType
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.rl_algorithm_test import MyEnv


class SACAlgorithmTest(unittest.TestCase):
    def test_sac_algorithm(self):
        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=1,
            mini_batch_length=2,
            mini_batch_size=64,
            initial_collect_steps=500,
            whole_replay_buffer_training=False,
            clear_replay_buffer=False,
            num_envs=1,
        )
        env_class = PolicyUnittestEnv
        num_env = 1
        steps_per_episode = 13
        env = env_class(
            num_env, steps_per_episode, action_type=ActionType.Continuous)

        eval_env = env_class(
            num_env, steps_per_episode, action_type=ActionType.Continuous)

        obs_spec = env._observation_spec
        action_spec = env._action_spec

        fc_layer_params = [100, 100]

        actor_network = ActorDistributionNetwork(
            obs_spec,
            action_spec,
            fc_layer_params=fc_layer_params,
            continuous_projection_net_ctor=alf.networks.
            NormalProjectionNetwork)

        critic_network = CriticNetwork((obs_spec, action_spec), \
            joint_fc_layer_params=fc_layer_params)

        alg = SacAlgorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            actor_network=actor_network,
            critic_network=critic_network,
            env=env,
            config=config,
            actor_optimizer=torch.optim.Adam(lr=1e-2),
            critic_optimizer=torch.optim.Adam(lr=1e-2),
            alpha_optimizer=torch.optim.Adam(lr=1e-2),
            debug_summaries=False,
            name="MySAC")

        eval_env.reset()
        for _ in range(200):
            alg.train_iter()

        eval_env.reset()
        eval_time_step = unroll(eval_env, alg, steps_per_episode - 1)
        print(eval_time_step.reward.mean())

        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=1e-1)


def unroll(env, algorithm, steps):
    time_step = common.get_initial_time_step(env)
    policy_state = algorithm.get_initial_predict_state(env.batch_size)
    for _ in range(steps):
        policy_state = common.reset_state_if_necessary(
            policy_state, algorithm.get_initial_predict_state(env.batch_size),
            time_step.is_first())
        transformed_time_step = algorithm.transform_timestep(time_step)
        policy_step = algorithm.predict_step(
            transformed_time_step, policy_state, epsilon_greedy=1.0)
        time_step = env.step(policy_step.output)
        policy_state = policy_step.state
    return time_step


if __name__ == '__main__':
    unittest.main()
