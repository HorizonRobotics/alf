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
from functools import partial
import torch
import torch.distributions as td
import unittest

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.sac_algorithm import ActionType as SacActionType
from alf.algorithms.rl_algorithm_test import MyEnv
from alf.data_structures import StepType, TimeStep
from alf.environments.suite_unittest import (PolicyUnittestEnv, ActionType,
                                             MixedPolicyUnittestEnv)
from alf.networks import ActorDistributionNetwork, CriticNetwork, QNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.nest.utils import NestConcat
from alf.algorithms.ppo_algorithm_test import unroll
from alf.utils import common, dist_utils, tensor_utils
from alf.utils.math_ops import clipped_exp
from alf.tensor_specs import BoundedTensorSpec, TensorSpec


class SACAlgorithmTestInit(alf.test.TestCase):
    def test_sac_algorithm_init(self):
        observation_spec = BoundedTensorSpec((10, ))
        discrete_action_spec = BoundedTensorSpec((), dtype='int64')
        continuous_action_spec = [
            BoundedTensorSpec((3, )),
            BoundedTensorSpec((10, ))
        ]

        universal_q_network = partial(
            QNetwork, preprocessing_combiner=NestConcat())
        critic_network = partial(
            CriticNetwork, action_preprocessing_combiner=NestConcat())

        # q_network instead of critic_network is needed
        self.assertRaises(
            AssertionError,
            SacAlgorithm,
            observation_spec=observation_spec,
            action_spec=discrete_action_spec,
            q_network_cls=None)

        sac = SacAlgorithm(
            observation_spec=observation_spec,
            action_spec=discrete_action_spec,
            q_network_cls=QNetwork)
        self.assertEqual(sac._act_type, SacActionType.Discrete)
        self.assertEqual(sac.train_state_spec.actor, ())
        self.assertEqual(sac.train_state_spec.action.actor_network, ())

        # critic_network instead of q_network is needed
        self.assertRaises(
            AssertionError,
            SacAlgorithm,
            observation_spec=observation_spec,
            action_spec=continuous_action_spec,
            critic_network_cls=None)

        sac = SacAlgorithm(
            observation_spec=observation_spec,
            action_spec=continuous_action_spec,
            critic_network_cls=critic_network)
        self.assertEqual(sac._act_type, SacActionType.Continuous)
        self.assertEqual(sac.train_state_spec.action.critic, ())

        # action_spec order is incorrect
        self.assertRaises(
            AssertionError,
            SacAlgorithm,
            observation_spec=observation_spec,
            action_spec=(continuous_action_spec, discrete_action_spec),
            q_network_cls=universal_q_network)

        sac = SacAlgorithm(
            observation_spec=observation_spec,
            action_spec=(discrete_action_spec, continuous_action_spec),
            q_network_cls=universal_q_network)
        self.assertEqual(sac._act_type, SacActionType.Mixed)
        self.assertEqual(sac.train_state_spec.actor, ())

    def test_sac_algorithm_init_for_eval(self):
        observation_spec = BoundedTensorSpec((10, ))
        continuous_action_spec = [
            BoundedTensorSpec((3, )),
            BoundedTensorSpec((10, ))
        ]
        # None critic_network_cls could also mean predict_step only.
        alf.config("RLAlgorithm", is_eval=True)
        sac = SacAlgorithm(
            observation_spec=observation_spec,
            action_spec=continuous_action_spec,
            critic_network_cls=None)
        self.assertTrue(sac._is_eval)
        self.assertEqual(sac._critic_networks, None)


class SACAlgorithmTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((True, 1), (False, 3))
    def test_sac_algorithm(self, use_naive_parallel_network, reward_dim):
        num_env = 4
        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=1,
            mini_batch_length=2,
            mini_batch_size=64,
            initial_collect_steps=500,
            whole_replay_buffer_training=False,
            clear_replay_buffer=False)
        env_class = PolicyUnittestEnv
        steps_per_episode = 13
        env = env_class(
            num_env,
            steps_per_episode,
            action_type=ActionType.Continuous,
            reward_dim=reward_dim)

        eval_env = env_class(
            100,
            steps_per_episode,
            action_type=ActionType.Continuous,
            reward_dim=reward_dim)

        obs_spec = env._observation_spec
        action_spec = env._action_spec
        reward_spec = env._reward_spec

        fc_layer_params = (10, 10)

        continuous_projection_net_ctor = partial(
            alf.networks.NormalProjectionNetwork,
            state_dependent_std=True,
            scale_distribution=True,
            std_transform=clipped_exp)

        actor_network = partial(
            ActorDistributionNetwork,
            fc_layer_params=fc_layer_params,
            continuous_projection_net_ctor=continuous_projection_net_ctor)

        critic_network = partial(
            CriticNetwork,
            joint_fc_layer_params=fc_layer_params,
            use_naive_parallel_network=use_naive_parallel_network)

        alg = SacAlgorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network,
            critic_network_cls=critic_network,
            use_entropy_reward=reward_dim == 1,
            epsilon_greedy=0.1,
            env=env,
            config=config,
            actor_optimizer=alf.optimizers.Adam(lr=1e-2),
            critic_optimizer=alf.optimizers.Adam(lr=1e-2),
            alpha_optimizer=alf.optimizers.Adam(lr=1e-2),
            reproduce_locomotion=True,
            debug_summaries=False,
            name="MySAC")

        eval_env.reset()
        for i in range(700):
            alg.train_iter()
            if i < config.initial_collect_steps:
                continue
            eval_env.reset()
            eval_time_step = unroll(eval_env, alg, steps_per_episode - 1)
            logging.log_every_n_seconds(
                logging.INFO,
                "%d reward=%f" % (i, float(eval_time_step.reward.mean())),
                n_seconds=1)

        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=0.3)


class SACAlgorithmTestDiscrete(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((True, 1), (False, 3))
    def test_sac_algorithm_discrete(self, use_naive_parallel_network,
                                    reward_dim):
        num_env = 1
        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=1,
            mini_batch_length=2,
            mini_batch_size=64,
            initial_collect_steps=500,
            whole_replay_buffer_training=False,
            clear_replay_buffer=False,
        )
        env_class = PolicyUnittestEnv

        steps_per_episode = 13
        env = env_class(
            num_env,
            steps_per_episode,
            action_type=ActionType.Discrete,
            reward_dim=reward_dim)

        eval_env = env_class(
            100,
            steps_per_episode,
            action_type=ActionType.Discrete,
            reward_dim=reward_dim)

        obs_spec = env._observation_spec
        action_spec = env._action_spec
        reward_spec = env._reward_spec

        fc_layer_params = (10, 10)

        q_network = partial(
            QNetwork,
            fc_layer_params=fc_layer_params,
            use_naive_parallel_network=use_naive_parallel_network)

        alg2 = SacAlgorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            q_network_cls=q_network,
            use_entropy_reward=(reward_dim == 1),
            epsilon_greedy=0.1,
            env=env,
            config=config,
            critic_optimizer=alf.optimizers.Adam(lr=1e-3),
            alpha_optimizer=alf.optimizers.Adam(lr=1e-2),
            debug_summaries=False,
            name="MySAC")

        eval_env.reset()
        for i in range(700):
            alg2.train_iter()
            if i < config.initial_collect_steps:
                continue
            eval_env.reset()
            eval_time_step = unroll(eval_env, alg2, steps_per_episode - 1)
            logging.log_every_n_seconds(
                logging.INFO,
                "%d reward=%f" % (i, float(eval_time_step.reward.mean())),
                n_seconds=1)

        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=0.2)


class SACAlgorithmTestMixed(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((True, ), (False, ))
    def test_sac_algorithm_mixed(self, use_naive_parallel_network):
        num_env = 1
        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=1,
            mini_batch_length=2,
            mini_batch_size=64,
            initial_collect_steps=500,
            whole_replay_buffer_training=False,
            clear_replay_buffer=False,
        )
        env_class = MixedPolicyUnittestEnv

        steps_per_episode = 13
        env = env_class(num_env, steps_per_episode)

        eval_env = env_class(100, steps_per_episode)

        obs_spec = env._observation_spec
        action_spec = env._action_spec

        fc_layer_params = (10, 10, 10)

        continuous_projection_net_ctor = partial(
            alf.networks.NormalProjectionNetwork,
            state_dependent_std=True,
            scale_distribution=True,
            std_transform=clipped_exp)

        actor_network = partial(
            ActorDistributionNetwork,
            fc_layer_params=fc_layer_params,
            continuous_projection_net_ctor=continuous_projection_net_ctor)

        q_network = partial(
            QNetwork,
            preprocessing_combiner=NestConcat(),
            fc_layer_params=fc_layer_params,
            use_naive_parallel_network=use_naive_parallel_network)

        alg2 = SacAlgorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            actor_network_cls=actor_network,
            q_network_cls=q_network,
            epsilon_greedy=0.1,
            env=env,
            config=config,
            actor_optimizer=alf.optimizers.Adam(lr=1e-2),
            critic_optimizer=alf.optimizers.Adam(lr=1e-2),
            alpha_optimizer=alf.optimizers.Adam(lr=1e-2),
            debug_summaries=False,
            name="MySAC")

        eval_env.reset()
        for i in range(700):
            alg2.train_iter()
            if i < config.initial_collect_steps:
                continue

            eval_env.reset()
            eval_time_step = unroll(eval_env, alg2, steps_per_episode - 1)
            logging.log_every_n_seconds(
                logging.INFO,
                "%d reward=%f" % (i, float(eval_time_step.reward.mean())),
                n_seconds=1)

        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=0.2)


if __name__ == '__main__':
    alf.test.main()
