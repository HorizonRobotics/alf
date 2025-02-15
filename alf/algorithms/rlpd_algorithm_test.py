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
from alf.algorithms.rlpd_algorithm import RlpdAlgorithm
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


class RlpdAlgorithmTestInit(alf.test.TestCase):
    def test_rlpd_algorithm_init(self):
        observation_spec = BoundedTensorSpec((10, ))
        continuous_action_spec = [
            BoundedTensorSpec((3, )),
            BoundedTensorSpec((10, ))
        ]

        critic_network = partial(
            CriticNetwork, action_preprocessing_combiner=NestConcat())

        self.assertRaises(
            AssertionError,
            RlpdAlgorithm,
            observation_spec=observation_spec,
            action_spec=continuous_action_spec,
            critic_network_cls=None)

        rlpd = RlpdAlgorithm(
            observation_spec=observation_spec,
            action_spec=continuous_action_spec,
            critic_network_cls=critic_network)
        self.assertEqual(rlpd._act_type, SacActionType.Continuous)
        self.assertEqual(rlpd.train_state_spec.action.critic, ())

    def test_rlpd_algorithm_init_for_eval(self):
        observation_spec = BoundedTensorSpec((10, ))
        continuous_action_spec = [
            BoundedTensorSpec((3, )),
            BoundedTensorSpec((10, ))
        ]
        # None critic_network_cls could also mean predict_step only.
        alf.config("RLAlgorithm", is_eval=True)
        rlpd = RlpdAlgorithm(
            observation_spec=observation_spec,
            action_spec=continuous_action_spec,
            critic_network_cls=None)
        self.assertTrue(rlpd._is_eval)
        self.assertEqual(rlpd._critic_networks, None)


class RlpdAlgorithmTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((True, 1, 1), (False, 3, 2), (True, 1, 1, 1),
                              (True, 2, 1, 2, True))
    def test_rlpd_algorithm(self,
                            use_naive_parallel_network,
                            reward_dim,
                            num_sampled_critic_targets,
                            actor_utd=None,
                            critic_utd=None,
                            use_bootstrap_critics=False):
        num_env = 4
        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=1,
            mini_batch_length=2,
            mini_batch_size=64,
            initial_collect_steps=100,
            num_updates_per_train_iter=5,
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
            use_fc_ln=True,
            use_naive_parallel_network=use_naive_parallel_network)

        alg = RlpdAlgorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network,
            critic_network_cls=critic_network,
            use_entropy_reward=reward_dim == 1,
            num_critic_replicas=3,
            num_sampled_critic_targets=num_sampled_critic_targets,
            use_bootstrap_critics=use_bootstrap_critics,
            actor_utd=actor_utd,
            critic_utd=critic_utd,
            epsilon_greedy=0.1,
            env=env,
            config=config,
            actor_optimizer=alf.optimizers.Adam(lr=1e-2),
            critic_optimizer=alf.optimizers.Adam(lr=1e-2),
            alpha_optimizer=alf.optimizers.Adam(lr=1e-2),
            debug_summaries=False,
            name="MyRLPD")

        eval_env.reset()
        for i in range(200):
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


if __name__ == '__main__':
    alf.test.main()
