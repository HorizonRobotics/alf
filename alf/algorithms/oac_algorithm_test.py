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

from absl import logging
from absl.testing import parameterized
from functools import partial
import torch
import torch.distributions as td
import unittest

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.oac_algorithm import OacAlgorithm, NormalProjectionNetwork
from alf.algorithms.sac_algorithm import SacState
from alf.algorithms.rl_algorithm_test import MyEnv
from alf.data_structures import TimeStep
from alf.environments.suite_unittest import (PolicyUnittestEnv, ActionType)
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.utils import common
from alf.utils.math_ops import clipped_exp


class OACAlgorithmTest(alf.test.TestCase):
    def test_oac_algorithm(self):
        reward_dim = 3
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
            NormalProjectionNetwork,
            state_dependent_std=True,
            scale_distribution=True,
            std_transform=clipped_exp)

        actor_network = partial(
            ActorDistributionNetwork,
            fc_layer_params=fc_layer_params,
            continuous_projection_net_ctor=continuous_projection_net_ctor)

        critic_network = partial(
            CriticNetwork, joint_fc_layer_params=fc_layer_params)

        alg = OacAlgorithm(
            observation_spec=obs_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            actor_network_cls=actor_network,
            critic_network_cls=critic_network,
            use_entropy_reward=reward_dim == 1,
            env=env,
            config=config,
            explore_delta=1.,
            beta_ub=1.,
            actor_optimizer=alf.optimizers.Adam(lr=1e-2),
            critic_optimizer=alf.optimizers.Adam(lr=1e-2),
            alpha_optimizer=alf.optimizers.Adam(lr=1e-2),
            debug_summaries=False,
            name="MyOAC")

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


def unroll(env, algorithm, steps, epsilon_greedy=0.1):
    """Run `steps` environment steps using OacAlgorithm._predict_action()."""
    time_step = common.get_initial_time_step(env)
    policy_state = algorithm.get_initial_predict_state(env.batch_size)
    trans_state = algorithm.get_initial_transform_state(env.batch_size)
    for _ in range(steps):
        policy_state = common.reset_state_if_necessary(
            policy_state, algorithm.get_initial_predict_state(env.batch_size),
            time_step.is_first())
        transformed_time_step, trans_state = algorithm.transform_timestep(
            time_step, trans_state)
        action_dist, action, _, action_state = algorithm._predict_action(
            transformed_time_step.observation,
            policy_state.action,
            epsilon_greedy=epsilon_greedy,
            eps_greedy_sampling=True)
        time_step = env.step(action)
        policy_state = SacState(action=action_state)
    return time_step


if __name__ == '__main__':
    alf.test.main()
