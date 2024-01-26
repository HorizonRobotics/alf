# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
from alf.algorithms.td_loss import TDQRLoss
from alf.algorithms.one_step_loss import OneStepTDQRLoss
from alf.algorithms.dsac_algorithm import DSacAlgorithm
from alf.algorithms.doac_algorithm import DOacAlgorithm
from alf.algorithms.rl_algorithm import RLAlgorithm
from alf.algorithms.rl_algorithm_test import MyEnv
from alf.algorithms.sac_algorithm import SacState
from alf.data_structures import TimeStep
from alf.environments.suite_unittest import (PolicyUnittestEnv, ActionType)
from alf.nest.utils import NestConcat
from alf.tensor_specs import TensorSpec
from alf.utils import common
from alf.utils.math_ops import clipped_exp


class DSacAlgorithmTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(
        (
            'iqn',
            1,
            True,
            True,
            True,
            True,
            True,
            False,
        ), ('iqn', 1, False, False, False, True, True, True),
        ('fixed', 3, False, False, False, False, False, True))
    def test_dsac_algorithm(self, tau_type, reward_dim, use_epistemic_alpha,
                            use_naive_parallel_network, use_n_step_td,
                            min_critic_by_critic_mean, nested_observation,
                            use_doac):
        num_env = 1
        config = TrainerConfig(
            root_dir="dummy",
            unroll_length=1,
            mini_batch_length=3,
            mini_batch_size=40,
            initial_collect_steps=500,
            whole_replay_buffer_training=False,
            clear_replay_buffer=False)
        env_class = partial(
            PolicyUnittestEnv, nested_observation=nested_observation)
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
            alf.nn.NormalProjectionNetwork,
            state_dependent_std=True,
            scale_distribution=True,
            std_transform=partial(
                clipped_exp, clip_value_min=-10, clip_value_max=2))

        if nested_observation:
            obs_combiner = NestConcat()
        else:
            obs_combiner = None
        actor_network = partial(
            alf.nn.ActorDistributionNetwork,
            preprocessing_combiner=obs_combiner,
            fc_layer_params=fc_layer_params,
            continuous_projection_net_ctor=continuous_projection_net_ctor)

        num_quantiles = 32
        tau_embedding_dim = 64
        critic_network = partial(
            alf.nn.CriticQuantileNetwork,
            observation_preprocessing_combiner=obs_combiner,
            tau_embedding_dim=tau_embedding_dim,
            obs_act_tau_joint_fc_layer_params=fc_layer_params,
            use_naive_parallel_network=use_naive_parallel_network)

        if use_n_step_td:
            td_qr_loss_ctor = TDQRLoss
        else:
            td_qr_loss_ctor = OneStepTDQRLoss
        critic_loss = partial(td_qr_loss_ctor, num_quantiles=num_quantiles)

        if use_epistemic_alpha:
            alpha_optimizer = None
        else:
            alpha_optimizer = alf.optimizers.Adam(lr=1e-2)

        if use_doac:
            extra_kwargs = dict(explore_delta=1., beta_ub=5.)
            alg_ctor = DOacAlgorithm
        else:
            extra_kwargs = {}
            alg_ctor = DSacAlgorithm

        alg = alg_ctor(
            observation_spec=obs_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            num_quantiles=num_quantiles,
            tau_type=tau_type,
            actor_network_cls=actor_network,
            critic_network_cls=critic_network,
            critic_loss_ctor=critic_loss,
            min_critic_by_critic_mean=min_critic_by_critic_mean,
            use_entropy_reward=reward_dim == 1,
            env=env,
            config=config,
            actor_optimizer=alf.optimizers.Adam(lr=1e-2),
            critic_optimizer=alf.optimizers.Adam(lr=1e-2),
            alpha_optimizer=alpha_optimizer,
            debug_summaries=False,
            name="MyDSAC",
            **extra_kwargs)

        eval_env.reset()
        for i in range(550):
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


def unroll(env, algorithm, steps, epsilon_greedy: float = 0.1):
    """Run `steps` environment steps using QrsacAlgorithm._predict_action()."""
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
