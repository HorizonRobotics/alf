# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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
"""Tests for sarsa_algorithm.py."""

from absl import logging
from absl.testing import parameterized
import functools
import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.ppo_algorithm_test import unroll
from alf.algorithms.sarsa_algorithm import SarsaAlgorithm
from alf.environments.suite_unittest import ActionType, PolicyUnittestEnv
from alf.networks import (ActorDistributionNetwork,
                          ActorDistributionRNNNetwork, ActorNetwork,
                          ActorRNNNetwork, StableNormalProjectionNetwork,
                          CriticNetwork, CriticRNNNetwork)
from alf.utils import common
from alf.utils.math_ops import clipped_exp

DEBUGGING = True


def _create_algorithm(env, sac, use_rnn, on_policy, priority_replay):
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    fc_layer_params = (16, 16)
    continuous_projection_net_ctor = functools.partial(
        alf.networks.NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=clipped_exp)

    if use_rnn:
        if sac:
            actor_net = functools.partial(
                ActorDistributionRNNNetwork,
                fc_layer_params=fc_layer_params,
                lstm_hidden_size=(4, ),
                continuous_projection_net_ctor=continuous_projection_net_ctor)
        else:
            actor_net = functools.partial(
                ActorRNNNetwork,
                fc_layer_params=fc_layer_params,
                lstm_hidden_size=(4, ))
        critic_net = functools.partial(
            CriticRNNNetwork,
            joint_fc_layer_params=fc_layer_params,
            lstm_hidden_size=(4, ))
    else:
        if sac:
            actor_net = functools.partial(
                ActorDistributionNetwork,
                fc_layer_params=fc_layer_params,
                continuous_projection_net_ctor=continuous_projection_net_ctor)
        else:
            actor_net = functools.partial(
                ActorNetwork, fc_layer_params=fc_layer_params)

        critic_net = functools.partial(
            CriticNetwork, joint_fc_layer_params=fc_layer_params)

    config = TrainerConfig(
        root_dir="dummy",
        unroll_length=2,
        initial_collect_steps=12 * 128 * 5,
        use_rollout_state=True,
        mini_batch_length=1,
        mini_batch_size=256,
        num_updates_per_train_iter=1,
        whole_replay_buffer_training=False,
        clear_replay_buffer=False,
        priority_replay=priority_replay,
        debug_summaries=DEBUGGING,
        summarize_grads_and_vars=DEBUGGING,
        summarize_action_distributions=DEBUGGING)

    return SarsaAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        env=env,
        config=config,
        epsilon_greedy=0.1,
        calculate_priority=priority_replay,
        on_policy=on_policy,
        ou_stddev=0.2,
        ou_damping=0.5,
        actor_network_ctor=actor_net,
        critic_network_ctor=critic_net,
        actor_optimizer=alf.optimizers.AdamTF(lr=5e-3),
        critic_optimizer=alf.optimizers.AdamTF(lr=2e-2),
        alpha_optimizer=alf.optimizers.AdamTF(lr=2e-2),
        debug_summaries=DEBUGGING)


class SarsaTest(parameterized.TestCase, alf.test.TestCase):
    # TODO: on_policy=True is very unstable, try to figure out the possible
    # reason.
    @parameterized.parameters(
        dict(on_policy=False, sac=False), dict(on_policy=False, use_rnn=False),
        dict(on_policy=False, use_rnn=True), dict(priority_replay=True))
    def test_sarsa(self,
                   on_policy=False,
                   sac=True,
                   use_rnn=False,
                   priority_replay=False):
        logging.info(
            "sac=%d on_policy=%s use_rnn=%s" % (sac, on_policy, use_rnn))
        env_class = PolicyUnittestEnv
        iterations = 500
        num_env = 128
        if on_policy:
            num_env = 128
        steps_per_episode = 12
        env = env_class(
            num_env, steps_per_episode, action_type=ActionType.Continuous)
        eval_env = env_class(
            100, steps_per_episode, action_type=ActionType.Continuous)

        algorithm = _create_algorithm(
            env,
            on_policy=on_policy,
            sac=sac,
            use_rnn=use_rnn,
            priority_replay=priority_replay)

        env.reset()
        eval_env.reset()
        for i in range(iterations):
            algorithm.train_iter()

            eval_env.reset()
            eval_time_step = unroll(eval_env, algorithm, steps_per_episode - 1)
            logging.log_every_n_seconds(
                logging.INFO,
                "%d reward=%f" % (i, float(eval_time_step.reward.mean())),
                n_seconds=1)

        self.assertAlmostEqual(
            1.0, float(eval_time_step.reward.mean()), delta=0.3)


if __name__ == '__main__':
    alf.test.main()
