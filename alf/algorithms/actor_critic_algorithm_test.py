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

from absl import logging
import numpy as np

import unittest
import tensorflow as tf

from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork

from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.environments.suite_unittest import ValueUnittestEnv
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import RNNPolicyUnittestEnv
from alf.policies.training_policy import TrainingPolicy


class GradientTypeTest(unittest.TestCase):
    def test_gradient_tape(self):
        """
        Test whether we can use GradientTape in multiple sections
        """
        self._tape = tf.GradientTape()
        x = tf.constant(3.0)
        with self._tape:
            self._tape.watch(x)
            y = 3 * x

        with self._tape:
            y = 3 * y

        with self._tape:
            y = 2 * y

        dldx = self._tape.gradient(y, x)

        self.assertEqual(float(dldx), 18.0)


class ActorCriticAlgorithmTest(unittest.TestCase):
    def _create_policy(self,
                       env,
                       train_interval=1,
                       use_rnn=False,
                       learning_rate=1e-1):
        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        time_step_spec = env.time_step_spec()

        global_step = tf.Variable(
            0, dtype=tf.int64, trainable=False, name="global_step")

        if use_rnn:
            actor_net = ActorDistributionRnnNetwork(
                observation_spec,
                action_spec,
                input_fc_layer_params=(),
                output_fc_layer_params=(),
                lstm_size=(4, ))
            value_net = ValueRnnNetwork(
                observation_spec,
                input_fc_layer_params=(),
                output_fc_layer_params=(),
                lstm_size=(4, ))
        else:
            actor_net = ActorDistributionNetwork(
                observation_spec, action_spec, fc_layer_params=())
            value_net = ValueNetwork(observation_spec, fc_layer_params=())

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        algorithm = ActorCriticAlgorithm(
            action_spec=action_spec,
            actor_network=actor_net,
            value_network=value_net,
            loss=ActorCriticLoss(action_spec=action_spec, gamma=1.0),
            optimizer=optimizer)

        policy = TrainingPolicy(
            algorithm=algorithm,
            time_step_spec=time_step_spec,
            train_interval=train_interval,
            train_step_counter=global_step)
        return policy

    def test_actor_critic_value(self):
        batch_size = 1
        steps_per_episode = 13
        env = ValueUnittestEnv(batch_size, steps_per_episode)

        policy = self._create_policy(env)
        policy_state = policy.get_initial_state(batch_size)
        time_step = env.reset()
        for i in range(100):
            for _ in range(steps_per_episode):
                policy_step = policy.action(time_step, policy_state)
                policy_state = policy_step.state
                time_step = env.step(policy_step.action)

            if (i + 1) % 10 == 0:
                print('value=%s' % float(
                    tf.reduce_mean(policy._training_info[-1].info.value)))

        # It is surprising that although the expected discounted reward
        # should be steps_per_episode/2, using one step actor critic will
        # converge to a different value, which is steps_per_episode-1
        self.assertAlmostEqual(
            steps_per_episode - 1,
            float(tf.reduce_mean(policy._training_info[-1].info.value)),
            delta=0.1)

    def test_actor_critic_policy(self):
        batch_size = 100
        steps_per_episode = 13
        env = PolicyUnittestEnv(batch_size, steps_per_episode)

        policy = self._create_policy(env)
        policy_state = policy.get_initial_state(batch_size)
        time_step = env.reset()
        for i in range(100):
            for _ in range(steps_per_episode):
                reward = time_step.reward
                policy_step = policy.action(time_step, policy_state)
                policy_state = policy_step.state
                time_step = env.step(policy_step.action)

            if (i + 1) % 10 == 0:
                print('reward=%s' % float(tf.reduce_mean(reward)))

        self.assertAlmostEqual(1.0, float(tf.reduce_mean(reward)), delta=1e-3)

    def test_actor_critic_rnn_policy(self):
        batch_size = 100
        steps_per_episode = 5
        gap = 3
        env = RNNPolicyUnittestEnv(batch_size, steps_per_episode, gap)

        policy = self._create_policy(
            env, train_interval=8, use_rnn=True, learning_rate=2e-2)
        policy_state = policy.get_initial_state(batch_size)
        time_step = env.reset()
        for i in range(200):
            for _ in range(steps_per_episode):
                reward = time_step.reward
                policy_step = policy.action(time_step, policy_state)
                policy_state = policy_step.state
                time_step = env.step(policy_step.action)

            if (i + 1) % 10 == 0:
                print('reward=%s' % float(tf.reduce_mean(reward)))

        self.assertAlmostEqual(1.0, float(tf.reduce_mean(reward)), delta=1e-3)


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()
    unittest.main()
