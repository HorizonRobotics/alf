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
import time
import numpy as np

import unittest
import tensorflow as tf
import gin.tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import RNNPolicyUnittestEnv
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.environments.suite_unittest import ActionType
from alf.algorithms.actor_critic_algorithm import create_ac_algorithm


class OnPolicyDriverTest(unittest.TestCase):
    def setUp(self) -> None:
        gin.parse_config([
            "ActorDistributionRnnNetwork.lstm_size=(4,)",
            "ValueRnnNetwork.lstm_size=(4,)", "ActorCriticLoss.gamma=1.0"
        ])
        super().setUp()
        tf.random.set_seed(0)

    def test_actor_critic_policy(self):
        batch_size = 100
        steps_per_episode = 13
        env = PolicyUnittestEnv(batch_size, steps_per_episode)
        # We need to wrap env using TFPyEnvironment because the methods of env
        # has side effects (e.g, env._current_time_step can be changed)
        env = TFPyEnvironment(env)

        algorithm = create_ac_algorithm(
            env, actor_fc_layers=(), value_fc_layers=(), learning_rate=1e-1)
        driver = OnPolicyDriver(env, algorithm, train_interval=1)
        eval_driver = OnPolicyDriver(env, algorithm, training=False)

        driver.run = tf.function(driver.run)

        t0 = time.time()
        driver.run(max_num_steps=1300 * batch_size)
        print("time=%s" % (time.time() - t0))

        env.reset()
        time_step, _ = eval_driver.run(max_num_steps=4 * batch_size)
        print("reward=%s" % tf.reduce_mean(time_step.reward))
        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(time_step.reward)), delta=1e-2)

    def test_actor_critic_continuous_policy(self):
        batch_size = 100
        steps_per_episode = 13
        env = PolicyUnittestEnv(
            batch_size, steps_per_episode, action_type=ActionType.Continuous)
        # We need to wrap env using TFPyEnvironment because the methods of env
        # has side effects (e.g, env._current_time_step can be changed)
        env = TFPyEnvironment(env)

        algorithm = create_ac_algorithm(
            env, actor_fc_layers=(), value_fc_layers=(), learning_rate=1e-2)
        driver = OnPolicyDriver(env, algorithm, train_interval=1)
        eval_driver = OnPolicyDriver(env, algorithm, training=False)

        driver.run = tf.function(driver.run)

        t0 = time.time()
        driver.run(max_num_steps=1300 * batch_size)
        print("time=%s" % (time.time() - t0))

        env.reset()
        time_step, _ = eval_driver.run(max_num_steps=4 * batch_size)
        print("reward=%s" % tf.reduce_mean(time_step.reward))
        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(time_step.reward)), delta=5e-2)

    def test_actor_critic_rnn_policy(self):
        batch_size = 100
        steps_per_episode = 5
        gap = 3

        env = RNNPolicyUnittestEnv(batch_size, steps_per_episode, gap)
        # We need to wrap env using TFPyEnvironment because the methods of env
        # has side effects (e.g, env._current_time_step can be changed)
        env = TFPyEnvironment(env)

        algorithm = create_ac_algorithm(
            env,
            actor_fc_layers=(),
            value_fc_layers=(),
            use_rnns=True,
            learning_rate=2e-2)
        driver = OnPolicyDriver(env, algorithm, train_interval=8)
        eval_driver = OnPolicyDriver(env, algorithm, training=False)

        driver.run = tf.function(driver.run)

        t0 = time.time()
        driver.run(max_num_steps=1000 * batch_size)
        logging.info("time=%s" % (time.time() - t0))

        env.reset()
        time_step, _ = eval_driver.run(max_num_steps=4 * batch_size)
        logging.info("reward=%s" % tf.reduce_mean(time_step.reward))
        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(time_step.reward)), delta=1e-2)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    unittest.main()
