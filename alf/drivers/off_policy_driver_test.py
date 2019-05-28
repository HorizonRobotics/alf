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

import gin.tf
import unittest
import tensorflow as tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork

from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.actor_rnn_network import ActorRnnNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork
from alf.drivers.off_policy_driver import OffPolicyDriver
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import ActionType
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.utils import common


def create_ddpg_algorithm(env, use_rnn=False, learning_rate=1e-3):
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    actor_fc_layers = (100,)
    critic_fc_layers = (100,)
    if use_rnn:
        actor_net = ActorRnnNetwork(
            observation_spec,
            action_spec,
            input_fc_layer_params=actor_fc_layers,
            output_fc_layer_params=(),
            lstm_size=(4,))
        critic_net = CriticRnnNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=(),
            action_fc_layer_params=(),
            output_fc_layer_params=(),
            joint_fc_layer_params=critic_fc_layers,
            lstm_size=(4,))
    else:
        actor_net = ActorNetwork(
            observation_spec, action_spec,
            fc_layer_params=actor_fc_layers)
        critic_net = CriticNetwork(
            (observation_spec, action_spec),
            joint_fc_layer_params=critic_fc_layers)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    return DdpgAlgorithm(
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        ou_damping=0.15,
        ou_stddev=0.2,
        optimizer=optimizer,
        debug_summaries=True)


def create_sac_algorithm(env, use_rnn=False, learning_rate=1e-3):
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    actor_fc_layers = (100,)
    critic_fc_layers = (100,)
    if use_rnn:
        actor_net = ActorDistributionRnnNetwork(
            observation_spec,
            action_spec,
            input_fc_layer_params=actor_fc_layers,
            output_fc_layer_params=(),
            lstm_size=(4,))
        critic_net = CriticRnnNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=(),
            action_fc_layer_params=(),
            output_fc_layer_params=(),
            joint_fc_layer_params=critic_fc_layers,
            lstm_size=(4,))
    else:
        actor_net = ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layers)
        critic_net = CriticNetwork(
            (observation_spec, action_spec),
            joint_fc_layer_params=critic_fc_layers)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    return SacAlgorithm(
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        optimizer=optimizer,
        debug_summaries=True)


class OffPolicyDriverTest(unittest.TestCase):

    def test_sac(self):
        batch_size = 4
        steps_per_episode = 13
        env = PolicyUnittestEnv(
            batch_size, steps_per_episode,
            action_type=ActionType.Continuous)
        env = TFPyEnvironment(env)
        algorithm = create_sac_algorithm(env)
        driver = OffPolicyDriver(
            env, algorithm,
            debug_summaries=True)
        eval_driver = OffPolicyDriver(env, algorithm, training=False)
        driver.run = tf.function(driver.run)

        t0 = time.time()
        driver.run(max_num_steps=8000 * batch_size)
        print("time=%s" % (time.time() - t0))

        # def _record_run():
        #     t0 = time.time()
        #     driver.run(max_num_steps=8000 * batch_size)
        #     print("time=%s" % (time.time() - t0))
        #
        # common.run_under_record_context(_record_run, '~/tmp/sac', 10, 1)

        time_step, _ = eval_driver.run(max_num_steps=4 * batch_size)
        print("reward=%s" % tf.reduce_mean(time_step.reward))

        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(time_step.reward)), delta=2e-1)

    def test_ddpg(self):
        batch_size = 4
        steps_per_episode = 13
        env = PolicyUnittestEnv(
            batch_size, steps_per_episode,
            action_type=ActionType.Continuous)
        env = TFPyEnvironment(env)
        algorithm = create_ddpg_algorithm(env)
        driver = OffPolicyDriver(
            env, algorithm,
            debug_summaries=True)
        eval_driver = OffPolicyDriver(env, algorithm, training=False)
        driver.run = tf.function(driver.run)

        t0 = time.time()
        driver.run(max_num_steps=4000 * batch_size)
        print("time=%s" % (time.time() - t0))

        time_step, _ = eval_driver.run(max_num_steps=4 * batch_size)
        print("reward=%s" % tf.reduce_mean(time_step.reward))
        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(time_step.reward)), delta=2e-1)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    unittest.main()
