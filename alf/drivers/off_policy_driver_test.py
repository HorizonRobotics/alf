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

import time
import unittest
from absl.testing import parameterized

from absl import logging
import gin.tf
import tensorflow as tf

from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.actor_rnn_network import ActorRnnNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.agents.ddpg.critic_rnn_network import CriticRnnNetwork
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork

from tf_agents.environments.tf_py_environment import TFPyEnvironment

from alf.algorithms.ddpg_algorithm import DdpgAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.drivers.off_policy_driver import OffPolicyDriver
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import ActionType


def create_ddpg_algorithm(env, use_rnn=False, learning_rate=1e-1):
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    if use_rnn:
        actor_net = ActorRnnNetwork(
            observation_spec,
            action_spec,
            input_fc_layer_params=(),
            output_fc_layer_params=(),
            lstm_size=(4,))
        critic_net = CriticRnnNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=(),
            action_fc_layer_params=(),
            output_fc_layer_params=(),
            joint_fc_layer_params=(10,),
            lstm_size=(4,))
    else:
        actor_net = ActorNetwork(
            observation_spec, action_spec, fc_layer_params=())
        critic_net = CriticNetwork(
            (observation_spec, action_spec),
            joint_fc_layer_params=(10, 10))

    actor_optimizer = tf.optimizers.Adam(learning_rate=0.1 * learning_rate)
    critic_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    return DdpgAlgorithm(
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        ou_damping=1.,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        debug_summaries=True)


def create_sac_algorithm(env, use_rnn=False, learning_rate=5e-3):
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
            observation_spec, action_spec, fc_layer_params=actor_fc_layers)
        critic_net = CriticNetwork(
            (observation_spec, action_spec),
            joint_fc_layer_params=critic_fc_layers)

    actor_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    critic_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    alpha_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    return SacAlgorithm(action_spec=action_spec,
                        actor_network=actor_net,
                        critic_network=critic_net,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        alpha_optimizer=alpha_optimizer)


class OffPolicyDriverTest(parameterized.TestCase, unittest.TestCase):

    @parameterized.parameters(create_sac_algorithm, create_ddpg_algorithm, )
    def test_off_policy_algorithm(self, algorithm_ctor):
        batch_size = 100
        steps_per_episode = 13
        env = PolicyUnittestEnv(
            batch_size, steps_per_episode, action_type=ActionType.Continuous)
        env = TFPyEnvironment(env)

        eval_env = PolicyUnittestEnv(
            batch_size, steps_per_episode, action_type=ActionType.Continuous)
        eval_env = TFPyEnvironment(eval_env)

        algorithm = algorithm_ctor(env)
        driver = OffPolicyDriver(
            env,
            algorithm,
            debug_summaries=True,
            summarize_grads_and_vars=True)
        replay_buffer = driver.add_replay_buffer()
        eval_driver = OffPolicyDriver(
            eval_env, algorithm, greedy_predict=True)
        driver.run = tf.function(driver.run)
        eval_driver.run = tf.function(eval_driver.run)

        env.reset()
        eval_env.reset()
        time_step = driver.get_initial_time_step()
        policy_state = driver.get_initial_state()
        for i in range(5):
            time_step, policy_state = driver.run(
                max_num_steps=batch_size * steps_per_episode,
                time_step=time_step,
                policy_state=policy_state)

        for i in range(300):
            time_step, policy_state = driver.run(
                max_num_steps=batch_size * 4,
                time_step=time_step,
                policy_state=policy_state)
            experience, _ = replay_buffer.get_next(sample_batch_size=128, num_steps=2)
            driver.train(experience)
            eval_env.reset()
            eval_time_step, _ = eval_driver.run(
                max_num_steps=(steps_per_episode - 1) * batch_size)
            logging.info("%d reward=%f", i,
                         float(tf.reduce_mean(eval_time_step.reward)))

        eval_env.reset()
        eval_time_step, _ = eval_driver.run(
            max_num_steps=(steps_per_episode - 1) * batch_size)
        logging.info("reward=%f", float(tf.reduce_mean(eval_time_step.reward)))
        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(eval_time_step.reward)), delta=2e-1)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    unittest.main()
