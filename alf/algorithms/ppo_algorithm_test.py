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
import gin.tf
import tensorflow as tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork

from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.drivers.sync_off_policy_driver import SyncOffPolicyDriver
from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import ActionType
from alf.networks.stable_normal_projection_network import StableNormalProjectionNetwork

DEBUGGING = True


def create_algorithm(env, use_rnn=False, learning_rate=1e-1):
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

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
            observation_spec,
            action_spec,
            fc_layer_params=(),
            continuous_projection_net=StableNormalProjectionNetwork)
        value_net = ValueNetwork(observation_spec, fc_layer_params=())

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    return PPOAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        actor_network=actor_net,
        value_network=value_net,
        loss=PPOLoss(
            action_spec=action_spec, gamma=1.0, debug_summaries=DEBUGGING),
        optimizer=optimizer,
        debug_summaries=DEBUGGING)


class PpoTest(tf.test.TestCase):
    def test_ppo(self):
        env_class = PolicyUnittestEnv
        learning_rate = 1e-1
        iterations = 20
        batch_size = 100
        steps_per_episode = 13
        env = env_class(batch_size, steps_per_episode)
        env = TFPyEnvironment(env)

        eval_env = env_class(batch_size, steps_per_episode)
        eval_env = TFPyEnvironment(eval_env)

        algorithm = create_algorithm(env, learning_rate=learning_rate)
        algorithm.set_summary_settings(summarize_grads_and_vars=DEBUGGING)
        driver = SyncOffPolicyDriver(env, algorithm)
        eval_driver = OnPolicyDriver(eval_env, algorithm, training=False)

        env.reset()
        eval_env.reset()
        time_step = driver.get_initial_time_step()
        policy_state = driver.get_initial_policy_state()
        for i in range(iterations):
            time_step, policy_state = driver.run(
                max_num_steps=batch_size * steps_per_episode,
                time_step=time_step,
                policy_state=policy_state)

            algorithm.train(num_updates=4, mini_batch_size=25)
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
            1.0, float(tf.reduce_mean(eval_time_step.reward)), delta=1e-1)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
