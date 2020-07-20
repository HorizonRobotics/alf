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

import collections
from absl.testing import parameterized

from absl import logging
import tensorflow as tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from alf.environments.suite_unittest import ValueUnittestEnv
from alf.environments.suite_unittest import PolicyUnittestEnv, RNNPolicyUnittestEnv
from alf.environments.suite_unittest import ActionType
from alf.algorithms.ddpg_algorithm import DdpgAlgorithm
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.drivers.threads import NestFIFOQueue
from alf.drivers.async_off_policy_driver import AsyncOffPolicyDriver
from alf.drivers.sync_off_policy_driver import SyncOffPolicyDriver
from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.utils.common import flatten_once
from alf.utils import common


def _create_sac_algorithm():
    observation_spec = common.get_observation_spec()
    action_spec = common.get_action_spec()
    actor_net = ActorDistributionNetwork(
        observation_spec, action_spec, fc_layer_params=(16, 16))
    critic_net = CriticNetwork((observation_spec, action_spec),
                               joint_fc_layer_params=(16, 16))
    return SacAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.optimizers.Adam(learning_rate=5e-3),
        critic_optimizer=tf.optimizers.Adam(learning_rate=5e-3),
        alpha_optimizer=tf.optimizers.Adam(learning_rate=1e-1),
        debug_summaries=True)


def _create_ddpg_algorithm():
    observation_spec = common.get_observation_spec()
    action_spec = common.get_action_spec()
    actor_net = ActorNetwork(
        observation_spec, action_spec, fc_layer_params=(16, 16))
    critic_net = CriticNetwork((observation_spec, action_spec),
                               joint_fc_layer_params=(16, 16))
    return DdpgAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.optimizers.Adam(learning_rate=5e-3),
        critic_optimizer=tf.optimizers.Adam(learning_rate=1e-1),
        debug_summaries=True)


def _create_ppo_algorithm():
    observation_spec = common.get_observation_spec()
    action_spec = common.get_action_spec()
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)

    actor_net = ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        input_fc_layer_params=(),
        output_fc_layer_params=None)
    value_net = ValueRnnNetwork(
        observation_spec,
        input_fc_layer_params=(),
        output_fc_layer_params=None)

    return PPOAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        actor_network=actor_net,
        value_network=value_net,
        loss_class=PPOLoss,
        optimizer=optimizer,
        debug_summaries=True)


def _create_ac_algorithm():
    observation_spec = common.get_observation_spec()
    action_spec = common.get_action_spec()
    optimizer = tf.optimizers.Adam(learning_rate=5e-5)
    actor_net = ActorDistributionNetwork(
        observation_spec, action_spec, fc_layer_params=(8, ))
    value_net = ValueNetwork(observation_spec, fc_layer_params=(8, ))

    return ActorCriticAlgorithm(
        observation_spec=observation_spec,
        action_spec=action_spec,
        actor_network=actor_net,
        value_network=value_net,
        loss_class=ActorCriticLoss,
        optimizer=optimizer,
        debug_summaries=True)


class ThreadQueueTest(parameterized.TestCase, tf.test.TestCase):
    def test_nest_fifo(self):
        NamedTuple = collections.namedtuple('tuple', 'x y')
        t0 = NamedTuple(x=tf.ones([2, 3]), y=tf.ones([2]))
        # test whether any field can be an empty tuple
        t = NamedTuple(x=t0, y=())

        queue = NestFIFOQueue(capacity=2, sample_element=t)

        # assert that we can safely enqueue/dequeue a tf.nest structure
        queue.enqueue(t)
        tf.nest.assert_same_structure(t, queue.dequeue())

        # assert that we can safely dequeue multiple tf.nest structures
        queue.enqueue(t)
        queue.enqueue(t)
        t2 = tf.nest.map_structure(lambda x, y: tf.stack([x, y], axis=0), t, t)
        tf.nest.assert_same_structure(queue.dequeue_many(2), t2)

    def test_nest_pack_and_unpack(self):
        NamedTuple = collections.namedtuple('tuple', 'x y')
        t0 = NamedTuple(x=tf.ones([2, 3]), y=tf.ones([2, 10]))
        t1 = NamedTuple(x=t0, y=tf.zeros([2, 3, 2]))
        nested = NamedTuple(x=t0, y=t1)
        nested_ = tf.nest.map_structure(flatten_once, nested)
        nested_ = tf.nest.map_structure(
            lambda e: tf.reshape(e, [2, -1] + list(e.shape[1:])), nested_)
        tf.nest.map_structure(
            lambda e1, e2: self.assertTrue(tf.reduce_all(tf.equal(e1, e2))),
            nested, nested_)


class AsyncOffPolicyDriverTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters((50, 20, 10, 5, 5, 10), (20, 10, 100, 10, 1, 20))
    def test_alf_metrics(self, num_envs, learn_queue_cap, unroll_length,
                         actor_queue_cap, num_actors, num_iterations):
        episode_length = 5
        env_f = lambda: TFPyEnvironment(
            ValueUnittestEnv(batch_size=1, episode_length=episode_length))

        envs = [env_f() for _ in range(num_envs)]
        common.set_global_env(envs[0])
        alg = _create_ac_algorithm()
        driver = AsyncOffPolicyDriver(envs, alg, num_actors, unroll_length,
                                      learn_queue_cap, actor_queue_cap)
        driver.start()
        total_num_steps_ = 0
        for _ in range(num_iterations):
            total_num_steps_ += driver.run_async()
        driver.stop()

        total_num_steps = int(driver.get_metrics()[1].result())
        self.assertGreaterEqual(total_num_steps_, total_num_steps)

        # An exp is only put in the log queue after it's put in the learning queue
        # So when we stop the driver (which will force all queues to stop),
        # some exps might be missing from the metric. Here we assert an arbitrary
        # lower bound of 2/5. The upper bound is due to the fact that StepType.LAST
        # is not recorded by the metric (episode_length==5).
        self.assertLessEqual(total_num_steps, int(total_num_steps_ * 4 // 5))
        self.assertGreaterEqual(total_num_steps,
                                int(total_num_steps_ * 2 // 5))

        average_reward = int(driver.get_metrics()[2].result())
        self.assertEqual(average_reward, episode_length - 1)

        episode_length = int(driver.get_metrics()[3].result())
        self.assertEqual(episode_length, episode_length)


class OffPolicyDriverTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters((_create_sac_algorithm, False, True),
                              (_create_ddpg_algorithm, False, True),
                              (_create_ppo_algorithm, True, True),
                              (_create_ppo_algorithm, True, False))
    def test_off_policy_algorithm(self, algorithm_ctor, use_rollout_state,
                                  sync_driver):
        logging.info("{} {}".format(algorithm_ctor.__name__, sync_driver))

        batch_size = 128
        if use_rollout_state:
            steps_per_episode = 5
            mini_batch_length = 8
            unroll_length = 8
            env_class = RNNPolicyUnittestEnv
        else:
            steps_per_episode = 12
            mini_batch_length = 2
            unroll_length = 12
            env_class = PolicyUnittestEnv
        env = TFPyEnvironment(
            env_class(
                batch_size,
                steps_per_episode,
                action_type=ActionType.Continuous))

        eval_env = TFPyEnvironment(
            env_class(
                batch_size,
                steps_per_episode,
                action_type=ActionType.Continuous))

        common.set_global_env(env)
        algorithm = algorithm_ctor()
        algorithm.set_summary_settings(summarize_grads_and_vars=True)
        algorithm.use_rollout_state = use_rollout_state

        if sync_driver:
            driver = SyncOffPolicyDriver(env, algorithm)
        else:
            driver = AsyncOffPolicyDriver([env],
                                          algorithm,
                                          num_actor_queues=1,
                                          unroll_length=unroll_length,
                                          learn_queue_cap=1,
                                          actor_queue_cap=1)
        eval_driver = OnPolicyDriver(eval_env, algorithm, training=False)

        eval_env.reset()
        driver.start()
        if sync_driver:
            time_step = driver.get_initial_time_step()
            policy_state = driver.get_initial_policy_state()
            for i in range(5):
                time_step, policy_state = driver.run(
                    max_num_steps=batch_size * steps_per_episode,
                    time_step=time_step,
                    policy_state=policy_state)

        for i in range(500):
            if sync_driver:
                time_step, policy_state = driver.run(
                    max_num_steps=batch_size * mini_batch_length * 2,
                    time_step=time_step,
                    policy_state=policy_state)
                whole_replay_buffer_training = False
                clear_replay_buffer = False
            else:
                driver.run_async()
                whole_replay_buffer_training = True
                clear_replay_buffer = True

            driver.algorithm.train(
                mini_batch_size=128,
                mini_batch_length=mini_batch_length,
                whole_replay_buffer_training=whole_replay_buffer_training,
                clear_replay_buffer=clear_replay_buffer)
            eval_env.reset()
            eval_time_step, _ = eval_driver.run(
                max_num_steps=(steps_per_episode - 1) * batch_size)
            logging.log_every_n_seconds(
                logging.INFO,
                "%d reward=%f" %
                (i, float(tf.reduce_mean(eval_time_step.reward))),
                n_seconds=1)
        driver.stop()

        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(eval_time_step.reward)), delta=2e-1)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
