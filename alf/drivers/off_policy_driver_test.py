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

import unittest
import collections
from absl.testing import parameterized

from absl import logging
import tensorflow as tf
import tensorflow_probability as tfp
import threading

from tf_agents.environments.tf_py_environment import TFPyEnvironment

from alf.environments.suite_unittest import ValueUnittestEnv
from alf.environments.suite_unittest import PolicyUnittestEnv, RNNPolicyUnittestEnv
from alf.environments.suite_unittest import ActionType
from alf.algorithms.ddpg_algorithm import create_ddpg_algorithm
from alf.algorithms.sac_algorithm import create_sac_algorithm
from alf.algorithms.actor_critic_algorithm import create_ac_algorithm
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.drivers.threads import NestFIFOQueue
from alf.drivers.threads import ActorThread, EnvThread
from alf.drivers.async_off_policy_driver import AsyncOffPolicyDriver
from alf.drivers.sync_off_policy_driver import SyncOffPolicyDriver
from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.utils.common import flatten_once


def _create_sac_algorithm(env):
    return create_sac_algorithm(
        env=env,
        actor_fc_layers=(16, 16),
        critic_fc_layers=(16, 16),
        alpha_learning_rate=5e-3,
        actor_learning_rate=5e-3,
        critic_learning_rate=5e-3)


def _create_ddpg_algorithm(env):
    return create_ddpg_algorithm(
        env=env,
        actor_fc_layers=(16, 16),
        critic_fc_layers=(16, 16),
        actor_learning_rate=1e-2,
        critic_learning_rate=1e-1)


def _create_ppo_algorithm(env):
    return create_ac_algorithm(
        env=env,
        actor_fc_layers=(),
        value_fc_layers=(),
        use_rnns=True,
        learning_rate=1e-3,
        algorithm_class=PPOAlgorithm)


def _create_ac_algorithm(env):
    return create_ac_algorithm(
        env=env, actor_fc_layers=(8, ), value_fc_layers=(8, ))


class ThreadQueueTest(parameterized.TestCase, unittest.TestCase):
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


class AsyncOffPolicyDriverTest(parameterized.TestCase, unittest.TestCase):
    @parameterized.parameters((50, 20, 10, 5, 5, 10), (20, 10, 100, 10, 1, 20))
    def test_alf_metrics(self, num_envs, learn_queue_cap, unroll_length,
                         actor_queue_cap, num_actors, num_iterations):
        episode_length = 5
        env_f = lambda: TFPyEnvironment(
            ValueUnittestEnv(batch_size=1, episode_length=episode_length))
        alg = _create_ac_algorithm(env_f())
        driver = AsyncOffPolicyDriver(env_f, alg, num_envs, num_actors,
                                      unroll_length, learn_queue_cap,
                                      actor_queue_cap)
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


class OffPolicyDriverTest(parameterized.TestCase, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        tf.random.set_seed(0)

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
        env_f = lambda: TFPyEnvironment(
            env_class(
                batch_size,
                steps_per_episode,
                action_type=ActionType.Continuous))

        eval_env = TFPyEnvironment(
            env_class(
                batch_size,
                steps_per_episode,
                action_type=ActionType.Continuous))

        algorithm = algorithm_ctor(env_f())
        algorithm.use_rollout_state = use_rollout_state

        if sync_driver:
            driver = SyncOffPolicyDriver(
                env_f(),
                algorithm,
                use_rollout_state=use_rollout_state,
                debug_summaries=True,
                summarize_grads_and_vars=True)
        else:
            driver = AsyncOffPolicyDriver(
                env_f,
                algorithm,
                use_rollout_state=algorithm.use_rollout_state,
                num_envs=1,
                num_actor_queues=1,
                unroll_length=unroll_length,
                learn_queue_cap=1,
                actor_queue_cap=1,
                debug_summaries=True,
                summarize_grads_and_vars=True)
        replayer = driver.exp_replayer
        eval_driver = OnPolicyDriver(
            eval_env, algorithm, training=False, greedy_predict=True)

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
                experience, _ = replayer.replay(
                    sample_batch_size=128, mini_batch_length=mini_batch_length)
            else:
                driver.run_async()
                experience = replayer.replay_all()

            driver.train(
                experience,
                mini_batch_size=128,
                mini_batch_length=mini_batch_length)
            eval_env.reset()
            eval_time_step, _ = eval_driver.run(
                max_num_steps=(steps_per_episode - 1) * batch_size)
            logging.info("%d reward=%f", i,
                         float(tf.reduce_mean(eval_time_step.reward)))
        driver.stop()

        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(eval_time_step.reward)), delta=2e-1)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    unittest.main()
