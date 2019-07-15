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
from alf.environments.suite_unittest import PolicyUnittestEnv
from alf.environments.suite_unittest import ActionType
from alf.algorithms.ddpg_algorithm import create_ddpg_algorithm
from alf.algorithms.sac_algorithm import create_sac_algorithm
from alf.algorithms.actor_critic_algorithm import create_ac_algorithm
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.on_policy_algorithm import OffPolicyAdapter
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
    return PPOAlgorithm(
        create_ac_algorithm(
            env=env,
            actor_fc_layers=(16, 16),
            value_fc_layers=(16, 16),
            learning_rate=1e-3))


def _create_ac_algorithm(env):
    return OffPolicyAdapter(
        create_ac_algorithm(
            env=env, actor_fc_layers=(8, ), value_fc_layers=(8, )))


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
    @parameterized.parameters((50, 20, 10, 5, 5, 10))
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
        self.assertGreaterEqual(
            total_num_steps,  # multiply by 2/3 because 1/3 of steps are StepType.LAST
            total_num_steps_ * 2 // 3)
        average_reward = int(driver.get_metrics()[2].result())
        self.assertEqual(average_reward, episode_length - 1)
        episode_length = int(driver.get_metrics()[3].result())
        self.assertEqual(episode_length, episode_length)


class OffPolicyDriverTest(parameterized.TestCase, unittest.TestCase):
    @parameterized.parameters((_create_sac_algorithm, True),
                              (_create_ddpg_algorithm, True),
                              (_create_ppo_algorithm, False))
    def test_off_policy_algorithm(self, algorithm_ctor, sync_driver):
        logging.info("{} {}".format(algorithm_ctor.__name__, sync_driver))

        batch_size = 128
        steps_per_episode = 12
        env_f = lambda: TFPyEnvironment(
            PolicyUnittestEnv(
                batch_size,
                steps_per_episode,
                action_type=ActionType.Continuous))

        eval_env = TFPyEnvironment(
            PolicyUnittestEnv(
                batch_size,
                steps_per_episode,
                action_type=ActionType.Continuous))

        algorithm = algorithm_ctor(env_f())

        if sync_driver:
            driver = SyncOffPolicyDriver(
                env_f(),
                algorithm,
                debug_summaries=True,
                summarize_grads_and_vars=True)
        else:
            driver = AsyncOffPolicyDriver(
                env_f,
                algorithm,
                num_envs=1,
                num_actor_queues=1,
                unroll_length=steps_per_episode,
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

        for i in range(300):
            if sync_driver:
                time_step, policy_state = driver.run(
                    max_num_steps=batch_size * 4,
                    time_step=time_step,
                    policy_state=policy_state)
                experience, _ = replayer.replay(
                    sample_batch_size=128, mini_batch_length=2)
            else:
                driver.run_async()
                experience = replayer.replay_all()

            driver.train(experience, mini_batch_size=128, mini_batch_length=2)
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
