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
import os
import psutil
import time

import unittest
import tensorflow as tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.encoding_network import EncodingNetwork
from tf_agents.specs.tensor_spec import TensorSpec

from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.decoding_algorithm import DecodingAlgorithm
from alf.algorithms.merlin_algorithm import MerlinAlgorithm
from alf.drivers.on_policy_driver import OnPolicyDriver
from alf.environments.suite_unittest import RNNPolicyUnittestEnv
from alf.policies.training_policy import TrainingPolicy
from alf.utils.common import run_under_record_context


class MerlinAlgorithmTest(unittest.TestCase):
    def _create_algorithm(self, env, learning_rate=1e-1):
        observation_spec = env.observation_spec()
        action_spec = env.action_spec()

        latent_dim = 3
        memory_size = 20

        global_step = tf.summary.experimental.get_step()

        encoder = EncodingNetwork(
            input_tensor_spec=observation_spec,
            fc_layer_params=(3, ),
            activation_fn=None,
            name="ObsEncoder")

        decoder = DecodingAlgorithm(
            decoder=EncodingNetwork(
                input_tensor_spec=TensorSpec((latent_dim, ), dtype=tf.float32),
                fc_layer_params=(3, ),
                activation_fn=None,
                name="ObsDecoder"),
            loss_weight=100.)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        algorithm = MerlinAlgorithm(
            observation_spec=observation_spec,
            action_spec=action_spec,
            encoders=encoder,
            decoders=decoder,
            latent_dim=latent_dim,
            lstm_size=(4, ),
            memory_size=memory_size,
            rl_loss=ActorCriticLoss(action_spec=action_spec, gamma=1.0),
            train_step_counter=global_step,
            optimizer=optimizer,
            debug_summaries=True)

        return algorithm

    def _create_policy(self, env, train_interval=1, learning_rate=1e-1):
        algorithm = self._create_algorithm(env, learning_rate)

        global_step = tf.summary.experimental.get_step()

        policy = TrainingPolicy(
            algorithm=algorithm,
            time_step_spec=env.time_step_spec(),
            train_interval=train_interval,
            train_step_counter=global_step,
            debug_summaries=True,
            summarize_grads_and_vars=False)

        return policy

    def test_merlin_algorithm(self):
        batch_size = 100
        steps_per_episode = 15
        gap = 10
        env = RNNPolicyUnittestEnv(
            batch_size, steps_per_episode, gap, obs_dim=3)

        proc = psutil.Process(os.getpid())

        policy = self._create_policy(env, train_interval=6, learning_rate=1e-3)
        policy_state = policy.get_initial_state(batch_size)
        time_step = env.reset()
        for i in range(100):
            t0 = time.time()
            for _ in range(10 * steps_per_episode):
                reward = time_step.reward
                policy_step = policy.action(time_step, policy_state)
                policy_state = policy_step.state
                time_step = env.step(policy_step.action)

            mem = proc.memory_info().rss // 1e6
            print('%s time=%.3f mem=%s reward=%.3f' %
                  (i, time.time() - t0, mem, float(tf.reduce_mean(reward))))

        self.assertAlmostEqual(1.0, float(tf.reduce_mean(reward)), delta=1e-1)

    def test_merlin_algorithm_on_policy_driver(self):
        batch_size = 100
        steps_per_episode = 15
        gap = 10
        env = RNNPolicyUnittestEnv(
            batch_size, steps_per_episode, gap, obs_dim=3)
        env = TFPyEnvironment(env)

        algorithm = self._create_algorithm(env, learning_rate=1e-3)
        driver = OnPolicyDriver(
            env,
            algorithm,
            train_interval=6,
            debug_summaries=True,
            summarize_grads_and_vars=False)

        eval_driver = OnPolicyDriver(env, algorithm, training=False)

        driver.run = tf.function(driver.run)

        proc = psutil.Process(os.getpid())

        policy_state = driver.get_initial_state()
        time_step = driver.get_initial_time_step()
        for i in range(100):
            t0 = time.time()
            time_step, policy_state = driver.run(
                max_num_steps=150 * batch_size,
                time_step=time_step,
                policy_state=policy_state)
            mem = proc.memory_info().rss // 1e6
            logging.info('%s time=%.3f mem=%s' % (i, time.time() - t0, mem))

        env.reset()
        time_step, _ = eval_driver.run(max_num_steps=14 * batch_size)
        logging.info("eval reward=%.3f" % tf.reduce_mean(time_step.reward))
        self.assertAlmostEqual(
            1.0, float(tf.reduce_mean(time_step.reward)), delta=1e-2)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()

    run_under_record_context(
        unittest.main,
        summary_dir="~/tmp/debug",
        summary_interval=1,
        flush_millis=1000)
