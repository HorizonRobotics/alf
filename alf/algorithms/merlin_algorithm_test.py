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

from tf_agents.networks.encoding_network import EncodingNetwork
from tf_agents.specs.tensor_spec import TensorSpec

from alf.algorithms.actor_critic_loss import ActorCriticLoss
from alf.algorithms.decoding_algorithm import DecodingAlgorithm
from alf.algorithms.merlin_algorithm import MerlinAlgorithm
from alf.environments.suite_unittest import RNNPolicyUnittestEnv
from alf.policies.training_policy import TrainingPolicy


class MerlinAlgorithmTest(unittest.TestCase):
    def _create_policy(self, env, train_interval=1, learning_rate=1e-1):
        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        time_step_spec = env.time_step_spec()

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

        policy = TrainingPolicy(
            algorithm=algorithm,
            time_step_spec=time_step_spec,
            train_interval=train_interval,
            train_step_counter=global_step,
            debug_summaries=True,
            summarize_grads_and_vars=True)

        return policy

    def test_merlin_algorithm(self):
        batch_size = 100
        steps_per_episode = 15
        gap = 10
        env = RNNPolicyUnittestEnv(
            batch_size, steps_per_episode, gap, obs_dim=3)

        policy = self._create_policy(env, train_interval=6, learning_rate=1e-3)
        policy_state = policy.get_initial_state(batch_size)
        time_step = env.reset()
        for i in range(1000):
            for _ in range(steps_per_episode):
                reward = time_step.reward
                policy_step = policy.action(time_step, policy_state)
                policy_state = policy_step.state
                time_step = env.step(policy_step.action)

            if (i + 1) % 10 == 0:
                print('%s reward=%s' % (i + 1, float(tf.reduce_mean(reward))))

        self.assertAlmostEqual(1.0, float(tf.reduce_mean(reward)), delta=1e-3)


def run_under_summary_context(func, summary_dir, record_cond, flush_millis):
    import os
    summary_dir = os.path.expanduser(summary_dir)
    summary_writer = tf.summary.create_file_writer(
        summary_dir, flush_millis=flush_millis)
    summary_writer.set_as_default()
    global_step = tf.Variable(
        0, dtype=tf.int64, trainable=False, name="global_step")
    tf.summary.experimental.set_step(global_step)
    with tf.summary.record_if(record_cond):
        func()


if __name__ == '__main__':
    from alf.utils.common import set_per_process_memory_growth
    set_per_process_memory_growth()

    run_under_summary_context(
        unittest.main,
        summary_dir="~/tmp/debug",
        record_cond=lambda: True,
        flush_millis=1000)
