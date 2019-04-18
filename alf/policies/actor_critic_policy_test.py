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

import unittest
import tensorflow as tf

from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.specs.tensor_spec import TensorSpec, BoundedTensorSpec
from tf_agents.trajectories.time_step import TimeStep, StepType

from alf.policies.actor_critic_policy import ActorCriticPolicy


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


class ActorCriticPolicyTest(unittest.TestCase):
    def test_actor_critic_policy(self):
        observation_spec = TensorSpec(shape=(1, ), dtype=tf.float32)
        action_spec = BoundedTensorSpec(
            shape=(1, ), dtype=tf.int64, minimum=0, maximum=1)
        time_step_spec = TimeStep(
            step_type=TensorSpec(shape=(), dtype=tf.int32),
            reward=TensorSpec(shape=(), dtype=tf.float32),
            discount=TensorSpec(shape=(), dtype=tf.float32),
            observation=TensorSpec(shape=(1, ), dtype=tf.float32))

        learning_rate = 1e-1
        global_step = tf.Variable(
            0, dtype=tf.int64, trainable=False, name="global_step")
        batch_size = 1
        train_interval = 1
        steps_per_episode = 13

        actor_net = ActorDistributionNetwork(
            observation_spec, action_spec, fc_layer_params=())
        value_net = ValueNetwork(observation_spec, fc_layer_params=())

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        policy = ActorCriticPolicy(
            actor_network=actor_net,
            value_network=value_net,
            optimizer=optimizer,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            train_interval=train_interval,
            gamma=1.,
            train_step_counter=global_step)

        policy_state = policy.get_initial_state(batch_size)
        for i in range(200):
            for s in range(steps_per_episode):
                step_type = StepType.MID
                discount = 1.0
                if s == 0:
                    step_type = StepType.FIRST
                elif s == steps_per_episode - 1:
                    step_type = StepType.LAST
                    discount = 0.0

                time_step = TimeStep(
                    step_type=tf.constant([step_type] * batch_size),
                    reward=tf.constant([1.] * batch_size),
                    discount=tf.constant([discount] * batch_size),
                    observation=tf.constant([[1.]] * batch_size))

                policy_step = policy.action(time_step, policy_state)
                policy_state = policy_step.state

            if (i + 1) % 10 == 0:
                print('value=%s' % float(
                    tf.reduce_mean(policy._training_info[-1].value)))

        # It is a surprising that although the expected discounted reward
        # should be steps_per_episode/2, using one step actor critic will
        # converge to a different value, which is steps_per_episode-1
        self.assertAlmostEqual(
            steps_per_episode - 1,
            float(tf.reduce_mean(policy._training_info[-1].value)),
            delta=0.1)


if __name__ == '__main__':
    tf.config.gpu.set_per_process_memory_growth(True)
    unittest.main()
