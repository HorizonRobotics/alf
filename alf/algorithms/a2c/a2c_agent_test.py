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

"""Tests for alf.algorithms.a2c.a2c_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest
from absl.testing import parameterized
import tensorflow_probability as tfp
from tf_agents.environments import time_step as ts
from tf_agents.networks import network, value_network, actor_distribution_network
from tf_agents.specs import tensor_spec
from tf_agents.networks import utils as network_utils
from tf_agents.utils import common
from tf_agents.environments import trajectory

from alf.algorithms.a2c import a2c_agent


class DummyActorNet(network.Network):

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 unbounded_actions=False,
                 stateful=False):
        state_spec = (tf.TensorSpec(input_tensor_spec.shape, tf.float32)
                      if stateful else ())
        super(DummyActorNet, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=state_spec,
            name='DummyActorNet')
        single_action_spec = tf.nest.flatten(output_tensor_spec)[0]
        activation_fn = None if unbounded_actions else tf.nn.tanh
        self._output_tensor_spec = output_tensor_spec
        self._layers = [
            tf.keras.layers.Dense(
                single_action_spec.shape.num_elements() * 2,
                activation=activation_fn,
                kernel_initializer=tf.compat.v1.initializers.constant(
                    [[2, 1], [1, 1]]),
                bias_initializer=tf.compat.v1.initializers.constant(5),
            ), ]

    def call(self, observations, step_type=None, network_state=()):
        states = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
        for layer in self.layers:
            states = layer(states)

        single_action_spec = tf.nest.flatten(self._output_tensor_spec)[0]
        actions, stdevs = tf.split(states, 2, axis=1)
        actions = tf.reshape(actions, [-1] + single_action_spec.shape.as_list())
        stdevs = tf.reshape(stdevs, [-1] + single_action_spec.shape.as_list())
        actions = tf.nest.pack_sequence_as(self._output_tensor_spec, [actions])
        stdevs = tf.nest.pack_sequence_as(self._output_tensor_spec, [stdevs])

        distribution = nest.map_structure_up_to(
            self._output_tensor_spec, tfp.distributions.Normal, actions, stdevs)
        return distribution, network_state


class DummyValueNet(network.Network):

    def __init__(self, observation_spec, name=None, outer_rank=1):
        super(DummyValueNet, self).__init__(observation_spec, (), 'DummyValueNet')
        self._outer_rank = outer_rank
        self._layers.append(
            tf.keras.layers.Dense(
                1,
                kernel_initializer=tf.compat.v1.initializers.constant([2, 1]),
                bias_initializer=tf.compat.v1.initializers.constant([5])))

    def call(self, inputs, unused_step_type=None, network_state=()):
        hidden_state = tf.cast(tf.nest.flatten(inputs), tf.float32)[0]
        batch_squash = network_utils.BatchSquash(self._outer_rank)
        hidden_state = batch_squash.flatten(hidden_state)
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        value_pred = tf.squeeze(batch_squash.unflatten(hidden_state), axis=-1)
        return value_pred, network_state


class A2cAgentTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super(A2cAgentTest, self).setUp()
        tf.compat.v1.enable_resource_variables()
        self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
        self._time_step_spec = ts.time_step_spec(self._obs_spec)
        self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)

    def test_create_agent(self):
        a2c_agent.A2CAgent(
            self._time_step_spec,
            self._action_spec,
            actor_net=DummyActorNet(
                self._obs_spec, self._action_spec, unbounded_actions=False),
            value_net=DummyValueNet(self._obs_spec),
            optimizer=None)

    def test_policy(self):
        agent = a2c_agent.A2CAgent(
            self._time_step_spec,
            self._action_spec,
            actor_net=DummyActorNet(
                self._obs_spec, self._action_spec, unbounded_actions=False),
            value_net=DummyValueNet(self._obs_spec),
            optimizer=None)
        observations = tf.constant([[1, 2]], dtype=tf.float32)
        time_steps = ts.restart(observations, batch_size=2)
        actions = agent.policy.action(time_steps).action
        self.assertEqual(actions.shape.as_list(), [1, 1])
        self.evaluate(tf.compat.v1.global_variables_initializer())
        action_values = self.evaluate(actions)
        tf.nest.map_structure(
            lambda v, s: self.assertAllInRange(v, s.minimum, s.maximum),
            action_values, self._action_spec)

    def test_train(self):
        with tf.compat.v2.summary.record_if(False):
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                self._obs_spec,
                self._action_spec)
            value_net = value_network.ValueNetwork(
                self._obs_spec)
            counter = common.create_variable('test_train_counter')
            agent = a2c_agent.A2CAgent(
                self._time_step_spec,
                self._action_spec,
                actor_net=actor_net,
                value_net=value_net,
                optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
                train_step_counter=counter)

            batch_size = 5
            observations = tf.constant(
                [[[1, 2], [3, 4], [5, 6]]] * batch_size, dtype=tf.float32)
            time_steps = ts.TimeStep(
                step_type=tf.constant([[1] * 3] * batch_size, dtype=tf.int32),
                reward=tf.constant([[1] * 3] * batch_size, dtype=tf.float32),
                discount=tf.constant([[1] * 3] * batch_size, dtype=tf.float32),
                observation=observations)
            actions = tf.constant([[[0], [1], [1]]] * batch_size, dtype=tf.float32)

            experience = trajectory.Trajectory(
                time_steps.step_type, observations, actions, (),
                time_steps.step_type, time_steps.reward, time_steps.discount)

            agent.policy.variables()
            if tf.executing_eagerly():
                loss = lambda: agent.train(experience)
            else:
                loss = agent.train(experience)
            self.evaluate(tf.compat.v1.initialize_all_variables())
            self.assertEqual(self.evaluate(counter), 0)
            self.evaluate(loss)
            self.assertEqual(self.evaluate(counter), 1)


if __name__ == '__main__':
    tf.test.main()
