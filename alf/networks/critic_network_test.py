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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from alf.networks import critic_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import sequential_layer
from tensorflow.python.framework import test_util


class CriticNetworkTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_build(self, outer_dims):
        num_obs_dims = 5
        num_actions_dims = 2
        obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
        action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)

        obs = tf.random.uniform(list(outer_dims) + [num_obs_dims])
        actions = tf.random.uniform(list(outer_dims) + [num_actions_dims])
        critic_net = critic_network.CriticNetwork((obs_spec, action_spec))

        q_values, _ = critic_net((obs, actions))
        self.assertAllEqual(q_values.shape.as_list(), list(outer_dims))
        self.assertLen(critic_net.trainable_variables, 2)

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_add_obs_conv_layers(self, outer_dims):
        num_obs_dims = 5
        num_actions_dims = 2

        obs_spec = tensor_spec.TensorSpec([3, 3, num_obs_dims], tf.float32)
        action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)
        critic_net = critic_network.CriticNetwork(
            (obs_spec, action_spec),
            observation_conv_layer_params=[(16, 3, 2)])

        obs = tf.random.uniform(list(outer_dims) + [3, 3, num_obs_dims])
        actions = tf.random.uniform(list(outer_dims) + [num_actions_dims])
        q_values, _ = critic_net((obs, actions))

        self.assertAllEqual(q_values.shape.as_list(), list(outer_dims))
        self.assertLen(critic_net.trainable_variables, 4)

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_add_obs_fc_layers(self, outer_dims):
        num_obs_dims = 5
        num_actions_dims = 2

        obs_spec = tensor_spec.TensorSpec([3, 3, num_obs_dims], tf.float32)
        action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)
        critic_net = critic_network.CriticNetwork(
            (obs_spec, action_spec), observation_fc_layer_params=[20, 10])

        obs = tf.random.uniform(list(outer_dims) + [3, 3, num_obs_dims])
        actions = tf.random.uniform(list(outer_dims) + [num_actions_dims])
        q_values, _ = critic_net((obs, actions))

        self.assertAllEqual(q_values.shape.as_list(), list(outer_dims))
        self.assertLen(critic_net.trainable_variables, 6)

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_add_action_fc_layers(self, outer_dims):
        num_obs_dims = 5
        num_actions_dims = 2

        obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
        action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)
        critic_net = critic_network.CriticNetwork((obs_spec, action_spec),
                                                  action_fc_layer_params=[20])

        obs = tf.random.uniform(list(outer_dims) + [num_obs_dims])
        actions = tf.random.uniform(list(outer_dims) + [num_actions_dims])
        q_values, _ = critic_net((obs, actions))
        self.assertAllEqual(q_values.shape.as_list(), list(outer_dims))
        self.assertLen(critic_net.trainable_variables, 4)

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_add_joint_fc_layers(self, outer_dims):
        num_obs_dims = 5
        num_actions_dims = 2

        obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
        action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)
        critic_net = critic_network.CriticNetwork((obs_spec, action_spec),
                                                  joint_fc_layer_params=[20])

        obs = tf.random.uniform(list(outer_dims) + [num_obs_dims])
        actions = tf.random.uniform(list(outer_dims) + [num_actions_dims])
        q_values, _ = critic_net((obs, actions))
        self.assertAllEqual(q_values.shape.as_list(), list(outer_dims))
        self.assertLen(critic_net.trainable_variables, 4)

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_handle_preprocessing_layers(self, outer_dims):
        num_actions_dims = 2

        observation_spec = (tensor_spec.TensorSpec([1], tf.float32),
                            tensor_spec.TensorSpec([], tf.float32))
        time_step_spec = ts.time_step_spec(observation_spec)
        time_step = tensor_spec.sample_spec_nest(
            time_step_spec, outer_dims=outer_dims)

        action_spec = tensor_spec.BoundedTensorSpec((2, ), tf.float32, 2, 3)

        actions = tf.random.uniform(list(outer_dims) + [num_actions_dims])

        preprocessing_layers = (tf.keras.layers.Dense(4),
                                sequential_layer.SequentialLayer([
                                    tf.keras.layers.Reshape((1, )),
                                    tf.keras.layers.Dense(4)
                                ]))

        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_preprocessing_layers=preprocessing_layers,
            observation_preprocessing_combiner=tf.keras.layers.Add())

        q_values, _ = critic_net((time_step.observation, actions))
        self.assertAllEqual(q_values.shape.as_list(), list(outer_dims))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
