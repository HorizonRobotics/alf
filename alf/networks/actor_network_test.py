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

from absl.testing import parameterized
from absl import logging
import numpy as np
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import sequential_layer
from alf.networks import actor_network
from tensorflow.python.framework import test_util


class ActorNetworkTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_build(self, outer_dims):
        num_obs_dims = 5
        action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
        obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
        actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

        obs = tf.random.uniform(list(outer_dims) + [num_obs_dims])
        actions, _ = actor_net(obs)
        self.assertAllEqual(actions.shape.as_list(),
                            list(outer_dims) + action_spec.shape.as_list())

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_scalar_action(self, outer_dims):
        num_obs_dims = 5
        obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
        action_spec = tensor_spec.BoundedTensorSpec([], tf.float32, 2., 3.)

        actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

        obs = tf.random.uniform(list(outer_dims) + [num_obs_dims])
        actions, _ = actor_net(obs)
        self.assertAllEqual(actions.shape.as_list(),
                            list(outer_dims) + action_spec.shape.as_list())
        self.assertEqual(len(actor_net.trainable_variables), 2)

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_2d_action(self, outer_dims):
        num_obs_dims = 5
        obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
        action_spec = tensor_spec.BoundedTensorSpec([2, 3], tf.float32, 2., 3.)
        actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

        obs = tf.random.uniform(list(outer_dims) + [num_obs_dims])
        actions, _ = actor_net(obs)
        self.assertAllEqual(actions.shape.as_list(),
                            list(outer_dims) + action_spec.shape.as_list())
        self.assertEqual(len(actor_net.trainable_variables), 2)

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_actions_within_range(self, outer_dims):
        num_obs_dims = 5
        obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
        action_spec = tensor_spec.BoundedTensorSpec([2, 3], tf.float32, 2., 3.)
        actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

        obs = tf.random.uniform(list(outer_dims) + [num_obs_dims])
        actions, _ = actor_net(obs)
        actions_ = self.evaluate(actions)
        self.assertTrue(np.all(actions_ >= action_spec.minimum))
        self.assertTrue(np.all(actions_ <= action_spec.maximum))

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_list_of_single_action(self, outer_dims):
        num_obs_dims = 5
        obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
        action_spec = [tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)]

        actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

        obs = tf.random.uniform(list(outer_dims) + [num_obs_dims])
        actions, _ = actor_net(obs)

        self.assertAllEqual(actions[0].shape.as_list(),
                            list(outer_dims) + action_spec[0].shape.as_list())
        self.assertEqual(len(actor_net.trainable_variables), 2)

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_dict_of_single_action(self, outer_dims):
        num_obs_dims = 5
        obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
        action_spec = {
            'motor': tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
        }
        actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

        obs = tf.random.uniform(list(outer_dims) + [num_obs_dims])
        actions, _ = actor_net(obs)
        self.assertAllEqual(
            actions['motor'].shape.as_list(),
            list(outer_dims) + action_spec['motor'].shape.as_list())
        self.assertEqual(len(actor_net.trainable_variables), 2)

    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    @test_util.run_in_graph_and_eager_modes()
    def test_handle_preprocessing_layers(self, outer_dims):
        observation_spec = (tensor_spec.TensorSpec([1], tf.float32),
                            tensor_spec.TensorSpec([], tf.float32))
        time_step_spec = ts.time_step_spec(observation_spec)
        time_step = tensor_spec.sample_spec_nest(
            time_step_spec, outer_dims=outer_dims)

        action_spec = tensor_spec.BoundedTensorSpec((2, ), tf.float32, 2, 3)

        preprocessing_layers = (tf.keras.layers.Dense(4),
                                sequential_layer.SequentialLayer([
                                    tf.keras.layers.Reshape((1, )),
                                    tf.keras.layers.Dense(4)
                                ]))

        net = actor_network.ActorNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=tf.keras.layers.Add())

        action, _ = net(time_step.observation, time_step.step_type, ())
        self.assertEqual(list(outer_dims) + [2], action.shape.as_list())
        self.assertGreater(len(net.trainable_variables), 4)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
