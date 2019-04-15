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
"""Test for alf.algorithms.icm.inverse_network."""

import tensorflow as tf
import numpy as np
from alf.algorithms.icm import inverse_network
from tf_agents.specs import tensor_spec
from tensorflow.python.framework import test_util
from absl.testing import parameterized


def _output_shape(output_spec):
    if not output_spec.dtype.is_integer:
        return output_spec.shape
    unique_num_actions = np.unique(
        output_spec.maximum - output_spec.minimum + 1)
    output_shape = output_spec.shape.concatenate([unique_num_actions])
    return output_shape


class InverseNetworkTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(False, True)
    @test_util.run_in_graph_and_eager_modes()
    def test_build(self, action_is_discrete):
        batch_size = 8
        feature_dim = 32
        actions_dim = (4,)
        action_type = tf.int32 if action_is_discrete else tf.float32
        feature_spec = tensor_spec.TensorSpec([feature_dim], tf.float32)
        action_spec = tensor_spec.BoundedTensorSpec(
            list(actions_dim), action_type, -2.0, 2.0)
        features = tf.random.uniform([batch_size, feature_dim])
        next_features = tf.random.uniform([batch_size, feature_dim])
        inverse_net = inverse_network.InverseNetwork(
            (feature_spec, feature_spec), action_spec, [128])
        actions, _ = inverse_net((features, next_features))
        self.assertAllEqual(
            actions.shape.as_list(),
            [batch_size] + _output_shape(action_spec).as_list())
        self.assertEqual(len(inverse_net.trainable_variables), 4)


if __name__ == '__main__':
    tf.test.main()
