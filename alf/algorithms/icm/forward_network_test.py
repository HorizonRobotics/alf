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
"""Test for alf.algorithms.icm.forward_network."""

import tensorflow as tf

from alf.algorithms.icm import forward_network
from tf_agents.specs import tensor_spec

from absl.testing import parameterized


class ForwardNetworkTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters([
        {'feature_dim': (1,), 'actions_dim': (), 'steps': None},
        {'feature_dim': (1,), 'actions_dim': (1,), 'steps': None},
        {'feature_dim': (1,), 'actions_dim': (2, 2), 'steps': None},
        {'feature_dim': (1,), 'actions_dim': (2, 2), 'steps': 2}])
    def test_forward_network(self, feature_dim, actions_dim, steps):
        batch_size = 1
        feature_spec = tensor_spec.TensorSpec(feature_dim, tf.float32)
        action_spec = tensor_spec.TensorSpec(actions_dim, tf.float32)
        action_fc_layer_params = (64,)
        joint_fc_layer_params = (128, 128)
        forward_net = forward_network.ForwardNetwork(
            (feature_spec, action_spec),
            feature_spec,
            action_fc_layer_params=action_fc_layer_params,
            joint_fc_layer_params=joint_fc_layer_params)

        # [BxT]
        outer_ranks = [batch_size, steps] if steps else [batch_size]
        features = tf.random.uniform(outer_ranks + list(feature_dim))

        actions = tf.random.uniform(outer_ranks + list(actions_dim))
        next_features_pred, _ = forward_net((features, actions))

        self.assertAllEqual(next_features_pred.shape.as_list(),
                            outer_ranks + list(feature_dim))
        self.assertEqual(len(forward_net.trainable_variables), 8)


if __name__ == '__main__':
    tf.test.main()
