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
import tensorflow as tf
from alf.networks import critic_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts


class CriticRnnNetworkTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters({'outer_dims': (3, )}, {'outer_dims': (3, 5)})
    def test_build(self, outer_dims):
        observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32,
                                                         0, 1)
        time_step_spec = ts.time_step_spec(observation_spec)
        time_step = tensor_spec.sample_spec_nest(
            time_step_spec, outer_dims=outer_dims)

        action_spec = tensor_spec.BoundedTensorSpec((2, ), tf.float32, 2, 3)
        network_input_spec = (observation_spec, action_spec)
        action = tensor_spec.sample_spec_nest(
            action_spec, outer_dims=outer_dims)
        net = critic_rnn_network.CriticRnnNetwork(
            network_input_spec,
            observation_conv_layer_params=[(4, 2, 2)],
            observation_fc_layer_params=(4, ),
            action_fc_layer_params=(5, ),
            joint_fc_layer_params=(5, ),
            lstm_size=(3, ),
            output_fc_layer_params=(5, ),
        )

        network_input = (time_step.observation, action)
        q_values, network_state = net(network_input, time_step.step_type)

        self.assertEqual(list(outer_dims), q_values.shape.as_list())

        self.assertEqual(15, len(net.variables))
        # Obs Conv Net Kernel
        self.assertEqual((2, 2, 3, 4), net.variables[0].shape)
        # Obs Conv Net bias
        self.assertEqual((4, ), net.variables[1].shape)
        # Obs Fc Kernel
        self.assertEqual((64, 4), net.variables[2].shape)
        # Obs Fc Bias
        self.assertEqual((4, ), net.variables[3].shape)
        # Action Fc kernel
        self.assertEqual((2, 5), net.variables[4].shape)
        # Action Fc Bias
        self.assertEqual((5, ), net.variables[5].shape)
        # Joint Fc Kernel
        self.assertEqual((9, 5), net.variables[6].shape)
        # Joint Fc Bias
        self.assertEqual((5, ), net.variables[7].shape)
        # LSTM Cell Kernel
        self.assertEqual((5, 12), net.variables[8].shape)
        # LSTM Cell Recurrent Kernel
        self.assertEqual((3, 12), net.variables[9].shape)
        # LSTM Cell Bias
        self.assertEqual((12, ), net.variables[10].shape)
        # Output Fc Kernel
        self.assertEqual((3, 5), net.variables[11].shape)
        # Output Fc Bias
        self.assertEqual((5, ), net.variables[12].shape)
        # Q Value Kernel
        self.assertEqual((5, 1), net.variables[13].shape)
        # Q Value Bias
        self.assertEqual((1, ), net.variables[14].shape)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    from alf.utils.common import set_per_process_memory_growth

    set_per_process_memory_growth()
    tf.test.main()
