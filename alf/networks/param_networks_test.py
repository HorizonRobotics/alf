# Copyright (c) 2020 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
import numpy as np
import torch

import alf
from alf.networks import ParamConvNet, ParamNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops


class ParamNetworksTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((1, True), (3, True, False),
                              (3, False, True, True),
                              (3, False, True, True, True))
    def test_param_convnet(self,
                           batch_size=1,
                           same_padding=False,
                           use_bias=True,
                           use_ln=False,
                           flatten_output=False):
        replica = 2
        input_spec = TensorSpec((3, 32, 32), torch.float32)
        network = ParamConvNet(
            input_channels=input_spec.shape[0],
            input_size=input_spec.shape[1:],
            conv_layer_params=((16, (2, 2), 1, (1, 0)), (15, 2, (1, 2), 1, 2)),
            use_bias=use_bias,
            use_ln=use_ln,
            n_groups=replica,
            same_padding=same_padding,
            activation=torch.tanh,
            flatten_output=flatten_output)
        self.assertLen(network._conv_layers, 2)

        # test non-parallel forward
        if not use_ln:
            # when use_ln, n_groups (replica=2) has to be specified when
            # initializing ParamConvNet, so it is parallel by default.
            image = input_spec.zeros(outer_dims=(batch_size, ))
            output, _ = network(image)
            if same_padding:
                output_shape = (batch_size, 15, 15, 7)
            else:
                output_shape = (batch_size, 15, 17, 8)
            if flatten_output:
                output_shape = (batch_size, np.prod(output_shape[1:]))
            self.assertEqual(output_shape[1:], network.output_spec.shape)
            self.assertEqual(output_shape, tuple(output.size()))

        # test parallel forward
        image = input_spec.zeros(outer_dims=(batch_size, ))
        replica_image = input_spec.zeros(outer_dims=(batch_size, replica))
        params = torch.randn(replica, network.param_length)
        network.set_parameters(params)
        output, _ = network(image)
        replica_output, _ = network(replica_image)
        self.assertEqual(output.shape, replica_output.shape)

        if same_padding:
            output_shape = (batch_size, replica, 15, 15, 7)
        else:
            output_shape = (batch_size, replica, 15, 17, 8)
        if flatten_output:
            output_shape = (*output_shape[0:2], np.prod(output_shape[2:]))
        self.assertEqual(output_shape[1:], network.output_spec.shape)
        self.assertEqual(output_shape, tuple(output.size()))

    @parameterized.parameters((1, ), (3, ), (3, True))
    def test_param_network(self, batch_size=1, use_ln=False):
        input_spec = TensorSpec((3, 32, 32), torch.float32)
        conv_layer_params = ((16, (2, 2), 1, (1, 0)), (15, 2, (1, 2), 1, 2))
        fc_layer_params = (128, )
        last_layer_size = 10
        last_activation = math_ops.identity
        replica = 2
        network = ParamNetwork(
            input_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            use_conv_ln=use_ln,
            use_fc_bias=True,
            use_fc_ln=use_ln,
            n_groups=replica,
            last_layer_size=last_layer_size,
            last_use_bias=True,
            last_use_ln=use_ln,
            last_activation=last_activation)
        self.assertLen(network._fc_layers, 2)

        # test non-parallel forward
        if not use_ln:
            # when use_ln, n_groups (replica=2) has to be specified when
            # initializing ParamNetwork, so it is parallel by default.
            image = input_spec.zeros(outer_dims=(batch_size, ))
            output, _ = network(image)
            output_shape = (batch_size, last_layer_size)
            self.assertEqual(output_shape[1:], network.output_spec.shape)
            self.assertEqual(output_shape, tuple(output.size()))

        # test parallel forward
        image = input_spec.zeros(outer_dims=(batch_size, ))
        replica_image = input_spec.zeros(outer_dims=(batch_size, replica))
        params = torch.randn(replica, network.param_length)
        network.set_parameters(params)
        output, _ = network(image)
        replica_output, _ = network(replica_image)
        self.assertEqual(output.shape, replica_output.shape)

        output_shape = (batch_size, replica, last_layer_size)
        self.assertEqual(output_shape[1:],
                         (replica, ) + network.output_spec.shape)
        self.assertEqual(output_shape, tuple(output.size()))


if __name__ == "__main__":
    alf.test.main()
