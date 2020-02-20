# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
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
"""Tests for alf.encoding_networks."""

from absl.testing import parameterized
import numpy as np
import unittest

import torch
import torch.nn as nn

from alf.networks.encoding_networks import ImageEncodingNetwork
from alf.networks.encoding_networks import ImageDecodingNetwork
from alf.networks.encoding_networks import EncodingNetwork
from alf.tensor_specs import TensorSpec


class EncodingNetworkTest(parameterized.TestCase, unittest.TestCase):
    def test_empty_layers(self):
        input_spec = TensorSpec((3, ), torch.float32)
        network = EncodingNetwork(input_spec)
        self.assertEmpty(list(network.parameters()))
        self.assertEmpty(network._fc_layers)
        self.assertTrue(network._img_encoding_net is None)

    @parameterized.parameters((True, ), (False, ))
    def test_image_encoding_network(self, flatten_output):
        input_spec = TensorSpec((3, 32, 32), torch.float32)
        img = input_spec.zeros(outer_dims=(1, ))
        network = ImageEncodingNetwork(
            input_channels=input_spec.shape[0],
            input_size=input_spec.shape[1:],
            conv_layer_params=[(16, (2, 2), 1, (1, 0)), (15, 2, (1, 2), 1)],
            activation=torch.tanh,
            flatten_output=flatten_output)

        self.assertLen(list(network.parameters()), 4)  # two conv2d layers

        output_shape = network.output_shape()
        output = network(img)
        self.assertEqual(output_shape, tuple(output.size()[1:]))

    @parameterized.parameters((None, ), ((100, 100), ))
    def test_image_decoding_network(self, preprocessing_fc_layers):
        input_spec = TensorSpec((100, ), torch.float32)
        embedding = input_spec.zeros(outer_dims=(1, ))
        network = ImageDecodingNetwork(
            input_size=input_spec.shape[0],
            transconv_layer_params=[(16, (2, 2), 1, (1, 0)), (64, 3, (1, 2),
                                                              0)],
            start_decoding_size=(20, 31),
            start_decoding_channels=8,
            preprocess_fc_layer_params=preprocessing_fc_layers)

        num_layers = 3 if preprocessing_fc_layers is None else 5
        self.assertLen(list(network.parameters()), num_layers * 2)

        output_shape = network.output_shape()
        output = network(embedding)
        self.assertEqual(output_shape, tuple(output.size()[1:]))

    @parameterized.parameters((None, None), (200, None), (50, torch.relu))
    def test_encoding_network_nonimg(self, last_layer_size, last_activation):
        input_spec = TensorSpec((100, ), torch.float32)
        embedding = input_spec.zeros(outer_dims=(1, ))
        network = EncodingNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=[30, 40, 50],
            activation=torch.tanh,
            last_layer_size=last_layer_size,
            last_activation=last_activation)

        num_layers = 3 if last_layer_size is None else 4
        self.assertLen(list(network.parameters()), num_layers * 2)

        if last_activation is None:
            self.assertEqual(network._fc_layers[-1]._activation, torch.tanh)
        else:
            self.assertEqual(network._fc_layers[-1]._activation,
                             last_activation)

        output = network(embedding)
        if last_layer_size is None:
            self.assertEqual(output.size()[1], 50)
        else:
            self.assertEqual(output.size()[1], last_layer_size)

    def test_encoding_network_img(self):
        input_spec = TensorSpec((3, 80, 80), torch.float32)
        img = input_spec.zeros(outer_dims=(1, ))
        network = EncodingNetwork(
            input_tensor_spec=input_spec,
            conv_layer_params=[(16, (5, 3), 2, (1, 1)), (15, 3, (2, 2), 0)])

        self.assertLen(list(network.parameters()), 4)

        output = network(img)
        output_shape = network._img_encoding_net.output_shape()
        self.assertEqual(output.shape[-1], np.prod(output_shape))


if __name__ == '__main__':
    unittest.main()
