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
"""Tests for alf.encoding_networks."""

from absl.testing import parameterized
from absl import logging
import numpy as np
import functools
import time
import torch
import torch.nn as nn

import alf
from alf.networks.encoding_networks import ImageEncodingNetwork
from alf.networks.encoding_networks import ImageDecodingNetwork
from alf.networks.encoding_networks import EncodingNetwork
from alf.networks.encoding_networks import ParallelEncodingNetwork
from alf.networks.encoding_networks import ParallelImageEncodingNetwork
from alf.networks.encoding_networks import ParallelImageDecodingNetwork
from alf.networks import Network
from alf.networks.encoding_networks import LSTMEncodingNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, math_ops
from alf.nest.utils import NestSum, NestConcat


class EncodingNetworkTest(parameterized.TestCase, alf.test.TestCase):
    def test_empty_layers(self):
        input_spec = TensorSpec((3, ), torch.float32)
        network = EncodingNetwork(input_spec)
        self.assertEmpty(list(network.parameters()))
        self.assertEmpty(network._fc_layers)
        self.assertTrue(network._img_encoding_net is None)

    @parameterized.parameters((True, False), (False, True))
    def test_image_encoding_network(self, flatten_output, same_padding):
        input_spec = TensorSpec((3, 32, 32), torch.float32)
        img = input_spec.zeros(outer_dims=(1, ))
        network = ImageEncodingNetwork(
            input_channels=input_spec.shape[0],
            input_size=input_spec.shape[1:],
            conv_layer_params=((16, (2, 2), 1, (1, 0)), (15, 2, (1, 2), 1)),
            same_padding=same_padding,
            activation=torch.tanh,
            flatten_output=flatten_output)

        self.assertLen(list(network.parameters()), 4)  # two conv2d layers

        output, _ = network(img)
        if same_padding:
            output_shape = (15, 30, 15)
        else:
            output_shape = (15, 34, 16)
        if flatten_output:
            output_shape = (np.prod(output_shape), )
        self.assertEqual(output_shape, network.output_spec.shape)
        self.assertEqual(output_shape, tuple(output.size()[1:]))

    @parameterized.parameters((None, True), ((100, 100), False))
    def test_image_decoding_network(self, preprocessing_fc_layers,
                                    same_padding):
        input_spec = TensorSpec((100, ), torch.float32)
        embedding = input_spec.zeros(outer_dims=(1, ))
        network = ImageDecodingNetwork(
            input_size=input_spec.shape[0],
            transconv_layer_params=((16, (2, 2), 1, (1, 0)), (64, 3, (1, 2),
                                                              0)),
            start_decoding_size=(20, 31),
            start_decoding_channels=8,
            same_padding=same_padding,
            preprocess_fc_layer_params=preprocessing_fc_layers)

        num_layers = 3 if preprocessing_fc_layers is None else 5
        self.assertLen(list(network.parameters()), num_layers * 2)

        output, _ = network(embedding)
        if same_padding:
            output_shape = (64, 21, 63)
        else:
            output_shape = (64, 21, 65)
        self.assertEqual(output_shape, network.output_spec.shape)
        self.assertEqual(output_shape, tuple(output.size()[1:]))

    @parameterized.parameters(
        (None, None, None),
        (200, None, None),
        (50, torch.relu, None),
        (None, None, TensorSpec((5, 10), torch.float32)),
        (50, torch.relu, TensorSpec((5, 10), torch.float32)),
    )
    def test_encoding_network_nonimg(self, last_layer_size, last_activation,
                                     output_tensor_spec):
        input_spec = TensorSpec((100, ), torch.float32)
        embedding = input_spec.zeros(outer_dims=(1, ))

        if (last_layer_size is None and last_activation is not None) or (
                last_activation is None and last_layer_size is not None):
            with self.assertRaises(AssertionError):
                network = EncodingNetwork(
                    input_tensor_spec=input_spec,
                    output_tensor_spec=output_tensor_spec,
                    fc_layer_params=(30, 40, 50),
                    activation=torch.tanh,
                    last_layer_size=last_layer_size,
                    last_activation=last_activation)
        else:
            network = EncodingNetwork(
                input_tensor_spec=input_spec,
                output_tensor_spec=output_tensor_spec,
                fc_layer_params=(30, 40, 50),
                activation=torch.tanh,
                last_layer_size=last_layer_size,
                last_activation=last_activation)

            num_layers = 3 if last_layer_size is None else 4
            self.assertLen(list(network.parameters()), num_layers * 2)

            if last_activation is None:
                self.assertEqual(network._fc_layers[-1]._activation,
                                 torch.tanh)
            else:
                self.assertEqual(network._fc_layers[-1]._activation,
                                 last_activation)

            output, _ = network(embedding)

            if output_tensor_spec is None:
                if last_layer_size is None:
                    self.assertEqual(output.size()[1], 50)
                else:
                    self.assertEqual(output.size()[1], last_layer_size)
                self.assertEqual(network.output_spec.shape,
                                 tuple(output.size()[1:]))
            else:
                self.assertEqual(
                    tuple(output.size()[1:]), output_tensor_spec.shape)
                self.assertEqual(network.output_spec.shape,
                                 output_tensor_spec.shape)

    def test_encoding_network_img(self):
        input_spec = TensorSpec((3, 80, 80), torch.float32)
        img = input_spec.zeros(outer_dims=(1, ))
        network = EncodingNetwork(
            input_tensor_spec=input_spec,
            conv_layer_params=((16, (5, 3), 2, (1, 1)), (15, 3, (2, 2), 0)))

        self.assertLen(list(network.parameters()), 4)

        output, _ = network(img)
        output_spec = network._img_encoding_net.output_spec
        self.assertEqual(output.shape[-1], np.prod(output_spec.shape))

    def test_encoding_network_preprocessing_combiner(self):
        input_spec = dict(
            a=TensorSpec((3, 80, 80)),
            b=[TensorSpec((80, 80)), TensorSpec(())])
        imgs = common.zero_tensor_from_nested_spec(input_spec, batch_size=1)
        network = EncodingNetwork(
            input_tensor_spec=input_spec,
            preprocessing_combiner=NestSum(average=True),
            conv_layer_params=((1, 2, 2, 0), ))

        self.assertEqual(network._processed_input_tensor_spec,
                         TensorSpec((3, 80, 80)))

        output, _ = network(imgs)
        self.assertTensorEqual(output, torch.zeros((40 * 40, )))

    def test_encoding_network_input_preprocessor(self):
        input_spec = TensorSpec((1, ))
        inputs = common.zero_tensor_from_nested_spec(input_spec, batch_size=1)
        network = EncodingNetwork(
            input_tensor_spec=input_spec, input_preprocessors=torch.tanh)
        output, _ = network(inputs)
        self.assertEqual(output.size()[1], 1)

    @parameterized.parameters((True, ), (False, ))
    def test_encoding_network_nested_input(self, lstm):
        input_spec = dict(
            a=TensorSpec((3, 80, 80)),
            b=[
                TensorSpec((80, )),
                BoundedTensorSpec((), dtype="int64"),
                dict(x=TensorSpec((100, )), y=TensorSpec((200, )))
            ])
        imgs = common.zero_tensor_from_nested_spec(input_spec, batch_size=1)
        input_preprocessors = dict(
            a=EmbeddingPreprocessor(
                input_spec["a"],
                conv_layer_params=((1, 2, 2, 0), ),
                embedding_dim=100),
            b=[
                EmbeddingPreprocessor(input_spec["b"][0], embedding_dim=50),
                EmbeddingPreprocessor(input_spec["b"][1], embedding_dim=50),
                dict(x=None, y=torch.relu)
            ])

        if lstm:
            network_ctor = functools.partial(
                LSTMEncodingNetwork, hidden_size=(100, ))
        else:
            network_ctor = EncodingNetwork

        network = network_ctor(
            input_tensor_spec=input_spec,
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=NestConcat())
        output, _ = network(imgs, state=[(torch.zeros((1, 100)), ) * 2])

        if lstm:
            self.assertEqual(network.output_spec, TensorSpec((100, )))
            self.assertEqual(output.size()[-1], 100)
        else:
            self.assertEqual(len(list(network.parameters())), 4 + 2 + 1)
            self.assertEqual(network.output_spec, TensorSpec((500, )))
            self.assertEqual(output.size()[-1], 500)

    @parameterized.parameters(
        None,
        TensorSpec((), torch.float32),
        TensorSpec((1, ), torch.float32),
    )
    def test_make_parallel(self, output_spec):
        batch_size = 128
        input_spec = TensorSpec((1, 10, 10), torch.float32)

        conv_layer_params = ((2, 3, 2), (5, 3, 1))
        fc_layer_params = (256, 256)
        network = EncodingNetwork(
            input_tensor_spec=input_spec,
            output_tensor_spec=output_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=torch.relu_,
            last_layer_size=1,
            last_activation=math_ops.identity,
            name='base_encoding_network')
        replicas = 2
        num_layers = len(conv_layer_params) + len(fc_layer_params) + 1

        def _benchmark(pnet, name):
            t0 = time.time()
            outputs = []
            for _ in range(1000):
                embedding = input_spec.randn(outer_dims=(batch_size, ))
                output, _ = pnet(embedding)
                outputs.append(output)
            o = math_ops.add_n(outputs).sum()
            logging.info("%s time=%s %s" % (name, time.time() - t0, float(o)))

            if output_spec is None:
                self.assertEqual(output.shape, (batch_size, replicas, 1))
                self.assertEqual(pnet.output_spec.shape, (replicas, 1))
            else:
                self.assertEqual(output.shape,
                                 (batch_size, replicas, *output_spec.shape))
                self.assertEqual(pnet.output_spec.shape,
                                 (replicas, *output_spec.shape))

        pnet = network.make_parallel(replicas)
        self.assertTrue(isinstance(pnet, ParallelEncodingNetwork))
        self.assertEqual(len(list(pnet.parameters())), num_layers * 2)
        _benchmark(pnet, "ParallelENcodingNetwork")
        self.assertEqual(pnet.name, "parallel_" + network.name)

        pnet = alf.networks.network.NaiveParallelNetwork(network, replicas)
        _benchmark(pnet, "NaiveParallelNetwork")

        # test on default network name
        self.assertEqual(pnet.name, "naive_parallel_" + network.name)

        # test on user-defined network name
        pnet = alf.networks.network.NaiveParallelNetwork(
            network, replicas, name="pnet")
        self.assertEqual(pnet.name, "pnet")

    @parameterized.parameters((None, True), ((100, 100), False))
    def test_parallel_image_decoding_network(self, preprocessing_fc_layers,
                                             same_padding):
        input_spec = TensorSpec((100, ), torch.float32)

        replica = 2
        network = ParallelImageDecodingNetwork(
            input_size=input_spec.shape[0],
            n=replica,
            transconv_layer_params=((16, (2, 2), 1, (1, 0)), (64, 3, (1, 2),
                                                              0)),
            start_decoding_size=(20, 31),
            start_decoding_channels=8,
            same_padding=same_padding,
            preprocess_fc_layer_params=preprocessing_fc_layers)

        num_layers = 3 if preprocessing_fc_layers is None else 5
        self.assertLen(list(network.parameters()), num_layers * 2)

        batch_size = 3
        # 1) shared input case
        embedding = input_spec.zeros(outer_dims=(batch_size, ))
        output, _ = network(embedding)
        if same_padding:
            output_shape = (batch_size, replica, 64, 21, 63)
        else:
            output_shape = (batch_size, replica, 64, 21, 65)
        self.assertEqual(output_shape[1:], network.output_spec.shape)
        self.assertEqual(output_shape, tuple(output.size()))

        # 2) non-shared input case
        embedding = input_spec.zeros(outer_dims=(
            batch_size,
            replica,
        ))
        output, _ = network(embedding)
        if same_padding:
            output_shape = (batch_size, replica, 64, 21, 63)
        else:
            output_shape = (batch_size, replica, 64, 21, 65)
        self.assertEqual(output_shape[1:], network.output_spec.shape)
        self.assertEqual(output_shape, tuple(output.size()))

    @parameterized.parameters((True, True), (False, True), (True, False),
                              (False, False))
    def test_parallel_image_encoding_network(self, same_padding,
                                             flatten_output):
        input_spec = TensorSpec((3, 80, 80), torch.float32)

        replica = 2
        network = ParallelImageEncodingNetwork(
            input_channels=input_spec.shape[0],
            input_size=input_spec.shape[1:3],
            n=replica,
            conv_layer_params=((16, (5, 3), 2, (1, 1)), (15, 3, (2, 2), 0)),
            same_padding=same_padding,
            flatten_output=flatten_output)

        self.assertLen(list(network.parameters()), 4)

        batch_size = 3
        # 1) shared input case
        img = input_spec.zeros(outer_dims=(batch_size, ))
        output, _ = network(img)

        if same_padding:
            output_shape = (batch_size, replica, 15, 20, 20)
        else:
            output_shape = (batch_size, replica, 15, 19, 19)

        if flatten_output:
            self.assertEqual((*output_shape[1:2], np.prod(output_shape[2:])),
                             network.output_spec.shape)
            self.assertEqual((*output_shape[0:2], np.prod(output_shape[2:])),
                             tuple(output.size()))
        else:
            self.assertEqual(output_shape[1:], network.output_spec.shape)
            self.assertEqual(output_shape, tuple(output.size()))

        # 2) non-shared input case
        img = input_spec.zeros(outer_dims=(
            batch_size,
            replica,
        ))
        output, _ = network(img)

        if same_padding:
            output_shape = (batch_size, replica, 15, 20, 20)
        else:
            output_shape = (batch_size, replica, 15, 19, 19)

        if flatten_output:
            self.assertEqual((*output_shape[1:2], np.prod(output_shape[2:])),
                             network.output_spec.shape)
            self.assertEqual((*output_shape[0:2], np.prod(output_shape[2:])),
                             tuple(output.size()))
        else:
            self.assertEqual(output_shape[1:], network.output_spec.shape)
            self.assertEqual(output_shape, tuple(output.size()))

    @parameterized.parameters((1, ), (3, ))
    def test_parallel_network_output_size(self, replicas):
        batch_size = 128
        input_spec = TensorSpec((100, ), torch.float32)

        # a dummy encoding network which ouputs the input
        network = EncodingNetwork(input_tensor_spec=input_spec)

        pnet = network.make_parallel(replicas)
        nnet = alf.networks.network.NaiveParallelNetwork(network, replicas)

        def _check_output_size(embedding):
            p_output, _ = pnet(embedding)
            n_output, _ = nnet(embedding)
            self.assertTrue(p_output.shape == n_output.shape)
            self.assertTrue(p_output.shape[1:] == pnet._output_spec.shape)

        # the case with shared inputs
        embedding = input_spec.randn(outer_dims=(batch_size, ))
        _check_output_size(embedding)

        # the case with non-shared inputs
        embedding = input_spec.randn(outer_dims=(batch_size, replicas))
        _check_output_size(embedding)


class EncodingNetworkSideEffectsTest(alf.test.TestCase):
    def test_encoding_network_side_effects(self):
        input_spec = TensorSpec((100, ), torch.float32)

        fc_layer_params_list = [20, 10]
        self.assertRaises(
            AssertionError,
            EncodingNetwork,
            input_tensor_spec=input_spec,
            fc_layer_params=fc_layer_params_list,
        )

        fc_layer_params = (20, 10)
        enc_net = EncodingNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=fc_layer_params,
            last_layer_size=3,
            last_activation=torch.relu)

        self.assertTrue((fc_layer_params == (20, 10)))

        target_net = enc_net.copy()
        self.assertTrue(
            len(list(target_net.parameters())) == \
                len(list(enc_net.parameters())))


if __name__ == '__main__':
    alf.test.main()
