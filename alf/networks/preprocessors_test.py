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

from absl.testing import parameterized
import functools

import torch
import torch.nn as nn

import alf
from alf.nest.utils import NestConcat
from alf.networks.encoding_networks import EncodingNetwork
from alf.networks.encoding_networks import LSTMEncodingNetwork
from alf.networks.encoding_networks import ParallelEncodingNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.tensor_specs import TensorSpec
from alf.utils import common


class TestInpurpreprocessor(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((False, ), (True, ))
    def test_input_preprocessor(self, lstm):
        input_spec = TensorSpec((10, ))

        input_preprocessor_ctor = functools.partial(
            EmbeddingPreprocessor,
            input_tensor_spec=input_spec,
            embedding_dim=10)

        def _check_no_shared_param(net1, net2):
            for p1, p2 in zip(net1.parameters(), net2.parameters()):
                self.assertTrue(p1 is not p2)

        # 1) test input_preprocessor copy and each copy has its own parameters
        input_preprocessor = input_preprocessor_ctor()
        input_preprocessor_copy = input_preprocessor.copy()
        _check_no_shared_param(input_preprocessor, input_preprocessor_copy)

        if lstm:
            network_ctor = functools.partial(
                LSTMEncodingNetwork,
                hidden_size=(1, ),
                post_fc_layer_params=(2, 2))
        else:
            network_ctor = functools.partial(
                EncodingNetwork, fc_layer_params=(10, 10))

        net = network_ctor(
            input_tensor_spec=[input_spec, input_spec],
            input_preprocessor_ctors=[input_preprocessor_ctor, torch.nn.ReLU],
            preprocessing_combiner=NestConcat(dim=1))

        # 2) test copied network has its own parameters, including
        # parameters from input preprocessors
        copied_net = net.copy()
        _check_no_shared_param(net, copied_net)

        # 3) test for each replica of the NaiveParallelNetwork has its own
        # parameters, including parameters from input preprocessors
        replicas = 2
        p_net = alf.networks.network.NaiveParallelNetwork(net, replicas)
        _check_no_shared_param(p_net._networks[0], p_net._networks[1])


if __name__ == '__main__':
    alf.test.main()
