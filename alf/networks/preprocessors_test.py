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
    input_spec = TensorSpec((10, ))
    preproc = EmbeddingPreprocessor(
        input_tensor_spec=input_spec, embedding_dim=10)

    shared_preproc = preproc.copy().singleton()

    @parameterized.parameters((False, preproc), (True, preproc),
                              (False, shared_preproc), (True, shared_preproc))
    def test_input_preprocessor(self, lstm, preproc):
        def _check_with_shared_param(net1, net2, shared_subnet=None):
            net1_params = set(net1.parameters())
            net2_params = set(net2.parameters())
            # check that net1 and net2 share paramsters with shared_subnet
            if shared_subnet is not None:
                shared_params = set(shared_subnet.parameters())
                for p in shared_params:
                    self.assertTrue((p in net1_params) and (p in net2_params))

            # for the rest part, net1 and net2 do not share parameters
            for p1, p2 in zip(net1_params, net2_params):
                if shared_subnet is None or p1 not in shared_params:
                    self.assertTrue(p1 is not p2)

        # 1) test input_preprocessor copy and each copy has its own parameters
        input_preprocessor = preproc
        input_preprocessor_copy = input_preprocessor.copy()

        if not preproc._singleton_instance:
            _check_with_shared_param(input_preprocessor,
                                     input_preprocessor_copy)
        elif preproc._singleton_instance:
            _check_with_shared_param(input_preprocessor,
                                     input_preprocessor_copy,
                                     input_preprocessor)

        if lstm:
            network_ctor = functools.partial(
                LSTMEncodingNetwork,
                hidden_size=(1, ),
                post_fc_layer_params=(2, 2))
        else:
            network_ctor = functools.partial(
                EncodingNetwork, fc_layer_params=(10, 10))

        net = network_ctor(
            input_tensor_spec=[
                TestInpurpreprocessor.input_spec,
                TestInpurpreprocessor.input_spec
            ],
            input_preprocessors=[input_preprocessor, torch.nn.ReLU],
            preprocessing_combiner=NestConcat(dim=1))

        # 2) test copied network has its own parameters, including
        # parameters from input preprocessors
        copied_net = net.copy()
        if not preproc._singleton_instance:
            _check_with_shared_param(net, copied_net)
        elif preproc._singleton_instance:
            _check_with_shared_param(net, copied_net, input_preprocessor)

        # 3) test for each replica of the NaiveParallelNetwork has its own
        # parameters, including parameters from input preprocessors
        replicas = 2
        p_net = alf.networks.network.NaiveParallelNetwork(net, replicas)
        if not preproc._singleton_instance:
            _check_with_shared_param(p_net._networks[0], p_net._networks[1])
        elif preproc._singleton_instance:
            _check_with_shared_param(p_net._networks[0], p_net._networks[1],
                                     input_preprocessor)


if __name__ == '__main__':
    alf.test.main()
