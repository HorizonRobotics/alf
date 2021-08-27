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


class TestInputpreprocessor(parameterized.TestCase, alf.test.TestCase):
    input_spec = TensorSpec((10, ))

    def _make_preproc(self, shared):
        preproc = EmbeddingPreprocessor(
            input_tensor_spec=TestInputpreprocessor.input_spec,
            embedding_dim=10)
        if shared:
            return preproc.copy().singleton()
        else:
            return preproc

    @parameterized.parameters((False, False), (True, False), (False, True),
                              (True, True))
    def test_input_preprocessor(self, lstm, shared_preproc):
        preproc = self._make_preproc(shared_preproc)

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
                TestInputpreprocessor.input_spec,
                TestInputpreprocessor.input_spec
            ],
            input_preprocessors=[input_preprocessor, torch.relu],
            preprocessing_combiner=NestConcat(dim=0))

        # 2) test copied network has its own parameters, including
        # parameters from input preprocessors
        copied_net = net.copy()
        if not preproc._singleton_instance:
            _check_with_shared_param(net, copied_net)
        else:
            _check_with_shared_param(net, copied_net, input_preprocessor)

        # 3) test for each replica of the NaiveParallelNetwork has its own
        # parameters, including parameters from input preprocessors
        replicas = 2
        p_net = alf.networks.network.NaiveParallelNetwork(net, replicas)
        if not preproc._singleton_instance:
            _check_with_shared_param(p_net._networks[0], p_net._networks[1])
        else:
            _check_with_shared_param(p_net._networks[0], p_net._networks[1],
                                     input_preprocessor)

        # 4) test network forward
        batch_size = 6
        batch = TestInputpreprocessor.input_spec.zeros(
            outer_dims=(batch_size, ))

        if lstm:
            state = [(), (torch.zeros((batch_size, 1)), ) * 2, ()]
            p_state = [(), (torch.zeros((batch_size, replicas, 1)), ) * 2, ()]
        else:
            state = ()
            p_state = ()

        net([batch, batch], state)
        p_net([batch, batch], p_state)

    @parameterized.parameters(False, True)
    def test_input_preprocessor_state(self, shared_preproc):
        input_preprocessor = self._make_preproc(shared_preproc)
        batch_size = 6
        batch = TestInputpreprocessor.input_spec.zeros(
            outer_dims=(batch_size, ))

        input_preprocessor(batch)
        self.assertRaises(
            AssertionError, input_preprocessor, inputs=batch, state=batch)


if __name__ == '__main__':
    alf.test.main()
