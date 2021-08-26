# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Tests for alf.networks.disjoint_policy_value_network"""

from alf.algorithms.ppg import DisjointPolicyValueNetwork

from absl.testing import parameterized

import functools

import torch
import torch.distributions as td

import alf
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils.common import zero_tensor_from_nested_spec
from alf.networks import EncodingNetwork
from alf.nest.utils import NestConcat
from alf.utils.dist_utils import DistributionSpec


class TestDisjointPolicyValueNetwork(parameterized.TestCase,
                                     alf.test.TestCase):
    def setUp(self):
        self._batch_size = 3

        self._observation_spec = [
            TensorSpec((1, 20, 20), torch.float32),  # A greyscale image
            TensorSpec((3, 20, 20), torch.float32),  # Plus a color image
        ]
        self._conv_layer_params = ((8, 3, 1), (16, 3, 2, 1))
        self._fc_layer_params = (100, )
        self._action_spec = {
            'discrete': BoundedTensorSpec((), dtype='int32'),
            'continuous': BoundedTensorSpec((3, ))
        }

    # This test mainly test that the DisjointPolicyValueNetwork can be
    # correctly constructed and its forward() can generate outputs
    # that have the correct shape.
    #
    # It tests for the both the "dual" architecture and the "shared"
    # architecture of the network.
    @parameterized.parameters(False, True)
    def test_architecture(self, is_sharing_encoder):
        network = DisjointPolicyValueNetwork(
            observation_spec=self._observation_spec,
            action_spec=self._action_spec,
            encoding_network_ctor=functools.partial(
                EncodingNetwork,
                conv_layer_params=self._conv_layer_params,
                fc_layer_params=self._fc_layer_params,
                preprocessing_combiner=NestConcat(dim=0)),
            is_sharing_encoder=is_sharing_encoder)

        # Verify that the output specs are correct
        action_distribution_spec, aux_spec, value_spec = network.output_spec

        self.assertTrue(
            isinstance(action_distribution_spec["discrete"], DistributionSpec))
        self.assertTrue(
            isinstance(action_distribution_spec["continuous"],
                       DistributionSpec))
        self.assertEqual((), aux_spec.shape)
        self.assertEqual(torch.float32, aux_spec.dtype)
        self.assertEqual((), value_spec.shape)
        self.assertEqual(torch.float32, value_spec.dtype)

        # Verify that the outputs have the desired shape and type
        image = zero_tensor_from_nested_spec(self._observation_spec,
                                             self._batch_size)

        (action_distribution, value, aux), state = network(image, state=())

        self.assertTrue(
            isinstance(action_distribution['discrete'], td.Categorical))
        self.assertTrue(
            isinstance(action_distribution["continuous"].base_dist, td.Normal))

        self.assertEqual((self._batch_size, ), value.shape)
        self.assertEqual((self._batch_size, ), aux.shape)


if __name__ == "__main__":
    alf.test.main()
