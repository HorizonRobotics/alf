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
"""Tests for alf.networks.projection_networks."""

from absl.testing import parameterized
import unittest

import torch

from alf.networks.projection_networks import CategoricalProjectionNetwork
from alf.networks.projection_networks import NormalProjectionNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec

import alf.layers as layers


class TestCategoricalProjectionNetwork(unittest.TestCase):
    def test_uniform_projection_net(self):
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(1, ))

        net = CategoricalProjectionNetwork(
            input_size=input_spec.shape[0],
            num_actions=5,
            logits_init_output_factor=0)
        dist = net(embedding)
        self.assertTrue(torch.all(dist.probs == 0.2))

    def test_zero_mean_projection_net(self):
        input_spec = TensorSpec((10, ), torch.float32)
        embeddings = input_spec.ones(outer_dims=(1000, ))

        net = CategoricalProjectionNetwork(
            input_size=input_spec.shape[0],
            num_actions=5,
            logits_init_output_factor=1.0)
        dists = net(embeddings)
        self.assertTrue(dists.probs.std() > 0)
        self.assertTrue(
            torch.isclose(dists.probs.mean(), torch.as_tensor(0.2)))

    def test_zero_normal_projection_net(self):
        input_spec = TensorSpec((10, ), torch.float32)
        action_spec = TensorSpec((8, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(2, ))

        net = NormalProjectionNetwork(
            input_size=input_spec.shape[0],
            action_spec=action_spec,
            projection_output_init_gain=0,
            std_bias_initializer_value=0,
            squash_mean=False,
            std_transform=layers.identity)

        out = net(embedding).sample((10, ))
        self.assertEqual(tuple(out.size()), (
            10,
            2,
        ) + action_spec.shape)
        self.assertTrue(torch.all(out == 0))

    def test_squash_mean_normal_projection_net(self):
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(100, ))

        action_spec = TensorSpec((8, ), torch.float32)
        # For squashing mean, we need a bounded action spec
        self.assertRaises(AssertionError, NormalProjectionNetwork, input_spec,
                          action_spec)

        action_spec = BoundedTensorSpec((2, ),
                                        torch.float32,
                                        minimum=(0, -0.01),
                                        maximum=(0.01, 0))
        net = NormalProjectionNetwork(
            input_spec.shape[0], action_spec, projection_output_init_gain=1.0)
        dist = net(embedding)
        self.assertTrue(dist.mean.std() > 0)
        self.assertTrue(
            torch.all(dist.mean > torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(dist.mean < torch.as_tensor(action_spec.maximum)))

    def test_scale_distribution_normal_projection_net(self):
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = torch.rand((100, ) + input_spec.shape, dtype=torch.float32)

        action_spec = TensorSpec((8, ), torch.float32)
        # For scaling distribution, we need a bounded action spec
        self.assertRaises(AssertionError, NormalProjectionNetwork, input_spec,
                          action_spec)

        action_spec = BoundedTensorSpec((2, ),
                                        torch.float32,
                                        minimum=(0, -0.01),
                                        maximum=(0.01, 0))
        net = NormalProjectionNetwork(
            input_spec.shape[0],
            action_spec,
            projection_output_init_gain=1.0,
            squash_mean=True,
            scale_distribution=True)

        dist = net(embedding)

        # if scale_distribution=True, then squash_mean is ignored
        self.assertFalse(
            torch.all(
                dist.base_dist.mean > torch.as_tensor(action_spec.minimum)))
        self.assertFalse(
            torch.all(
                dist.base_dist.mean < torch.as_tensor(action_spec.maximum)))

        out = dist.sample((1, ))
        self.assertTrue(out.std() > 0)
        self.assertTrue(torch.all(out > torch.as_tensor(action_spec.minimum)))
        self.assertTrue(torch.all(out < torch.as_tensor(action_spec.maximum)))


if __name__ == "__main__":
    unittest.main()
