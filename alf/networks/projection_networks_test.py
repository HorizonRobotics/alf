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

import torch

import alf
from alf.networks import CategoricalProjectionNetwork
from alf.networks import NormalProjectionNetwork
from alf.networks import StableNormalProjectionNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
import alf.layers as layers
from alf.utils.dist_utils import DistributionSpec


class TestCategoricalProjectionNetwork(parameterized.TestCase,
                                       alf.test.TestCase):
    def test_uniform_projection_net(self):
        """A zero-weight net generates uniform actions."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(1, ))

        net = CategoricalProjectionNetwork(
            input_size=input_spec.shape[0],
            action_spec=BoundedTensorSpec((1, ), minimum=0, maximum=4),
            logits_init_output_factor=0)
        dist, _ = net(embedding)
        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
        self.assertEqual(dist.batch_shape, (1, ))
        self.assertEqual(dist.base_dist.batch_shape, (1, 1))
        self.assertTrue(torch.all(dist.base_dist.probs == 0.2))

    def test_close_uniform_projection_net(self):
        """A random-weight net generates close-uniform actions on average."""
        input_spec = TensorSpec((10, ), torch.float32)
        embeddings = input_spec.ones(outer_dims=(100, ))

        net = CategoricalProjectionNetwork(
            input_size=input_spec.shape[0],
            action_spec=BoundedTensorSpec((3, 2), minimum=0, maximum=4),
            logits_init_output_factor=1.0)
        dists, _ = net(embeddings)
        self.assertEqual(dists.batch_shape, (100, ))
        self.assertEqual(dists.base_dist.batch_shape, (100, 3, 2))
        self.assertTrue(dists.base_dist.probs.std() > 0)
        self.assertTrue(
            torch.isclose(dists.base_dist.probs.mean(), torch.as_tensor(0.2)))


class TestNormalProjectionNetwork(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((True, ), (False, ))
    def test_zero_normal_projection_net(self, state_dependent_std):
        """A zero-weight net generates zero actions."""
        input_spec = TensorSpec((10, ), torch.float32)
        action_spec = TensorSpec((8, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(2, ))

        net = NormalProjectionNetwork(
            input_size=input_spec.shape[0],
            action_spec=action_spec,
            projection_output_init_gain=0,
            std_bias_initializer_value=0,
            squash_mean=False,
            state_dependent_std=state_dependent_std,
            std_transform=layers.identity)

        out = net(embedding)[0].sample((10, ))
        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
        self.assertEqual(tuple(out.size()), (
            10,
            2,
        ) + action_spec.shape)
        self.assertTrue(torch.all(out == 0))

    @parameterized.parameters((NormalProjectionNetwork, ),
                              (StableNormalProjectionNetwork, ))
    def test_squash_mean_normal_projection_net(self, network_ctor):
        """A net with `sqaush_mean=True` should generate means within the action
        spec."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(100, ))

        action_spec = TensorSpec((8, ), torch.float32)
        # For squashing mean, we need a bounded action spec
        self.assertRaises(AssertionError, network_ctor, input_spec,
                          action_spec)

        action_spec = BoundedTensorSpec((2, ),
                                        torch.float32,
                                        minimum=(0, -0.01),
                                        maximum=(0.01, 0))
        net = network_ctor(
            input_spec.shape[0], action_spec, projection_output_init_gain=1.0)
        dist, _ = net(embedding)
        self.assertTrue(dist.mean.std() > 0)
        self.assertTrue(
            torch.all(dist.mean > torch.as_tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(dist.mean < torch.as_tensor(action_spec.maximum)))

    @parameterized.parameters((NormalProjectionNetwork, ),
                              (StableNormalProjectionNetwork, ))
    def test_scale_distribution_normal_projection_net(self, network_ctor):
        """A net with `scale_distribution=True` should always sample actions
        within the action spec."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = 10 * torch.rand(
            (100, ) + input_spec.shape, dtype=torch.float32)

        action_spec = TensorSpec((8, ), torch.float32)
        # For scaling distribution, we need a bounded action spec
        self.assertRaises(AssertionError, network_ctor, input_spec,
                          action_spec)

        action_spec = BoundedTensorSpec((2, ),
                                        torch.float32,
                                        minimum=(0, -0.01),
                                        maximum=(0.01, 0))
        net = network_ctor(
            input_spec.shape[0],
            action_spec,
            projection_output_init_gain=1.0,
            squash_mean=True,
            scale_distribution=True)

        dist, _ = net(embedding)

        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
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

    def test_stable_normal_projection_net_minmax_std(self):
        """Test max and min stds for StableNormalProjectionNetwork."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = torch.rand((100, ) + input_spec.shape, dtype=torch.float32)
        action_spec = TensorSpec((8, ), torch.float32)

        min_std, max_std, init_std = 0.2, 0.4, 0.3
        net = StableNormalProjectionNetwork(
            input_spec.shape[0],
            action_spec,
            projection_output_init_gain=1.0,
            state_dependent_std=True,
            squash_mean=False,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std)

        dist, _ = net(embedding)
        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
        self.assertTrue(torch.all(dist.base_dist.scale > min_std))
        self.assertTrue(torch.all(dist.base_dist.scale < max_std))


if __name__ == "__main__":
    alf.test.main()
