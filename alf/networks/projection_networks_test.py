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
"""Tests for alf.networks.projection_networks."""

from functools import partial
from absl.testing import parameterized

import torch

import alf
from alf.networks import BetaProjectionNetwork
from alf.networks import CategoricalProjectionNetwork
from alf.networks import NormalProjectionNetwork
from alf.networks import OnehotCategoricalProjectionNetwork
from alf.networks import StableNormalProjectionNetwork
from alf.networks import TruncatedProjectionNetwork
from alf.networks.projection_networks import MixtureProjectionNetwork
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import dist_utils
from alf.utils.dist_utils import DistributionSpec
import alf.utils.math_ops as math_ops


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

    def test_make_parallel(self):
        replicas = 4
        outer_dim = 3
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(outer_dim, ))

        net = CategoricalProjectionNetwork(
            input_size=input_spec.shape[0],
            action_spec=BoundedTensorSpec((1, ), minimum=0, maximum=4),
            logits_init_output_factor=0)
        parallel_net = net.make_parallel(replicas)
        dist, _ = parallel_net(embedding)
        self.assertTrue(isinstance(parallel_net.output_spec, DistributionSpec))
        self.assertEqual(dist.batch_shape, (outer_dim, replicas))
        self.assertEqual(dist.base_dist.batch_shape, (outer_dim, replicas, 1))
        self.assertTrue(torch.all(dist.base_dist.probs == 0.2))


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
            std_transform=math_ops.identity)

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
            torch.all(dist.mean > torch.tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(dist.mean < torch.tensor(action_spec.maximum)))

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
            torch.all(dist.base_dist.mean > torch.tensor(action_spec.minimum)))
        self.assertFalse(
            torch.all(dist.base_dist.mean < torch.tensor(action_spec.maximum)))

        out = dist.sample((1, ))
        self.assertTrue(out.std() > 0)
        self.assertTrue(torch.all(out >= torch.tensor(action_spec.minimum)))
        self.assertTrue(torch.all(out <= torch.tensor(action_spec.maximum)))

    @parameterized.parameters((NormalProjectionNetwork, True),
                              (NormalProjectionNetwork, False),
                              (StableNormalProjectionNetwork, True),
                              (StableNormalProjectionNetwork, False))
    def test_parallel_normal_projection_network(self, network_ctor,
                                                state_dependent_std):
        """Test normal projection net with specified parallelism of replicas.

        """
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(100, ))

        action_spec = BoundedTensorSpec((2, ),
                                        torch.float32,
                                        minimum=(0, -0.01),
                                        maximum=(0.01, 0))
        net = network_ctor(
            input_spec.shape[0],
            action_spec,
            state_dependent_std=state_dependent_std,
            projection_output_init_gain=1.0).make_parallel(5)
        dist, _ = net(embedding)

        self.assertEqual((100, 5), dist.batch_shape)
        self.assertEqual((2, ), dist.event_shape)
        self.assertTrue(dist.mean.std() > 0)
        self.assertTrue(
            torch.all(dist.mean > torch.tensor(action_spec.minimum)))
        self.assertTrue(
            torch.all(dist.mean < torch.tensor(action_spec.maximum)))

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

    def test_beta_projection_net(self):
        """Test max and min stds for BetaProjectionNetwork."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = torch.rand((100, ) + input_spec.shape, dtype=torch.float32)
        action_spec = BoundedTensorSpec((8, ),
                                        minimum=-1.,
                                        maximum=1.,
                                        dtype=torch.float32)

        net = BetaProjectionNetwork(
            input_spec.shape[0], action_spec, projection_output_init_gain=.0)

        dist, _ = net(embedding)
        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
        samples = dist.sample()
        self.assertTrue(torch.all(samples >= -1))
        self.assertTrue(torch.all(samples <= 1))
        self.assertTrue(torch.any(samples <= 0))
        self.assertTrue(torch.any(samples >= 0))

    def test_parallel_beta_projection_net(self):
        """Test the parallel version of BetaProjectionNetwork."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = torch.rand((100, ) + input_spec.shape, dtype=torch.float32)
        action_spec = BoundedTensorSpec((2, ),
                                        minimum=-1.,
                                        maximum=1.,
                                        dtype=torch.float32)

        net = BetaProjectionNetwork(
            input_spec.shape[0], action_spec,
            projection_output_init_gain=.0).make_parallel(5)

        dist, _ = net(embedding)
        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
        self.assertEqual((100, 5), dist.batch_shape)
        self.assertEqual((2, ), dist.event_shape)

    def test_truncated_projection_net(self):
        """Test max and min stds for BetaProjectionNetwork."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = torch.rand((100, ) + input_spec.shape, dtype=torch.float32)
        action_spec = BoundedTensorSpec((8, ),
                                        minimum=-1.,
                                        maximum=1.,
                                        dtype=torch.float32)

        net = TruncatedProjectionNetwork(
            input_spec.shape[0], action_spec, projection_output_init_gain=.0)

        dist, _ = net(embedding)
        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
        samples = dist.sample()
        self.assertTrue(torch.all(samples >= -1))
        self.assertTrue(torch.all(samples <= 1))
        self.assertTrue(torch.any(samples <= 0))
        self.assertTrue(torch.any(samples >= 0))


class TestOnehotCategoricalProjectionNetwork(parameterized.TestCase,
                                             alf.test.TestCase):
    @parameterized.parameters('st', 'st-gumbel', 'plain', 'gumbel')
    def test_onehot_categorical_uniform_projection_net(self, mode):
        """A zero-weight net generates uniform actions."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(1, ))

        net = OnehotCategoricalProjectionNetwork(
            input_size=input_spec.shape[0],
            mode=mode,
            action_spec=BoundedTensorSpec((1, ), minimum=0, maximum=4),
            logits_init_output_factor=0)
        dist, _ = net(embedding)
        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
        self.assertEqual(dist.batch_shape, (1, ))
        self.assertEqual(dist.base_dist.batch_shape, (1, 1))
        self.assertTrue(torch.all(dist.base_dist.probs == 0.2))

    @parameterized.parameters('st', 'st-gumbel', 'plain', 'gumbel')
    def test_onehot_samples(self, mode):
        """Samples from the projection net are onehot vectors."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(1, ))

        net = OnehotCategoricalProjectionNetwork(
            input_size=input_spec.shape[0],
            mode=mode,
            action_spec=BoundedTensorSpec((1, ), minimum=0, maximum=4),
            logits_init_output_factor=0.1)
        dist, _ = net(embedding)
        samples = dist.sample()
        self.assertTrue(torch.all(samples.sum(dim=-1) == 1))

    @parameterized.parameters('plain', 'st')
    def test_straight_through_gradient(self, mode):
        """Test the gradient with straight through estimator."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = input_spec.ones(outer_dims=(1, ))

        net = OnehotCategoricalProjectionNetwork(
            input_size=input_spec.shape[0],
            action_spec=BoundedTensorSpec((1, ), minimum=0, maximum=4),
            logits_init_output_factor=0.1,
            mode=mode)
        dist, _ = net(embedding)

        if mode == 'plain':
            self.assertTrue(dist.has_rsample == False)
        else:
            self.assertTrue(dist.has_rsample == True)
            _, log_prob = dist_utils.rsample_action_distribution(
                dist, return_log_prob=True)

            loss = log_prob.sum()
            loss.backward()

            p_layer = net._projection_layer
            self.assertTrue(p_layer.weight.grad.shape == p_layer.weight.shape)
            self.assertTrue(p_layer.bias.grad.shape == p_layer.bias.shape)

            self.assertTensorNotClose(p_layer.weight.grad,
                                      torch.zeros(p_layer.weight.grad.shape))
            self.assertTensorNotClose(p_layer.bias.grad,
                                      torch.zeros(p_layer.bias.grad.shape))

    @parameterized.parameters('st', 'st-gumbel', 'plain', 'gumbel')
    def test_mode(self, mode):
        """Test the mode of the onehot caregorical distribution."""
        input_spec = TensorSpec((10, ), torch.float32)
        embedding = input_spec.randn(outer_dims=(5, ))

        net1 = OnehotCategoricalProjectionNetwork(
            input_size=input_spec.shape[0],
            mode=mode,
            action_spec=BoundedTensorSpec((1, ), minimum=0, maximum=4),
            logits_init_output_factor=0.1)
        dist1, _ = net1(embedding)
        onehot_mode1 = dist_utils.get_mode(dist1)

        # create a categorical projection network for comparison
        net2 = CategoricalProjectionNetwork(
            input_size=input_spec.shape[0],
            action_spec=BoundedTensorSpec((1, ), minimum=0, maximum=4),
            logits_init_output_factor=0)

        # copy parameters from net1 -> net2
        for ws, wt in zip(net1.parameters(), net2.parameters()):
            wt.data.copy_(ws)

        dist2, _ = net2(embedding)
        mode2 = dist_utils.get_mode(dist2)
        onehot_mode2 = torch.nn.functional.one_hot(
            mode2, num_classes=dist2.base_dist.logits.shape[-1])

        self.assertTensorClose(onehot_mode1, onehot_mode2)


class TestMixtureProjectionNetwork(parameterized.TestCase, alf.test.TestCase):
    def test_mixture_of_gaussian_1d_action(self):
        input_spec = TensorSpec((10, ), torch.float32)

        net = MixtureProjectionNetwork(
            input_size=input_spec.shape[0],
            action_spec=BoundedTensorSpec((1, ), minimum=0.0, maximum=4.0),
            num_components=3,
            component_ctor=partial(
                NormalProjectionNetwork,
                projection_output_init_gain=0,
                std_bias_initializer_value=0,
                squash_mean=False,
                state_dependent_std=True,
                std_transform=math_ops.identity))

        self.assertEqual(3, net.num_components)

        embedding = input_spec.ones(outer_dims=(7, ))
        dist, _ = net(embedding)
        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
        self.assertEqual(dist.batch_shape, (7, ))
        x = dist.sample()
        self.assertEqual((7, 1), x.shape)
        mode = dist_utils.get_mode(dist)
        self.assertEqual((7, 1), mode.shape)
        mode = dist_utils.get_rmode(dist)
        self.assertEqual((7, 1), mode.shape)

    def test_mixture_of_stable_normal_2d_action(self):
        input_spec = TensorSpec((10, ), torch.float32)

        net = MixtureProjectionNetwork(
            input_size=input_spec.shape[0],
            action_spec=BoundedTensorSpec((2, ), minimum=0.0, maximum=4.0),
            num_components=3,
            component_ctor=partial(
                StableNormalProjectionNetwork,
                state_dependent_std=True,
                squash_mean=False,
                scale_distribution=True,
                min_std=1e-3,
                max_std=10.0))

        self.assertEqual(3, net.num_components)

        embedding = input_spec.ones(outer_dims=(7, ))
        dist, _ = net(embedding)
        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
        self.assertEqual(dist.batch_shape, (7, ))
        x = dist.sample()
        self.assertEqual((7, 2), x.shape)
        mode = dist_utils.get_mode(dist)
        self.assertEqual((7, 2), mode.shape)
        mode = dist_utils.get_rmode(dist)
        self.assertEqual((7, 2), mode.shape)

    def test_mixture_of_beta_2d_action(self):
        input_spec = TensorSpec((10, ), torch.float32)

        net = MixtureProjectionNetwork(
            input_size=input_spec.shape[0],
            action_spec=BoundedTensorSpec((2, ), minimum=0.0, maximum=4.0),
            num_components=3,
            component_ctor=partial(
                BetaProjectionNetwork, min_concentration=1.0))

        self.assertEqual(3, net.num_components)

        embedding = input_spec.ones(outer_dims=(7, ))
        dist, _ = net(embedding)
        self.assertTrue(isinstance(net.output_spec, DistributionSpec))
        self.assertEqual(dist.batch_shape, (7, ))
        x = dist.sample()
        self.assertEqual((7, 2), x.shape)
        mode = dist_utils.get_mode(dist)
        self.assertEqual((7, 2), mode.shape)
        mode = dist_utils.get_rmode(dist)
        self.assertEqual((7, 2), mode.shape)


if __name__ == "__main__":
    alf.test.main()
