# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
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

from absl import logging
from absl.testing import parameterized
from collections import namedtuple
import torch
import torch.distributions as td

import alf
import alf.utils.dist_utils as dist_utils

ActionDistribution = namedtuple('ActionDistribution', ['a', 'b'])


class EstimatedEntropyTest(parameterized.TestCase, alf.test.TestCase):
    def setUp(self):
        self.skipTest("estimate_entropy is not implemented yet")

    def assertArrayAlmostEqual(self, x, y, eps):
        self.assertLess(tf.reduce_max(tf.abs(x - y)), eps)

    @parameterized.parameters(False, True)
    def test_estimated_entropy(self, assume_reparametrization):
        logging.info("assume_reparametrization=%s" % assume_reparametrization)
        num_samples = 1000000
        seed_stream = tfp.util.SeedStream(
            seed=1, salt='test_estimated_entropy')
        batch_shape = (2, )
        loc = tf.random.normal(shape=batch_shape, seed=seed_stream())
        scale = tf.abs(tf.random.normal(shape=batch_shape, seed=seed_stream()))


class DistributionSpecTest(alf.test.TestCase):
    def test_normal(self):
        dist = td.Normal(
            loc=torch.tensor([1., 2.]), scale=torch.tensor([0.5, 0.25]))
        spec = dist_utils.DistributionSpec.from_distribution(dist)
        params1 = {
            'loc': torch.tensor([0.5, 1.5]),
            'scale': torch.tensor([2., 4.])
        }
        dist1 = spec.build_distribution(params1)
        self.assertEqual(type(dist1), td.Normal)
        self.assertEqual(dist1.mean, params1['loc'])
        self.assertEqual(dist1.stddev, params1['scale'])

    def test_categorical(self):
        dist = td.Categorical(logits=torch.tensor([1., 2.]))
        spec = dist_utils.DistributionSpec.from_distribution(dist)
        params1 = {'logits': torch.tensor([0.5, 1.5])}
        dist1 = spec.build_distribution(params1)
        self.assertEqual(type(dist1), td.Categorical)
        # Categorical distribution will substract logsumexp(logits) from logits.
        # So dist1.logits is not equal to the supplied logits
        d = dist1.logits - params1['logits']
        self.assertAlmostEqual(d[0], d[1])

    def test_diag_multivariate_normal(self):
        dist = dist_utils.DiagMultivariateNormal(
            torch.tensor([[1., 2.], [2., 2.]]),
            torch.tensor([[2., 3.], [1., 1.]]))
        spec = dist_utils.DistributionSpec.from_distribution(dist)

        params1 = {
            'loc': torch.tensor([[0.5, 1.5], [1.0, 1.0]]),
            'scale': torch.tensor([[2., 4.], [2., 1.]])
        }
        dist1 = spec.build_distribution(params1)
        self.assertEqual(dist1.event_shape, dist.event_shape)
        self.assertEqual(type(dist1), td.Independent)
        self.assertEqual(type(dist1.base_dist), td.Normal)
        self.assertEqual(dist1.base_dist.mean, params1['loc'])
        self.assertEqual(dist1.base_dist.stddev, params1['scale'])

        self.assertRaises(AssertionError, spec.build_distribution,
                          {'loc': torch.tensor([1., 2.])})

    def test_transformed(self):
        normal_dist = dist_utils.DiagMultivariateNormal(
            torch.tensor([[1., 2.], [2., 2.]]),
            torch.tensor([[2., 3.], [1., 1.]]))
        transforms = [dist_utils.SigmoidTransform()]
        dist = td.TransformedDistribution(
            base_distribution=normal_dist, transforms=transforms)
        spec = dist_utils.DistributionSpec.from_distribution(dist)

        params1 = {
            'loc': torch.tensor([[0.5, 1.5], [1.0, 1.0]]),
            'scale': torch.tensor([[2., 4.], [2., 1.]])
        }
        dist1 = spec.build_distribution(params1)
        self.assertEqual(type(dist1), td.TransformedDistribution)
        self.assertEqual(dist1.event_shape, dist.event_shape)
        self.assertEqual(dist1.transforms, transforms)
        self.assertEqual(type(dist1.base_dist), td.Independent)
        self.assertEqual(type(dist1.base_dist.base_dist), td.Normal)
        self.assertEqual(dist1.base_dist.base_dist.mean, params1['loc'])
        self.assertEqual(dist1.base_dist.base_dist.stddev, params1['scale'])

    def test_inversion(self):
        x = torch.tensor([-10.0, -8.6, -2.0, 0, 2, 8.6, 10.0])
        loc = torch.tensor([0.5])
        scale = torch.tensor([1.5])
        transforms = [
            dist_utils.StableTanh(),
            dist_utils.AffineTransform(loc=loc, scale=scale)
        ]

        y = x
        # forward
        for transform in transforms:
            y = transform(y)

        # inverse
        x_recovered = y
        for transform in reversed(transforms):
            x_recovered = transform.inv(x_recovered)

        self.assertTensorEqual(x, x_recovered)
        self.assertTrue(x is x_recovered)


class TestConversions(alf.test.TestCase):
    def test_conversions(self):
        dists = {
            't':
                torch.tensor([[1., 2., 4.], [3., 3., 1.]]),
            'd':
                dist_utils.DiagMultivariateNormal(
                    torch.tensor([[1., 2.], [2., 2.]]),
                    torch.tensor([[2., 3.], [1., 1.]]))
        }
        params = dist_utils.distributions_to_params(dists)
        dists_spec = dist_utils.extract_spec(dists, from_dim=1)
        self.assertEqual(dists_spec['t'],
                         alf.TensorSpec(shape=(3, ), dtype=torch.float32))
        self.assertEqual(type(dists_spec['d']), dist_utils.DistributionSpec)
        self.assertEqual(len(params), 2)
        self.assertEqual(dists['t'], params['t'])
        self.assertEqual(dists['d'].base_dist.mean, params['d']['loc'])
        self.assertEqual(dists['d'].base_dist.stddev, params['d']['scale'])

        dists1 = dist_utils.params_to_distributions(params, dists_spec)
        self.assertEqual(len(dists1), 2)
        self.assertEqual(dists1['t'], dists['t'])
        self.assertEqual(type(dists1['d']), type(dists['d']))

        params_spec = dist_utils.to_distribution_param_spec(dists_spec)
        alf.nest.assert_same_structure(params_spec, params)
        params1_spec = dist_utils.extract_spec(params)
        self.assertEqual(params_spec, params1_spec)


class TestActionSamplingCategorical(alf.test.TestCase):
    def test_action_sampling_categorical(self):
        m = torch.distributions.categorical.Categorical(
            torch.Tensor([0.25, 0.75]))
        M = m.expand([10])
        epsilon = 0.0
        action_expected = torch.Tensor([1]).repeat(10)
        action_obtained = dist_utils.epsilon_greedy_sample(M, epsilon)
        self.assertTrue((action_expected == action_obtained).all())


class TestActionSamplingNormal(alf.test.TestCase):
    def test_action_sampling_normal(self):
        m = torch.distributions.normal.Normal(
            torch.Tensor([0.3, 0.7]), torch.Tensor([1.0, 1.0]))
        M = m.expand([10, 2])
        epsilon = 0.0
        action_expected = torch.Tensor([0.3, 0.7]).repeat(10, 1)
        action_obtained = dist_utils.epsilon_greedy_sample(M, epsilon)
        self.assertTrue((action_expected == action_obtained).all())


class TestActionSamplingTransformedNormal(alf.test.TestCase):
    def test_action_sampling_transformed_normal(self):
        def _get_transformed_normal(means, stds):
            normal_dist = td.Independent(td.Normal(loc=means, scale=stds), 1)
            transforms = [
                dist_utils.StableTanh(),
                dist_utils.AffineTransform(
                    loc=torch.tensor(0.), scale=torch.tensor(5.0))
            ]
            squashed_dist = td.TransformedDistribution(
                base_distribution=normal_dist, transforms=transforms)
            return squashed_dist, transforms

        means = torch.Tensor([0.3, 0.7])
        dist, transforms = _get_transformed_normal(
            means=means, stds=torch.Tensor([1.0, 1.0]))

        mode = dist_utils.get_mode(dist)

        transformed_mode = means
        for transform in transforms:
            transformed_mode = transform(transformed_mode)

        self.assertTrue((transformed_mode == mode).all())

        epsilon = 0.0
        action_obtained = dist_utils.epsilon_greedy_sample(dist, epsilon)
        self.assertTrue((transformed_mode == action_obtained).all())


class TestActionSamplingTransformedCategorical(alf.test.TestCase):
    def test_action_sampling_transformed_categorical(self):
        def _get_transformed_categorical(probs):
            categorical_dist = td.Independent(td.Categorical(probs=probs), 1)
            return categorical_dist

        probs = torch.Tensor([[0.3, 0.5, 0.2], [0.6, 0.4, 0.0]])
        dist = _get_transformed_categorical(probs=probs)
        mode = dist_utils.get_mode(dist)
        expected_mode = torch.argmax(probs, dim=1)

        self.assertTensorEqual(expected_mode, mode)

        epsilon = 0.0
        action_obtained = dist_utils.epsilon_greedy_sample(dist, epsilon)
        self.assertTensorEqual(expected_mode, action_obtained)


class TestRSampleActionDistribution(alf.test.TestCase):
    def test_rsample_action_distribution(self):
        c = torch.distributions.categorical.Categorical(
            torch.Tensor([0.25, 0.75]))
        C = c.expand([10])
        self.assertRaises(AssertionError,
                          dist_utils.rsample_action_distribution, C)

        n = torch.distributions.normal.Normal(
            torch.Tensor([0.3, 0.7]), torch.Tensor([1.0, 1.0]))
        N = n.expand([10, 2])

        action_distribution = ActionDistribution(a=C, b=N)
        self.assertRaises(AssertionError,
                          dist_utils.rsample_action_distribution,
                          action_distribution)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    alf.test.main()
