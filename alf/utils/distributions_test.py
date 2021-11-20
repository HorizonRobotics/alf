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

import numpy as np
import torch
import alf.utils.distributions as ad
import alf


class DistributionTest(alf.test.TestCase):
    def _test_its(self, x, its: ad.InverseTransformSampling):
        x.requires_grad_()
        y = its.cdf(x)
        x1 = its.icdf(y)
        p = its.log_prob(x).exp()
        step = x[1] - x[0]

        psum = p.sum() * step
        self.assertAlmostEqual(psum, 1., delta=0.01)
        self.assertTensorClose(x1, x, 0.01)

        grad = torch.autograd.grad(y.sum(), x)[0]
        self.assertTensorClose(grad.log(), p.log())

    def test_normal_its(self):
        self._test_its(torch.arange(-4., 4., 1 / 128), ad.NormalITS())

    def test_cauchy_its(self):
        self._test_its(torch.arange(-100., 100., 1 / 128), ad.CauchyITS())

    def test_t3_its(self):
        self._test_its(torch.arange(-20., 20., 1 / 128), ad.T2ITS())

    def _test_truncated(self, its: ad.InverseTransformSampling):
        batch_size = 6
        dim = 1
        lower_bound = -1.5 * torch.ones(dim)
        upper_bound = 2.5 * torch.ones(dim)

        loc = torch.ones((batch_size, dim))
        loc[0, :] = -2
        loc[1, :] = -1
        loc[2, :] = 0
        loc[3, :] = 1
        loc[4, :] = 2
        loc[5, :] = 2

        scale = torch.ones((6, dim))
        scale[2, :] = 0.5
        scale[3, :] = 1.5

        dist = ad.TruncatedDistribution(
            loc=loc,
            scale=scale,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            its=its)

        # Test prob sum to 1.
        step = 1 / 128
        x = torch.arange(-1.5, 2.5, step)[:, None, None].expand(
            -1, batch_size, dim)
        log_prob = dist.log_prob(x)
        prob = log_prob.exp() * step
        self.assertTensorClose(
            prob.sum(dim=0), torch.ones((batch_size, )), 0.01)

        # Test samples are within bound
        samples = dist.rsample((1000, ))
        self.assertTrue((samples > lower_bound).all())
        self.assertTrue((samples < upper_bound).all())

    def test_truncated_normal(self):
        self._test_truncated(ad.NormalITS())

    def test_truncated_cauchy(self):
        self._test_truncated(ad.CauchyITS())

    def test_truncated_T2(self):
        self._test_truncated(ad.T2ITS())

    def test_truncated_normal_mode(self):
        dist = ad.TruncatedNormal(
            loc=torch.Tensor([[1.5, -3.0, 4.5]]),
            scale=torch.tensor([[0.8, 1.9, 1.2]]),
            lower_bound=torch.tensor([1.0, 1.0, 1.0]),
            upper_bound=torch.tensor([2.0, 2.0, 2.0]))
        self.assertTrue(torch.all(torch.tensor([1.5, 1.0, 2.0]) == dist.mode))

    def test_truncated_normal_kl_divergence(self):
        def _numerical_kl_divergence(lower_bound, upper_bound, loc_p, scale_p,
                                     loc_q, scale_q):
            p = ad.TruncatedNormal(loc_p, scale_p, lower_bound, upper_bound)
            q = ad.TruncatedNormal(loc_q, scale_q, lower_bound, upper_bound)

            delta = 1e-3
            accu = torch.as_tensor(0.0)
            for x in np.arange(lower_bound, upper_bound, delta):
                log_p_x = p.log_prob(torch.as_tensor(x))
                log_q_x = q.log_prob(torch.as_tensor(x))

                accu += torch.exp(log_p_x) * (log_p_x - log_q_x) * delta

            return accu

        dim = 1
        lower_bound = -1.5 * torch.ones(dim)
        upper_bound = 2.5 * torch.ones(dim)

        batch_size = 4

        loc1 = torch.ones((batch_size, dim))
        loc1[0, :] = -2.0
        loc1[1, :] = -1.0
        loc1[2, :] = 0.0
        loc1[3, :] = 1.0

        scale1 = torch.ones((batch_size, dim))
        scale1[1, :] = 0.5
        scale1[2, :] = 1.5
        scale1[3, :] = 2.56

        dist1 = ad.TruncatedNormal(
            loc=loc1,
            scale=scale1,
            lower_bound=lower_bound,
            upper_bound=upper_bound)

        loc2 = torch.ones((batch_size, dim))
        loc2[0, :] = -1.0
        loc2[1, :] = -2.0
        loc2[2, :] = 1.0
        loc2[3, :] = 3.0

        scale2 = torch.ones((batch_size, dim))
        scale2[0, :] = 0.2
        scale2[1, :] = 1.5
        scale2[2, :] = 0.5

        dist2 = ad.TruncatedNormal(
            loc=loc2,
            scale=scale2,
            lower_bound=lower_bound,
            upper_bound=upper_bound)

        kl = torch.distributions.kl_divergence(dist1, dist2)

        for i in range(batch_size):
            expected = _numerical_kl_divergence(lower_bound[0], upper_bound[0],
                                                loc1[i][0], scale1[i][0],
                                                loc2[i][0], scale2[i][0])
            np.testing.assert_array_almost_equal(kl[i], expected, decimal=3)


if __name__ == '__main__':
    alf.test.main()
