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


if __name__ == '__main__':
    alf.test.main()
