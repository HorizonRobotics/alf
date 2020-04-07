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
import math
import torch
import torch.distributions as td
import torch.nn.functional as F

import alf
from alf.algorithms.mi_estimator import MIEstimator, ScalarAdaptiveAverager
from alf.networks import Network
from alf.utils import math_ops
from alf.utils.dist_utils import DiagMultivariateNormal


class Net_YGivenZX_manual(Network):
    def __init__(self, input_spec):
        super().__init__(input_tensor_spec=input_spec, name="Net")

    def forward(self, input, state=()):
        z, x = input
        z = F.relu(z)
        return torch.cat([z, x * z], dim=-1), ()


class NetML(Network):
    def __init__(self, input_spec):
        super().__init__(input_tensor_spec=input_spec, name="Net")
        size = sum([x.shape[-1] for x in alf.nest.flatten(input_spec)])
        self._l1a = alf.layers.FC(size, 128, activation=F.relu)
        self._l1b = alf.layers.FC(size, 128, activation=F.relu)
        self._l1 = alf.layers.FC(128, 128, activation=F.relu)

    def forward(self, input, state=()):
        x = torch.cat(alf.nest.flatten(input), dim=-1)
        x = self._l1(self._l1a(x) * self._l1b(x))
        return x, ()


class NetJSD(Network):
    def __init__(self, input_spec):
        super().__init__(input_tensor_spec=input_spec, name="Net")
        size = sum([x.shape[-1] for x in alf.nest.flatten(input_spec)])
        self._l1a = alf.layers.FC(size, 128, activation=F.relu)
        self._l1b = alf.layers.FC(size, 128, activation=F.relu)

        self._l2a = alf.layers.FC(128, 128, activation=F.relu)
        self._l2b = alf.layers.FC(128, 128, activation=F.relu)
        self._l_out = alf.layers.FC(128, 1)

    def forward(self, input):
        x = torch.cat(alf.nest.flatten(input), dim=-1)
        x = self._l1a(x) * self._l1b(x)
        x = self._l2a(x) * self._l2b(x)
        return self._l_out(x), ()


class MIEstimatorTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters(
        dict(estimator='DV', rho=0.0, eps=0.03),
        dict(estimator='KLD', rho=0.0, eps=0.03),
        dict(estimator='JSD', rho=0.0, eps=0.03),
        dict(estimator='DV', rho=0.5, eps=0.4),
        dict(estimator='DV', rho=0.5, eps=0.4, sampler='double_buffer'),
        dict(estimator='DV', rho=0.5, eps=0.4, buffer_size=1048576),
        dict(estimator='DV', rho=0.5, eps=0.6, sampler='shuffle'),
        dict(estimator='DV', rho=0.5, eps=0.4, sampler='shift'),
        dict(estimator='KLD', rho=0.5, eps=0.4),
        dict(estimator='JSD', rho=0.5, eps=0.4),
        dict(estimator='DV', rho=0.9, eps=7.0),
        dict(estimator='KLD', rho=0.9, eps=7.0),
        dict(estimator='JSD', rho=0.9, eps=12.0),
    )
    def test_mi_estimator(self,
                          estimator='DV',
                          sampler='buffer',
                          rho=0.9,
                          eps=1000.0,
                          buffer_size=65536,
                          dim=20):
        mi_estimator = MIEstimator(
            x_spec=[
                alf.TensorSpec(shape=(dim // 3, ), dtype=torch.float32),
                alf.TensorSpec(shape=(dim - dim // 3, ), dtype=torch.float32)
            ],
            y_spec=[
                alf.TensorSpec(shape=(dim // 2, ), dtype=torch.float32),
                alf.TensorSpec(shape=(dim // 2, ), dtype=torch.float32)
            ],
            fc_layers=(512, ),
            buffer_size=buffer_size,
            estimator_type=estimator,
            sampler=sampler,
            averager=ScalarAdaptiveAverager(),
            optimizer=alf.optimizers.AdamTF(lr=1e-4))

        a = 0.5 * (math.sqrt(1 + rho) + math.sqrt(1 - rho))
        b = 0.5 * (math.sqrt(1 + rho) - math.sqrt(1 - rho))
        # This matrix transforms standard Gaussian to a Gaussian with variance
        # [[1, rho], [rho, 1]]
        w = torch.tensor([[a, b], [b, a]], dtype=torch.float32)
        var = torch.matmul(w, w)
        entropy = 0.5 * torch.logdet(2 * math.pi * math.e * var)
        entropy_x = 0.5 * torch.log(2 * math.pi * math.e * var[0, 0])
        entropy_y = 0.5 * torch.log(2 * math.pi * math.e * var[1, 1])
        mi = float(dim * (entropy_x + entropy_y - entropy))

        def _get_batch(batch_size):
            xy = torch.randn(batch_size * dim, 2)
            xy = torch.matmul(xy, w)
            x = xy[:, 0]
            y = xy[:, 1]
            x = x.reshape(-1, dim)
            y = y.reshape(-1, dim)
            x = [x[..., :dim // 3], x[..., dim // 3:]]
            y = [y[..., :dim // 2], y[..., dim // 2:]]
            return x, y

        def _calc_estimated_mi(i, mi_samples):
            estimated_mi = mi_samples.mean(dim=0)
            var = torch.var(mi_samples, dim=0, unbiased=False)
            estimated_mi = float(estimated_mi)
            # For DV estimator, the following std is an approximated std.
            logging.info(
                "%s estimated mi=%s std=%s" %
                (i, estimated_mi, math.sqrt(var / mi_samples.shape[0])))
            return estimated_mi

        batch_size = 512
        info = "mi=%s estimator=%s buffer_size=%s sampler=%s dim=%s" % (
            float(mi), estimator, buffer_size, sampler, dim)

        def _train():
            x, y = _get_batch(batch_size)
            alg_step = mi_estimator.train_step((x, y))
            mi_estimator.update_with_gradient(alg_step.info)
            return alg_step

        for i in range(5000):
            alg_step = _train()
            if i % 1000 == 0:
                _calc_estimated_mi(i, alg_step.output)
        x, y = _get_batch(16384)
        log_ratio = mi_estimator.calc_pmi(x, y)
        estimated_mi = _calc_estimated_mi(info, log_ratio)
        if estimator == 'JSD':
            self.assertAlmostEqual(estimated_mi, mi, delta=eps)
        else:
            self.assertLess(estimated_mi, mi)
            self.assertGreater(estimated_mi, mi - eps)
        return mi, estimated_mi

    @parameterized.parameters(
        dict(
            estimator='JSD', switch_xy=False, use_default_model=True, eps=0.2),
        dict(
            estimator='JSD', switch_xy=False, use_default_model=False,
            eps=0.2),
        dict(estimator='JSD', switch_xy=True, use_default_model=True, eps=0.2),
        dict(
            estimator='JSD', switch_xy=True, use_default_model=False, eps=0.2),
        dict(estimator='ML', switch_xy=False, use_default_model=True),
        dict(estimator='ML', switch_xy=False, use_default_model=False),
        dict(estimator='ML', switch_xy=True, use_default_model=True),
        dict(estimator='ML', switch_xy=True, use_default_model=False),
    )
    def __test_conditional_mi_estimator(self,
                                        estimator='ML',
                                        switch_xy=False,
                                        use_default_model=True,
                                        eps=0.02,
                                        dim=2):
        """Estimate the conditional mutual information MI(X;Y|Z)

        X, Y and Z are generated by the following procedure:
            Z ~ N(0, 1)
            X|z ~ N(z, 1)
            if z >= 0:
                Y|x,z ~ N(z + xz, e^2)
            else:
                Y|x,z ~ N(0, 1)
        When z>0,
            [X, Y] ~ N([z, z+z^2], [[1, z], [z, e^2+z^2]])
            MI(X;Y|z) = 0.5 * log(1+z^2/e^2)
        """
        x_spec = [
            alf.TensorSpec(shape=(dim, ), dtype=torch.float32),
            alf.TensorSpec(shape=(dim, ), dtype=torch.float32)
        ]
        y_spec = alf.TensorSpec(shape=(dim, ), dtype=torch.float32)
        if use_default_model:
            model = None
        elif estimator == 'ML':
            model = NetML(x_spec)
        else:
            model = NetJSD([x_spec, y_spec])
        mi_estimator = MIEstimator(
            x_spec=x_spec,
            y_spec=y_spec,
            fc_layers=(256, 256),
            model=model,
            estimator_type=estimator,
            optimizer=alf.optimizers.AdamTF(lr=2e-4))

        z = torch.randn(10000, )
        e = 0.5
        mi = 0.25 * dim * torch.mean(torch.log(1 + (z / e)**2))

        def _get_batch(batch_size, z=None):
            if z is None:
                z = torch.randn(batch_size, dim)
            x_dist = DiagMultivariateNormal(loc=z, scale=torch.ones_like(z))
            mask = (z > 0).to(torch.float32)
            y_dist = DiagMultivariateNormal(
                loc=(z + z * z) * mask,
                scale=1 - mask + mask * torch.sqrt(e * e + z * z))
            x = x_dist.sample()
            y = (z + x * z) * mask + (1 - mask + e * mask) * torch.randn(
                batch_size, dim)
            if not switch_xy:
                X = [z, x]
                Y = y
                Y_dist = y_dist
            else:
                X = [z, y]
                Y = x
                Y_dist = x_dist
            return dict(x=x, y=y, z=z, X=X, Y=Y, Y_dist=Y_dist)

        def _estimate_mi(i, batch):
            estimated_pmi = mi_estimator.calc_pmi(batch['X'], batch['Y'],
                                                  batch['Y_dist'])
            batch_size = estimated_pmi.shape[0]
            x, y, z = batch['x'], batch['y'], batch['z']
            pmi = 0.5 * (math_ops.square(y - z - z * z) /
                         (e * e + z * z) - math_ops.square(y - z - x * z) /
                         (e * e) + torch.log(1 + (z / e)**2))
            pmi = pmi * (z > 0).to(torch.float32)
            pmi = torch.sum(pmi, dim=-1)
            pmi_rmse = torch.sqrt(
                torch.mean(math_ops.square(pmi - estimated_pmi)))
            estimated_mi = estimated_pmi.mean(dim=0)
            var = torch.var(estimated_pmi, dim=0, unbiased=False)
            estimated_mi = float(estimated_mi)
            logging.info("%s estimated_mi=%s std=%s pmi_rmse=%s" % (
                i, estimated_mi, math.sqrt(var / batch_size), float(pmi_rmse)))
            return estimated_mi

        batch_size = 512

        info = "mi=%s estimator=%s use_default_model=%s switch_xy=%s dim=%s" % (
            float(mi), estimator, use_default_model, switch_xy, dim)

        def _train():
            batch = _get_batch(batch_size)
            alg_step = mi_estimator.train_step((batch['X'], batch['Y']),
                                               y_distribution=batch['Y_dist'])
            mi_estimator.update_with_gradient(alg_step.info)
            return alg_step

        for i in range(20000):
            _train()
            if i % 1000 == 0:
                batch = _get_batch(batch_size)
                _estimate_mi(i, batch)

        batch_size = 16384
        batch = _get_batch(batch_size)
        estimated_mi = _estimate_mi(info, batch)
        self.assertAlmostEqual(estimated_mi, mi, delta=eps)

        # Set detail_reault=True to show the conditional mutual information for
        # different values of z
        detail_result = False
        if detail_result:
            for z in torch.arange(-2., 2.001, 0.125):
                batch = _get_batch(batch_size, z * torch.ones(batch_size, dim))
                info = "z={z} mi={mi}".format(
                    z=float(z),
                    mi=float(
                        0.5 * torch.log(1 + math_ops.square(F.relu(z / e)))))
                _estimate_mi(info, batch)

        return mi, estimated_mi


if __name__ == '__main__':
    alf.test.main()
