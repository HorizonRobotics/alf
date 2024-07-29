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
"""ParticleVI algorithm test."""

import math

from absl import logging
from absl.testing import parameterized
import torch
import torch.nn as nn

import alf
from alf.algorithms.particle_vi_algorithm import ParVIAlgorithm
from alf.networks import Network
from alf.tensor_specs import TensorSpec


class ParVIAlgorithmTest(parameterized.TestCase, alf.test.TestCase):
    def assertArrayEqual(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertLessEqual(float(torch.max(abs(x - y))), eps)

    def cov(self, data, rowvar=False):
        """Estimate a covariance matrix given data.

        Args:
            data (tensor): A 1-D or 2-D tensor containing multiple observations
                of multiple dimensions. Each row of ``mat`` represents a
                dimension of the observation, and each column a single
                observation.
            rowvar (bool): If True, then each row represents a dimension, with
                observations in the columns. Otherwise, each column represents
                a dimension while the rows contains observations.

        Returns:
            The covariance matrix
        """
        x = data.detach().clone()
        if x.dim() > 2:
            raise ValueError('data has more than 2 dimensions')
        if x.dim() < 2:
            x = x.view(1, -1)
        if not rowvar and x.size(0) != 1:
            x = x.t()
        fact = 1.0 / (x.size(1) - 1)
        x -= torch.mean(x, dim=1, keepdim=True)
        return fact * x.matmul(x.t()).squeeze()

    @parameterized.parameters(('svgd'), ('gfsf'), ('minmax'))
    def test_par_vi_algorithm(self, par_vi='svgd'):
        """
        The par_vi algorithm is trained to match the likelihood of a Gaussian
        distribution with zero mean and diagonal variance :math:`(1, 4)`.
        After training, :math:`w^T w` is the variance of the distribution
        implied by the particles. So it should be :math:`diag(1,4)`.
        """
        logging.info("par_vi: %s" % (par_vi))
        dim = 2
        num_particles = 256
        ParVI = ParVIAlgorithm(
            dim,
            num_particles=num_particles,
            par_vi=par_vi,
            critic_hidden_layers=(20, ),
            critic_optimizer=alf.optimizers.Adam(lr=1e-3),
            optimizer=alf.optimizers.AdamTF(lr=1e-2))

        var = torch.tensor([1, 4], dtype=torch.float32)
        precision = 1. / var

        def _neglogprob(x):
            return torch.squeeze(
                0.5 * torch.matmul(x * x, torch.reshape(precision, (dim, 1))),
                axis=-1)

        def _train():
            alg_step = ParVI.train_step(loss_func=_neglogprob)
            ParVI.update_with_gradient(alg_step.info)

        for i in range(2000):
            _train()
            learned_var = self.cov(ParVI.particles)
            if i % 500 == 0:
                print(i, "learned var=", learned_var)

        self.assertArrayEqual(torch.diag(var), learned_var, 0.4)


if __name__ == '__main__':
    alf.test.main()
