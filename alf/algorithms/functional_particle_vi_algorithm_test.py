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

import absl
from absl.testing import parameterized
import numpy as np
import torch
import torch.nn.functional as F

import alf
from alf.algorithms.functional_particle_vi_algorithm import FuncParVIAlgorithm
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops
from alf.utils.datagen import TestDataSet


class FuncParVIAlgorithmTest(parameterized.TestCase, alf.test.TestCase):
    def cov(self, data, rowvar=False):
        """Estimate a covariance matrix given data.

        Args:
            data (tensor): A 1-D or 2-D tensor containing multiple observations 
                of multiple dimentions. Each row of ``mat`` represents a
                dimension of the observation, and each column a single
                observation.
            rowvar (bool): If True, then each row represents a dimension, with
                observations in the columns. Othewise, each column represents
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

    def assertArrayGreater(self, x, y, eps):
        self.assertEqual(x.shape, y.shape)
        self.assertGreater(float(torch.min(x - y)), eps)

    @parameterized.parameters(('gfsf'), ('svgd'), ('gfsf', True),
                              ('svgd', True))
    def test_functional_par_vi_algorithm(self,
                                         par_vi='svgd',
                                         function_vi=False,
                                         num_particles=256,
                                         batch_size=10):
        """
        The hypernetwork is trained to generate the parameter vector for a linear
        regressor. The target linear regressor is :math:`y = X\beta + e`, where 
        :math:`e\sim N(0, I)` is random noise, :math:`X` is the input data matrix, 
        and :math:`y` is target ouputs. The posterior of :math:`\beta` has a 
        closed-form :math:`p(\beta|X,y)\sim N((X^TX)^{-1}X^Ty, X^TX)`.
        For a linear generator with weight W and bias b, and takes standard Gaussian 
        noise as input, the output follows a Gaussian :math:`N(b, WW^T)`, which should 
        match the posterior :math:`p(\beta|X,y)` for both svgd and gfsf.
        
        """
        input_dim = 3
        input_spec = TensorSpec((input_dim, ), torch.float32)
        output_dim = 1
        size = 150
        beta = torch.rand(input_dim, output_dim) + 5.
        absl.logging.info("beta: {}".format(beta))

        trainset = TestDataSet(
            input_dim=input_dim, output_dim=output_dim, size=size, weight=beta)
        testset = TestDataSet(
            input_dim=input_dim, output_dim=output_dim, size=size, weight=beta)
        inputs = trainset.get_features()
        targets = trainset.get_targets()
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1)
        true_cov = torch.inverse(inputs.t() @ inputs)
        true_mean = true_cov @ inputs.t() @ targets

        algorithm = FuncParVIAlgorithm(
            input_tensor_spec=input_spec,
            last_layer_param=(output_dim, False),
            last_activation=math_ops.identity,
            num_particles=num_particles,
            loss_type='regression',
            par_vi=par_vi,
            function_vi=function_vi,
            function_bs=batch_size,
            optimizer=alf.optimizers.Adam(lr=1e-2),
            logging_evaluate=True)

        algorithm.set_data_loader(
            train_loader,
            test_loader=test_loader,
            entropy_regularization=batch_size / size)
        absl.logging.info("ground truth mean: {}".format(true_mean))
        absl.logging.info("ground truth cov: {}".format(true_cov))
        absl.logging.info("ground truth cov norm: {}".format(true_cov.norm()))

        def _test(i):
            params = algorithm.particles
            computed_mean = params.mean(0)
            computed_cov = self.cov(params)

            absl.logging.info("-" * 68)
            pred_step = algorithm.predict_step(inputs)
            preds = pred_step.output.squeeze()  # [batch, n_particles]
            computed_preds = inputs @ computed_mean  # [batch]

            pred_err = torch.norm((preds - targets).mean(1))

            mean_err = torch.norm(computed_mean - true_mean.squeeze())
            mean_err = mean_err / torch.norm(true_mean)

            cov_err = torch.norm(computed_cov - true_cov)
            cov_err = cov_err / torch.norm(true_cov)

            absl.logging.info("train_iter {}: pred err {}".format(i, pred_err))
            absl.logging.info("train_iter {}: mean err {}".format(i, mean_err))
            absl.logging.info("train_iter {}: cov err {}".format(i, cov_err))
            absl.logging.info("computed_cov norm: {}".format(
                computed_cov.norm()))

        train_iter = 2000
        for i in range(train_iter):
            algorithm.train_iter()
            if i % 1000 == 0:
                _test(i)

        algorithm.evaluate()

        params = algorithm.particles
        computed_mean = params.mean(0)
        computed_cov = self.cov(params)
        mean_err = torch.norm(computed_mean - true_mean.squeeze())
        mean_err = mean_err / torch.norm(true_mean)
        cov_err = torch.norm(computed_cov - true_cov)
        cov_err = cov_err / torch.norm(true_cov)
        absl.logging.info("-" * 68)
        absl.logging.info("train_iter {}: mean err {}".format(
            train_iter, mean_err))
        absl.logging.info("train_iter {}: cov err {}".format(
            train_iter, cov_err))

        self.assertLess(mean_err, 0.5)
        self.assertLess(cov_err, 0.5)


if __name__ == "__main__":
    alf.test.main()
