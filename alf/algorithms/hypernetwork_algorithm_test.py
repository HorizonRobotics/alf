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

from absl.testing import parameterized
import numpy as np
import torch
import torch.nn.functional as F

import alf
from alf.algorithms.hypernetwork_algorithm import HyperNetwork
from alf.algorithms.hypernetwork_networks import ParamConvNet, ParamNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import math_ops


class HyperNetworkTest(parameterized.TestCase, alf.test.TestCase):
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

    @parameterized.parameters(('gfsf', 512), ('gfsf', 512, 100), ('svgd2'),
                              ('svgd2', 32, 100), ('svgd3'),
                              ('svgd3', 32, 100))
    def test_bayesian_linear_regression(self,
                                        par_vi='svgd3',
                                        particles=32,
                                        train_batch_size=10):
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
        input_size = 3
        input_spec = TensorSpec((input_size, ), torch.float32)
        output_dim = 1
        batch_size = 100
        inputs = input_spec.randn(outer_dims=(batch_size, ))
        beta = torch.rand(input_size, output_dim) + 5.
        print("beta: {}".format(beta))
        noise = torch.randn(batch_size, output_dim)
        targets = inputs @ beta + noise
        true_cov = torch.inverse(
            inputs.t() @ inputs)  # + torch.eye(input_size))
        true_mean = true_cov @ inputs.t() @ targets
        noise_dim = 3
        algorithm = HyperNetwork(
            input_tensor_spec=input_spec,
            last_layer_param=(output_dim, False),
            last_activation=math_ops.identity,
            noise_dim=noise_dim,
            # hidden_layers=(16, ),
            hidden_layers=None,
            loss_type='regression',
            par_vi=par_vi,
            optimizer=alf.optimizers.Adam(lr=1e-4))
        print("ground truth mean: {}".format(true_mean))
        print("ground truth cov norm: {}".format(true_cov.norm()))
        print("ground truth cov: {}".format(true_cov))

        def _train():
            perm = torch.randperm(batch_size)
            idx = perm[:train_batch_size]
            train_inputs = inputs[idx]
            train_targets = targets[idx]
            alg_step = algorithm.train_step(
                inputs=(train_inputs, train_targets),
                entropy_regularization=train_batch_size / batch_size,
                particles=particles)

            algorithm.update_with_gradient(alg_step.info)

        def _test(i):

            params = algorithm.sample_parameters(particles=200)
            computed_mean = params.mean(0)
            computed_cov = self.cov(params)

            print("-" * 68)
            weight = algorithm._generator._net._fc_layers[0].weight
            print("norm of generator weight: {}".format(weight.norm()))
            learned_cov = weight @ weight.t()
            learned_mean = algorithm._generator._net._fc_layers[0].bias

            pred_step = algorithm.predict_step(inputs, params=params)
            sampled_preds = pred_step.output.squeeze()  # [batch, particles]

            computed_preds = inputs @ computed_mean  # [batch]
            predicts = inputs @ learned_mean  # [batch]

            spred_err = torch.norm((sampled_preds - targets).mean(1))
            cpred_err = torch.norm(computed_preds - targets.squeeze())
            pred_err = torch.norm(predicts - targets.squeeze())

            smean_err = torch.norm(computed_mean - true_mean.squeeze())
            smean_err = smean_err / torch.norm(true_mean)

            mean_err = torch.norm(learned_mean - true_mean.squeeze())
            mean_err = mean_err / torch.norm(true_mean)

            scov_err = torch.norm(computed_cov - true_cov)
            scov_err = scov_err / torch.norm(true_cov)

            cov_err = torch.norm(learned_cov - true_cov)
            cov_err = cov_err / torch.norm(true_cov)
            print("train_iter {}: pred err {}".format(i, pred_err))
            print("train_iter {}: sampled pred err {}".format(i, spred_err))
            print("train_iter {}: mean err {}".format(i, mean_err))
            print("train_iter {}: sampled mean err {}".format(i, smean_err))
            print("train_iter {}: sampled cov err {}".format(i, scov_err))
            print("train_iter {}: cov err {}".format(i, cov_err))
            print("learned_cov norm: {}".format(learned_cov.norm()))

        # train_iter = 1000
        # for t in range(int(batch_size / train_batch_size)):
        #     train_inputs = inputs[t*train_batch_size : (t+1)*train_batch_size]
        #     train_targets = targets[t*train_batch_size : (t+1)*train_batch_size]
        #     for j in range(train_iter):
        #         alg_step = algorithm.train_step(
        #             inputs=(train_inputs, train_targets),
        #             entropy_regularization=1/(t+1),
        #             particles=particles)
        #         algorithm.update_with_gradient(alg_step.info)
        #     _test(t)

        for i in range(40000):
            _train()
            if i % 1000 == 0:
                _test(i)

        learned_mean = algorithm._generator._net._fc_layers[0].bias
        mean_err = torch.norm(learned_mean - true_mean.squeeze())
        mean_err = mean_err / torch.norm(true_mean)
        weight = algorithm._generator._net._fc_layers[0].weight
        learned_cov = weight @ weight.t()
        cov_err = torch.norm(learned_cov - true_cov)
        cov_err = cov_err / torch.norm(true_cov)

        self.assertLess(mean_err, 0.01)
        self.assertLess(cov_err, 0.1)

    def test_hypernetwork_classification(self):
        # TODO: out of distribution tests
        # If simply use a linear classifier with random weights,
        # the cross_entropy loss does not seem to capture the distribution.
        pass


if __name__ == "__main__":
    alf.test.main()
