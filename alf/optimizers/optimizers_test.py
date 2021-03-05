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

from absl.testing import parameterized
import torch
import torch.nn as nn

import alf
from alf.algorithms.hypernetwork_algorithm import regression_loss
from alf.optimizers import Adam, AdamTF
from alf.optimizers.optimizers import score_func
from alf.tensor_specs import TensorSpec
from alf.utils import tensor_utils


class LRBatchEnsemble(nn.Module):
    def __init__(self, input_size, ensemble_size, gfsf_group=None):
        nn.Module.__init__(self)
        self._beta = nn.Parameter(torch.rand(ensemble_size, input_size))
        if gfsf_group is not None:
            assert isinstance(gfsf_group,
                              int), ("gfsf_group has to be an integer!")
            self._beta.gfsf_group = gfsf_group
        self._input_size = input_size
        self._ensemble_size = ensemble_size

    def forward(self, inputs):
        """Forward computation

        Args:
            inputs (Tensor): shape of ``[batch_size, input_size]``

        Returns:
            outputs (Tensor): shape of ``[ensemble_size, batch_size]``
        """
        assert inputs.ndim == 2 and inputs.shape[-1] == self._input_size
        return self._beta @ inputs.t()


class OptimizersTest(parameterized.TestCase, alf.test.TestCase):
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

    def test_optimizer_name(self):
        i = Adam.counter
        j = AdamTF.counter
        opt1 = Adam(lr=0.1)
        opt2 = AdamTF(lr=0.1)
        opt3 = Adam(lr=0.1)
        opt4 = AdamTF(lr=0.1, name="AdamTF")
        self.assertEqual(opt1.name, "Adam_%s" % i)
        self.assertEqual(opt2.name, "AdamTF_%s" % j)
        self.assertEqual(opt3.name, "Adam_%s" % (i + 1))
        self.assertEqual(opt4.name, "AdamTF")

    def test_gradient_clipping(self):
        layer = torch.nn.Linear(5, 3)
        x = torch.randn(2, 5)
        y = layer(x)
        loss = torch.sum(y**2)
        clip_norm = 1e-4
        opt = AdamTF(
            lr=0.1, gradient_clipping=clip_norm, clip_by_global_norm=True)
        opt.add_param_group({'params': layer.parameters()})
        opt.zero_grad()
        loss.backward()

        def _grad_norm(params):
            grads = [p.grad for p in params]
            return tensor_utils.global_norm(grads)

        params = []
        for param_group in opt.param_groups:
            params.extend(param_group["params"])
        self.assertGreater(_grad_norm(params), clip_norm)
        opt.step()
        self.assertTensorClose(_grad_norm(params), torch.as_tensor(clip_norm))

    # def test_gfsf_grad(self):
    #     ensemble_size = 4
    #     input_size = 3
    #     output_size = 2
    #     batch_size = 2
    #     W = torch.nn.Parameter(
    #         torch.rand(ensemble_size, input_size, output_size))
    #     W.gfsf_group = 0
    #     x = torch.randn(batch_size, input_size)
    #     y = torch.einsum('nij,bi->nbj', W, x)
    #     loss = torch.sum(y**2)
    #     opt = AdamTF(lr=0.1, gfsf_grad_weight=1.)
    #     opt.add_param_group({'params': W})
    #     opt.zero_grad()
    #     loss.backward()
    #     params = [W]
    #     opt.step()
    #     assertTensorClose(parms, W)

    @parameterized.parameters(
        dict(train_batch_size=100), dict(train_batch_size=10))
    def test_batch_ensemble_gfsf_blr(self,
                                     train_batch_size=10,
                                     num_particles=100):
        input_size = 3
        input_spec = TensorSpec((input_size, ), torch.float32)
        output_size = 1
        batch_size = 100
        inputs = input_spec.randn(outer_dims=(batch_size, ))
        beta = torch.rand(input_size, output_size) + 5.
        print("beta: {}".format(beta))
        noise = torch.randn(batch_size, output_size)
        targets = inputs @ beta + noise
        true_cov = torch.inverse(
            inputs.t() @ inputs)  # + torch.eye(input_size))
        true_mean = true_cov @ inputs.t() @ targets
        print("ground truth mean: {}".format(true_mean))
        print("ground truth cov: {}".format(true_cov))
        print("ground truth cov norm: {}".format(true_cov.norm()))

        layer = LRBatchEnsemble(input_size, num_particles, gfsf_group=0)
        entropy_regularization = train_batch_size / batch_size
        optimizer = alf.optimizers.Adam(
            lr=1e-2, gfsf_grad_weight=entropy_regularization)
        optimizer.add_param_group({'params': layer._beta})

        def _train(train_batch=None):
            if train_batch is None:
                perm = torch.randperm(batch_size)
                idx = perm[:train_batch_size]
                train_inputs = inputs[idx]
                train_targets = targets[idx]
            else:
                train_inputs, train_targets = train_batch

            layer_outputs = layer(train_inputs)
            train_targets = train_targets.view(-1)
            train_targets = train_targets.unsqueeze(0).expand(
                num_particles, *train_targets.shape)
            loss = regression_loss(layer_outputs, train_targets).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        def _test(i, sampled_predictive=False):
            print("-" * 68)
            learned_cov = self.cov(layer._beta)
            print("norm of learned_cov: {}".format(learned_cov.norm()))

            predicts = layer(inputs)
            pred_err = torch.norm(predicts.mean(0) - targets.squeeze())
            print("train_iter {}: pred err {}".format(i, pred_err))

            learned_mean = layer._beta.mean(0)
            mean_err = torch.norm(learned_mean - true_mean.squeeze())
            mean_err = mean_err / torch.norm(true_mean)
            print("train_iter {}: mean err {}".format(i, mean_err))

            cov_err = torch.norm(learned_cov - true_cov)
            cov_err = cov_err / torch.norm(true_cov)
            print("train_iter {}: cov err {}".format(i, cov_err))

        train_iter = 3000
        for i in range(train_iter):
            _train()
            if i % 1000 == 0:
                _test(i)

        learned_mean = layer._beta.mean(0)
        mean_err = torch.norm(learned_mean - true_mean.squeeze())
        mean_err = mean_err / torch.norm(true_mean)
        learned_cov = self.cov(layer._beta)
        cov_err = torch.norm(learned_cov - true_cov)
        cov_err = cov_err / torch.norm(true_cov)
        print("-" * 68)
        print("train_iter {}: mean err {}".format(train_iter, mean_err))
        print("train_iter {}: cov err {}".format(train_iter, cov_err))

        self.assertLess(mean_err, 0.5)
        self.assertLess(cov_err, 0.5)


if __name__ == "__main__":
    alf.test.main()
