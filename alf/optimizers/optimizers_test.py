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
import copy
import os
import tempfile
import torch
import torch.nn as nn
from unittest import mock

import alf
from alf.algorithms.hypernetwork_algorithm import regression_loss
from alf.algorithms.particle_vi_algorithm import ParVIAlgorithm
from alf.optimizers import Adam, AdamTF
from alf.tensor_specs import TensorSpec
from alf.utils import common, tensor_utils


class GradientBatchingTest(parameterized.TestCase, alf.test.TestCase):

    @parameterized.product(
        device=('cpu', 'cuda'),
        dtype=(torch.float32, torch.float64, torch.bfloat16),
        accumulation=(1, 3),
        clip_norm=(None, 0.5, 100.0))
    def test_matches_unbatched_updates(self, device, dtype, accumulation,
                                       clip_norm):
        if device == 'cuda' and not torch.cuda.is_available():
            self.skipTest('CUDA is required')
        params = [
            nn.Parameter(torch.ones(2, device=device, dtype=d))
            for d in (torch.float32, dtype)
        ]
        unused = nn.Parameter(torch.ones(2, device=device))
        reference = [p.detach().clone() for p in params]
        opt = alf.optimizers.SGD(lr=0.1,
                                 grad_accumulation_steps=accumulation,
                                 gradient_clipping=clip_norm,
                                 clip_by_global_norm=True,
                                 foreach=False)
        # Clipping must use one global norm, not one norm per optimizer group.
        opt.add_param_group({'params': [params[0], unused]})
        opt.add_param_group({'params': [params[1]]})
        tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-5
        for cycle in range(2):
            expected_grads = [torch.zeros_like(p) for p in params]
            for microstep in range(accumulation):
                opt.zero_grad(set_to_none=True)
                gradients = [
                    p.new_tensor([3., 4.]) * (cycle + microstep + i + 1)
                    for i, p in enumerate(params)
                ]
                sum((p * g).sum() for p, g in zip(params, gradients)).backward()
                for total, grad in zip(expected_grads, gradients):
                    total.add_(grad)
                opt.step()
                if microstep + 1 < accumulation:
                    self.assertFalse(opt._grad_stepped)
                    for actual, expected in zip(params, reference):
                        torch.testing.assert_close(actual, expected)
                    for actual, expected in zip(params, expected_grads):
                        torch.testing.assert_close(actual.grad, expected)
            for grad in expected_grads:
                grad.mul_(1.0 / accumulation)
            if clip_norm is not None:
                tensor_utils.clip_by_global_norm(
                    expected_grads, clip_norm, in_place=True)
            for actual, expected, grad in zip(params, reference, expected_grads):
                expected.add_(grad, alpha=-0.1)
                torch.testing.assert_close(actual.grad, grad,
                                           rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(actual, expected,
                                           rtol=tolerance, atol=tolerance)
            self.assertTrue(opt._grad_stepped)
            self.assertIsNone(unused.grad)
            torch.testing.assert_close(unused, torch.ones_like(unused))

    def test_foreach_only_at_update_boundary(self):
        param = nn.Parameter(torch.ones(2, device='cpu'))
        opt = alf.optimizers.SGD(lr=0.1, grad_accumulation_steps=2,
                                 gradient_clipping=0.5,
                                 clip_by_global_norm=True, foreach=False)
        opt.add_param_group({'params': [param]})
        with mock.patch('torch._foreach_mul_', wraps=torch._foreach_mul_) as mul, \
                mock.patch('torch.nn.utils.clip_grad_norm_',
                           wraps=torch.nn.utils.clip_grad_norm_) as clip:
            param.sum().backward()
            opt.step()
            mul.assert_not_called()
            clip.assert_not_called()
            opt.zero_grad(set_to_none=True)
            param.sum().backward()
            opt.step()
            self.assertGreaterEqual(mul.call_count, 2)  # averaging and clipping
            clip.assert_called_once()
            self.assertTrue(clip.call_args.kwargs['foreach'])

    @parameterized.parameters(False, True)
    def test_missing_or_zero_gradients(self, zero_gradients):
        param = nn.Parameter(torch.ones(2, device='cpu'))
        opt = alf.optimizers.SGD(lr=0.1, grad_accumulation_steps=2,
                                 gradient_clipping=0.5,
                                 clip_by_global_norm=True, foreach=False)
        opt.add_param_group({'params': [param]})
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            if zero_gradients:
                (param.sum() * 0).backward()
            opt.step()
        torch.testing.assert_close(param, torch.ones_like(param))

    def test_adamw8bit_accumulation(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for AdamW8bit')
        import bitsandbytes as bnb

        param = nn.Parameter(torch.ones(8192, device='cuda'))
        reference = nn.Parameter(param.detach().clone())
        opt = alf.optimizers.AdamW8bit(lr=0.01, grad_accumulation_steps=3,
                                       gradient_clipping=0.5,
                                       clip_by_global_norm=True)
        opt.add_param_group({'params': [param]})
        reference_opt = bnb.optim.AdamW8bit([reference], lr=0.01)
        for cycle in range(2):
            expected_grad = torch.zeros_like(reference)
            for microstep in range(3):
                opt.zero_grad(set_to_none=True)
                grad = torch.linspace(-1, 2, param.numel(), device='cuda')
                grad.mul_(cycle + microstep + 1)
                (param * grad).sum().backward()
                expected_grad.add_(grad)
                opt.step()
            reference.grad = expected_grad / 3
            tensor_utils.clip_by_global_norm(
                [reference.grad], 0.5, in_place=True)
            reference_opt.step()
            torch.testing.assert_close(param, reference, rtol=1e-4, atol=1e-5)

    def test_sparse_accumulation_keeps_existing_path(self):
        param = nn.Parameter(torch.ones(3, device='cpu'))
        opt = alf.optimizers.SGD(lr=0.1, grad_accumulation_steps=2,
                                 foreach=False)
        opt.add_param_group({'params': [param]})
        with mock.patch('torch._foreach_mul_') as mul:
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                grad = torch.sparse_coo_tensor([[1]], [2.], (3,), device='cpu')
                param.grad = grad if param.grad is None else param.grad + grad
                opt.step()
            mul.assert_not_called()
        torch.testing.assert_close(param, torch.tensor([1., 0.8, 1.], device='cpu'))

    def test_dtensor_keeps_existing_path(self):
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group(
                'gloo', init_method='file://' + os.path.join(directory, 'init'),
                rank=0, world_size=1)
            try:
                mesh = init_device_mesh('cpu', (1,))
                param = nn.Parameter(DTensor.from_local(
                    torch.ones(2, device='cpu'), mesh, [Shard(0)]))
                opt = alf.optimizers.SGD(lr=0.1, grad_accumulation_steps=2,
                                         gradient_clipping=0.5,
                                         clip_by_global_norm=True, foreach=False)
                opt.add_param_group({'params': [param]})
                with mock.patch('torch.nn.utils.clip_grad_norm_') as clip, \
                        mock.patch('torch._foreach_mul_') as mul:
                    for _ in range(2):
                        opt.zero_grad(set_to_none=True)
                        param.sum().backward()
                        opt.step()
                    clip.assert_not_called()
                    mul.assert_not_called()
                expected = torch.full((2,), 1 - 0.1 * 0.5 / 2**0.5,
                                      device='cpu')
                torch.testing.assert_close(param.full_tensor(), expected)
            finally:
                dist.destroy_process_group()


class LRBatchEnsemble(nn.Module):
    """A simple linear regressor with the parameter vector being a batch
    ensemble, used for parvi_batch_ensemble test.
    """

    def __init__(self, input_size, ensemble_size, ensemble_group=0):
        """
        Args:
            input_size (int): input size
            ensemble_size (int): ensemble size
            ensemble_group (int): the extra attribute ``ensemble_group`` added
                to ``self._r`` and ``self._s``, default value is 0.
                For alf.optimizers whose ``parvi`` is not ``None``,
                all parameters with the same ``ensemble_group`` will be updated
                by the ``SVGD`` or ``GFSF`` particle-based VI algorithm.
        """
        nn.Module.__init__(self)
        self._beta = nn.Parameter(torch.rand(ensemble_size, input_size))
        assert isinstance(ensemble_group,
                          int), ("ensemble_group has to be an integer!")
        self._beta.ensemble_group = ensemble_group
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
        opt = AdamTF(lr=0.1,
                     gradient_clipping=clip_norm,
                     clip_by_global_norm=True)
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

    @parameterized.parameters('svgd', 'gfsf')
    def test_parvi_grad_step(self, parvi='svgd'):
        """Check consistency of one step grad update with ParVIAlgorithm. """
        param_dim = 3
        ensemble_size = 4
        batch_size = 2
        init_w = torch.rand(ensemble_size, param_dim)
        inputs = torch.randn(batch_size, param_dim)

        def loss_func(w):
            y = w @ inputs.t()
            return torch.sum(y**2)

        # Gradient update with parvi_optimizer
        w1 = init_w.clone()
        w1.requires_grad = True
        w1 = torch.nn.Parameter(w1)
        w1.ensemble_group = 0
        loss = loss_func(w1)
        opt = AdamTF(lr=0.1, parvi=parvi)
        opt.add_param_group({'params': w1})
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Gradient update with ParVIAlgorithm
        alg = ParVIAlgorithm(param_dim,
                             num_particles=ensemble_size,
                             par_vi=parvi,
                             optimizer=AdamTF(lr=0.1))
        w2 = init_w.clone()
        w2.requires_grad = True
        w2 = torch.nn.Parameter(w2)
        alg._particles = torch.nn.Parameter(w2)
        alg_step = alg.train_step(loss_func)
        alg.update_with_gradient(alg_step.info)

        self.assertTensorClose(w1.data, alg.particles.data, 1e-4)

    @parameterized.parameters('svgd', 'gfsf')
    def test_parvi_batch_ensemble(self,
                                  parvi='svgd',
                                  train_batch_size=10,
                                  num_particles=32):
        r"""
        Bayesian linear regression test for a linear regressor with BatchEnsemble
        parameter vector and trained with ``SVGD`` or ``GFSF`` optimizer.
        The target linear regressor is :math:`y = X\beta + e`, where
        :math:`e\sim N(0, I)` is random noise, :math:`X` is the input data matrix,
        and :math:`y` is target outputs. The posterior of :math:`\beta` has a
        closed-form :math:`p(\beta|X,y)\sim N((X^TX)^{-1}X^Ty, X^TX)`.
        """
        input_size = 3
        input_spec = TensorSpec((input_size, ), torch.float32)
        output_size = 1
        batch_size = 100
        inputs = input_spec.randn(outer_dims=(batch_size, ))
        beta = torch.rand(input_size, output_size) + 5.
        print("beta: {}".format(beta))
        noise = torch.randn(batch_size, output_size)
        targets = inputs @ beta + noise
        true_cov = torch.inverse(inputs.t() @ inputs)
        true_mean = true_cov @ inputs.t() @ targets
        print("ground truth mean: {}".format(true_mean))
        print("ground truth cov: {}".format(true_cov))
        print("ground truth cov norm: {}".format(true_cov.norm()))

        layer = LRBatchEnsemble(input_size, num_particles)
        entropy_regularization = train_batch_size / batch_size
        optimizer = alf.optimizers.Adam(
            lr=1e-2, parvi=parvi, repulsive_weight=entropy_regularization)
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

    @parameterized.parameters(
        dict(opt_cls=Adam,
             capacity_ratio=0.2,
             masked_out_value=None,
             opt_steps=3),
        dict(opt_cls=AdamTF,
             capacity_ratio=0.7,
             masked_out_value=0,
             opt_steps=5))
    def test_capacity_scheduling(self, opt_cls, capacity_ratio,
                                 masked_out_value, opt_steps):
        layer = torch.nn.Linear(512, 512)
        clip_norm = 1e-4
        opt = opt_cls(lr=0.1,
                      gradient_clipping=clip_norm,
                      clip_by_global_norm=True,
                      capacity_ratio=capacity_ratio,
                      masked_out_value=masked_out_value,
                      min_capacity=1)
        opt.add_param_group({'params': layer.parameters()})

        # test load_state_dict
        opt2 = opt_cls(lr=0.1,
                       gradient_clipping=clip_norm,
                       clip_by_global_norm=True,
                       capacity_ratio=capacity_ratio,
                       masked_out_value=masked_out_value,
                       min_capacity=1)
        opt2.add_param_group({'params': layer.parameters()})
        state_dict = opt.state_dict()
        opt2.load_state_dict(state_dict)

        def _train_step():
            x = torch.randn(2, 512)
            y = layer(x)
            loss = torch.sum(y**2)
            return loss

        def _infer_capacity_mask_from_params_pairs(param_groups_before,
                                                   param_groups_after):
            """infer the capacity mask as the parameter elements with unchanged values
            in the provided before and after pairs.
            """
            capacity_mask = []
            for pg_before, pg_after in zip(param_groups_before,
                                           param_groups_after):
                for p_before, p_after in zip(pg_before['params'],
                                             pg_after['params']):
                    capacity_mask.append(p_before == p_after)
            return capacity_mask

        opt.zero_grad()
        loss = _train_step()
        param_groups_before = copy.deepcopy(opt.param_groups)
        loss.backward()

        # perform one opt step in order to infer capacity mask from parameters
        opt.step()
        initial_capacity_mask = _infer_capacity_mask_from_params_pairs(
            param_groups_before, opt.param_groups)

        for _ in range(opt_steps):
            loss = _train_step()
            loss.backward()
            opt.step()

        capacity_mask = _infer_capacity_mask_from_params_pairs(
            param_groups_before, opt.param_groups)

        # 1) check that the capacity mask remains unchanged across opt steps when
        # the specified capacity is unchanged
        alf.nest.map_structure(self.assertTensorEqual, initial_capacity_mask,
                               capacity_mask)

        # 2) check the empirical capacity matches the expected capacity
        capacity_ratios = alf.nest.map_structure(
            lambda x: 1 - x.float().mean(), capacity_mask)
        empirical_capacity_ratio = sum(capacity_ratios) / len(capacity_ratios)
        self.assertAlmostEqual(empirical_capacity_ratio,
                               capacity_ratio,
                               delta=0.1)


if __name__ == "__main__":
    alf.test.main()
