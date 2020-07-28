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
"""A generic generator."""

import gin
import numpy as np
import torch

from alf.algorithms.algorithm import Algorithm
from alf.algorithms.mi_estimator import MIEstimator
from alf.data_structures import AlgStep, LossInfo, namedtuple
import alf.nest as nest
from alf.networks import Network, EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops
from alf.utils.averager import AdaptiveAverager

GeneratorLossInfo = namedtuple("GeneratorLossInfo",
                               ["generator", "mi_estimator"])


@gin.configurable
class Generator(Algorithm):
    """Generator

    Generator generates outputs given `inputs` (can be None) by transforming
    a random noise and input using `net`:

        outputs = net([noise, input]) if input is not None
                  else net(noise)

    The generator is trained to minimize the following objective:

        :math:`E(loss_func(net([noise, input]))) - entropy_regulariztion \cdot H(P)`

    where P is the (conditional) distribution of outputs given the inputs
    implied by `net` and H(P) is the (conditional) entropy of P.

    If the loss is the (unnormalized) negative log probability of some
    distribution Q and the entropy_regularization is 1, this objective is
    equivalent to minimizing :math:`KL(P||Q)`.

    It uses two different ways to optimize `net` depending on
    entropy_regularization:

    * entropy_regularization = 0: the minimization is achieved by simply
      minimizing loss_func(net([noise, inputs]))

    * entropy_regularization > 0: the minimization is achieved using amortized
      Stein variational gradient descent (SVGD). See the following paper for
      reference:

      Feng et al "Learning to Draw Samples with Amortized Stein Variational
      Gradient Descent" https://arxiv.org/pdf/1707.06626.pdf

    It also supports an additional optional objective of maximizing the mutual
    information between [noise, inputs] and outputs by using mi_estimator to
    prevent mode collapse. This might be useful for entropy_regulariztion = 0
    as suggested in section 5.1 of the following paper:

    Hjelm et al "Learning Deep Representations by Mutual Information Estimation
    and Maximization" https://arxiv.org/pdf/1808.06670.pdf
    """

    def __init__(self,
                 output_dim,
                 noise_dim=32,
                 input_tensor_spec=None,
                 hidden_layers=(256, ),
                 net: Network = None,
                 net_moving_average_rate=None,
                 entropy_regularization=0.,
                 kernel_sharpness=2.,
                 mi_weight=None,
                 mi_estimator_cls=MIEstimator,
                 par_vi="gfsf",
                 optimizer=None,
                 name="Generator"):
        r"""Create a Generator.

        Args:
            output_dim (int): dimension of output
            noise_dim (int): dimension of noise
            input_tensor_spec (nested TensorSpec): spec of inputs. If there is
                no inputs, this should be None.
            hidden_layers (tuple): size of hidden layers.
            net (Network): network for generating outputs from [noise, inputs]
                or noise (if inputs is None). If None, a default one with
                hidden_layers will be created
            net_moving_average_rate (float): If provided, use a moving average
                version of net to do prediction. This has been shown to be
                effective for GAN training (arXiv:1907.02544, arXiv:1812.04948).
            entropy_regularization (float): weight of entropy regularization
            kernel_sharpness (float): Used only for entropy_regularization > 0.
                We calcualte the kernel in SVGD as:
                    :math:`\exp(-kernel_sharpness * reduce_mean(\frac{(x-y)^2}{width}))`
                where width is the elementwise moving average of :math:`(x-y)^2`
            mi_estimator_cls (type): the class of mutual information estimator
                for maximizing the mutual information between [noise, inputs]
                and [outputs, inputs].
            optimizer (torch.optim.Optimizer): (optional) optimizer for training
            name (str): name of this generator
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)
        self._noise_dim = noise_dim
        self._entropy_regularization = entropy_regularization
        if entropy_regularization == 0:
            self._grad_func = self._ml_grad
        else:
            if par_vi == 'gfsf':
                self._grad_func = self._gfsf_grad
            elif par_vi == 'svgd':
                self._grad_func = self._svgd_grad
            elif par_vi == 'svgd2':
                self._grad_func = self._svgd_grad2
            else:
                assert ValueError(
                    "par_vi only supports \"gfsf\", \"svgd\", and \"svgd2\"")

            self._kernel_width_averager = AdaptiveAverager(
                tensor_spec=TensorSpec(shape=(output_dim, )))
            self._kernel_sharpness = kernel_sharpness

        noise_spec = TensorSpec(shape=(noise_dim, ))

        if net is None:
            net_input_spec = noise_spec
            if input_tensor_spec is not None:
                net_input_spec = [net_input_spec, input_tensor_spec]
            net = EncodingNetwork(
                input_tensor_spec=net_input_spec,
                fc_layer_params=hidden_layers,
                last_layer_size=output_dim,
                last_activation=math_ops.identity,
                name="Generator")

        self._mi_estimator = None
        self._input_tensor_spec = input_tensor_spec
        if mi_weight is not None:
            x_spec = noise_spec
            y_spec = TensorSpec((output_dim, ))
            if input_tensor_spec is not None:
                x_spec = [x_spec, input_tensor_spec]
            self._mi_estimator = mi_estimator_cls(
                x_spec, y_spec, sampler='shift')
            self._mi_weight = mi_weight
        self._net = net
        self._predict_net = None
        self._net_moving_average_rate = net_moving_average_rate
        if net_moving_average_rate:
            self._predict_net = net.copy(name="Genrator_average")
            self._predict_net_updater = common.get_target_updater(
                self._net, self._predict_net, tau=net_moving_average_rate)

    def _trainable_attributes_to_ignore(self):
        return ["_predict_net"]

    @property
    def noise_dim(self):
        return self._noise_dim

    def _predict(self, inputs, noise=None, batch_size=None, training=True):
        if inputs is None:
            assert self._input_tensor_spec is None
            if noise is None:
                assert batch_size is not None
                noise = torch.randn(batch_size, self._noise_dim)
            gen_inputs = noise
        else:
            nest.assert_same_structure(inputs, self._input_tensor_spec)
            batch_size = nest.get_nest_batch_size(inputs)
            if noise is None:
                noise = torch.randn(batch_size, self._noise_dim)
            else:
                assert noise.shape[0] == batch_size
                assert noise.shape[1] == self._noise_dim
            gen_inputs = [noise, inputs]
        if self._predict_net and not training:
            outputs = self._predict_net(gen_inputs)[0]
        else:
            outputs = self._net(gen_inputs)[0]
        return outputs, gen_inputs

    def predict_step(self, inputs, batch_size=None, state=None):
        """Generate outputs given inputs.

        Args:
            inputs (nested Tensor): if None, the outputs is generated only from
                noise.
            batch_size (int): batch_size. Must be provided if inputs is None.
                Its is ignored if inputs is not None
            state: not used
        Returns:
            AlgorithmStep: outputs with shape (batch_size, output_dim)
        """
        outputs, _ = self._predict(
            inputs, batch_size=batch_size, training=False)
        return AlgStep(output=outputs, state=(), info=())

    def train_step(self, inputs, loss_func, batch_size=None, state=None):
        """
        Args:
            inputs (nested Tensor): if None, the outputs is generated only from
                noise.
            loss_func (Callable): loss_func([outputs, inputs])
                (loss_func(outputs) if inputs is None) returns a Tensor with
                shape [batch_size] as a loss for optimizing the generator
            batch_size (int): batch_size. Must be provided if inputs is None.
                Its is ignored if inputs is not None
            state: not used
        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        outputs, gen_inputs = self._predict(inputs, batch_size=batch_size)
        loss, grad = self._grad_func(inputs, outputs, loss_func)
        loss_propagated = torch.sum(grad.detach() * outputs, dim=-1)

        mi_loss = ()
        if self._mi_estimator is not None:
            mi_step = self._mi_estimator.train_step([gen_inputs, outputs])
            mi_loss = mi_step.info.loss
            loss_propagated = loss_propagated + self._mi_weight * mi_loss

        return AlgStep(
            output=outputs,
            state=(),
            info=LossInfo(
                loss=loss_propagated,
                extra=GeneratorLossInfo(generator=loss, mi_estimator=mi_loss)))

    def _ml_grad(self, inputs, outputs, loss_func):
        loss_inputs = outputs if inputs is None else [outputs, inputs]
        loss = loss_func(loss_inputs)

        grad = torch.autograd.grad(loss.sum(), outputs)[0]

        return loss, grad

    def _median_width(self, mat_dist):
        """Compute the kernel width from median of the distance matrix."""

        values, _ = torch.topk(
            mat_dist.view(-1), k=mat_dist.nelement() // 2 + 1)
        median = values[-1]
        return median / np.log(mat_dist.shape[0])

    def _kernel_func(self, x, y):
        d = (x - y)**2
        self._kernel_width_averager.update(torch.mean(d, dim=0))
        d = torch.sum(d / self._kernel_width_averager.get(), dim=-1)
        w = torch.exp(-self._kernel_sharpness * d)
        return w

    def _rbf_func(self, x, y, h_min=1e-3):
        r"""Compute the rbf kernel and its gradient w.r.t. first entry 
            :math:`K(x, y), \nabla_x K(x, y)`

        Args:
            x (Tensor): set of N particles, shape (Nx x W), where W is the 
                dimenseion of each particle
            y (Tensor): set of N particles, shape (Ny x W), where W is the 
                dimenseion of each particle
            h_min (float): minimum kernel bandwidth

        Returns:
            :math:`K(x, y)` (Tensor): the RBF kernel of shape (Nx x Ny)
            :math:`\nabla_x K(x, y)` (Tensor): the derivative of RBF kernel of shape (Nx x Ny x D)
            
        """
        Nx, Dx = x.shape
        Ny, Dy = y.shape
        assert Dx == Dy
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, W]
        dist_sq = torch.sum(diff**2, -1)  # [Nx, Ny]
        # h = self._median_width(dist_sq)
        # h = torch.max(h, torch.as_tensor([h_min]))
        h = 2

        kappa = torch.exp(-dist_sq / h)  # [Nx, Nx]
        kappa_grad = torch.einsum('ij,ijk->ijk', kappa,
                                  -2 * diff / h)  # [Nx, Ny, W]
        return kappa, kappa_grad

    def _score_func(self, x, alpha=1e-7, h_min=1e-3):
        r"""Compute the stein estimator of the score function 
            :math:`\nabla\log q = -(K + \alpha I)^{-1}\nabla K`

        Args:
            x (Tensor): set of N particles, shape (N x D), where D is the 
                dimenseion of each particle
            alpha (float): weight of regularization for inverse kernel
                this parameter turns out to be crucial for convergence.

        Returns:
            :math:`\nabla\log q` (Tensor): the score function of shape (N x D)
            
        """
        N, D = x.shape
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, D]
        dist_sq = torch.sum(diff**2, -1)  # [N, N]

        # diff = diff**2
        # self._kernel_width_averager.update(diff.mean(dim=0).mean(dim=0))
        # h = self._kernel_width_averager.get()
        # dist = diff / h
        # dist_sq = torch.sum(dist, dim=-1)
        # kappa = torch.exp(-self._kernel_sharpness * dist_sq)

        # compute the kernel width
        h = self._median_width(dist_sq)
        h = torch.max(h, torch.as_tensor([h_min]))
        # h =.1
        kappa = torch.exp(-dist_sq / h)  # [N, N]

        kappa_inv = torch.inverse(kappa + alpha * torch.eye(N))  # [N, N]
        kappa_grad = torch.einsum('ij,ijk->jk', kappa, -2 * diff / h)  # [N, D]
        # kappa_grad = torch.einsum(
        #     'ij,ijk->jk', kappa, -2 * self._kernel_sharpness * dist)  # [N, D]

        return kappa_inv @ kappa_grad

    def _score_func2(self, x, y, alpha=1e-8, h_min=1e-3):
        r"""Compute the stein estimator of the score function 
            :math:`\nabla\log q = -(K + \alpha I)^{-1}\nabla K`

        Args:
            x (Tensor): set of N particles, shape (N x D), where D is the 
                dimenseion of each particle
            alpha (float): weight of regularization for inverse kernel

        Returns:
            :math:`\nabla\log q` (Tensor): the score function of shape (N x D)
            
        """
        Nx, D = x.shape
        Ny, Dy = y.shape
        assert D == Dy
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, D]

        d = (x - y)**2
        self._kernel_width_averager.update(torch.mean(d, dim=0))
        h = self._kernel_width_averager.get()
        dist = diff**2 / h
        dist_sq = torch.sum(dist, dim=-1)
        kappa = torch.exp(-self._kernel_sharpness * dist_sq)  # [Nx, Ny]

        # compute the kernel width
        # h = self._median_width(dist_sq)
        # h = torch.max(h, torch.as_tensor([h_min]))
        # h =.1

        kappa_inv = torch.inverse(kappa + alpha * torch.eye(Nx))  # [Ny, Nx]
        # kappa_grad = torch.einsum('ij,ijk->jk', kappa, -2 * diff / h)  # [N, D]
        kappa_grad = torch.einsum(
            'ij,ijk->jk', kappa, -2 * self._kernel_sharpness * dist)  # [Ny, D]

        return kappa_inv.t() @ kappa_grad

    def _svgd_grad(self, inputs, outputs, loss_func):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by a single resampled particle. 
        """
        outputs2, _ = self._predict(inputs, batch_size=outputs.shape[0])
        kernel_weight = self._kernel_func(outputs, outputs2)
        weight_sum = self._entropy_regularization * kernel_weight.sum()

        kernel_grad = torch.autograd.grad(weight_sum, outputs2)[0]

        loss_inputs = outputs2 if inputs is None else [outputs2, inputs]
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        weighted_loss = kernel_weight.detach() * neglogp

        loss_grad = torch.autograd.grad(weighted_loss.sum(), outputs2)[0]

        return loss, loss_grad - kernel_grad

    def _svgd_grad2(self, inputs, outputs, loss_func):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by resampled particles of the same batch size. 
        """
        particles = outputs.shape[0]
        outputs2, _ = self._predict(inputs, batch_size=particles)
        # [N2, N], [N2, N, D]

        loss_inputs = outputs2 if inputs is None else [outputs2, inputs]
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), outputs2)[0]  # [N2, D]

        kernel_weight, kernel_grad = self._rbf_func(outputs2, outputs)
        kernel_logp = torch.matmul(kernel_weight.t(),
                                   loss_grad) / particles  # [N, D]

        return loss, kernel_logp - kernel_grad.mean(0)

    def _gfsf_grad(self, inputs, outputs, loss_func):
        """Compute particle gradients via GFSF (Stein estimator). """
        # particles = outputs.shape[0]
        # outputs2, _ = self._predict(inputs, batch_size=particles)

        loss_inputs = outputs if inputs is None else [outputs, inputs]
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), outputs)[0]  # [N2, D]

        logq_grad = self._score_func(outputs)

        return loss, loss_grad - logq_grad

    def after_update(self, training_info):
        if self._predict_net:
            self._predict_net_updater()
