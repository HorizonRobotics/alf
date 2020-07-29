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

from absl import logging
import functools
import gin
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable

import alf
from alf.data_structures import AlgStep, LossInfo, namedtuple
from alf.algorithms.generator import Generator
from alf.algorithms.hypernetwork_networks import ParamNetwork
from alf.networks import EncodingNetwork
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops
from alf.utils.summary_utils import record_time

HyperNetworkLossInfo = namedtuple("HyperNetworkLossInfo", ["loss", "extra"])


@gin.configurable
class HyperNetwork(Algorithm):
    """HyperNetwork 

    HyperrNetwork algorithm maintains a generator that generates a set of 
    parameters for a predefined neural network from a random noise input. 
    It is based on the following work:

    https://github.com/neale/HyperGAN

    Ratzlaff and Fuxin. "HyperGAN: A Generative Model for Diverse, 
    Performant Neural Networks." International Conference on Machine Learning. 2019.

    Major differences versus the original paper are:

    * A single genrator that generates parameters for all network layers.

    * Remove the mixer and the distriminator.

    * The generator is trained with Amortized particle-based variational 
      inference (ParVI) methods, please refer to generator.py for details.

    """

    def __init__(self,
                 input_tensor_spec,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 activation=torch.relu_,
                 last_layer_param=None,
                 last_activation=None,
                 noise_dim=32,
                 hidden_layers=(64, 64),
                 use_fc_bn=False,
                 particles=32,
                 entropy_regularization=1.,
                 kernel_sharpness=1.,
                 loss_type="classification",
                 loss_func: Callable = None,
                 voting="soft",
                 par_vi="gfsf",
                 optimizer=None,
                 logging_network=False,
                 name="HyperNetwork"):
        """
        Args:
            Args for the generated parametric network
            ====================================================================
            input_tensor_spec (nested TensorSpec): the (nested) tensor spec of
                the input. If nested, then ``preprocessing_combiner`` must not be
                None.
            conv_layer_params (tuple[tuple]): a tuple of tuples where each
                tuple takes a format 
                ``(filters, kernel_size, strides, padding, pooling_kernel)``,
                where ``padding`` and ``pooling_kernel`` are optional.
            fc_layer_params (tuple[tuple]): a tuple of tuples where each tuple
                takes a format ``(FC layer sizes. use_bias)``, where 
                ``use_bias`` is optional.
            activation (nn.functional): activation used for all the layers but
                the last layer.
            last_layer_param (tuple): an optional tuple of the format
                ``(size, use_bias)``, where ``use_bias`` is optional,
                it appends an additional layer at the very end. 
                Note that if ``last_activation`` is specified, 
                ``last_layer_param`` has to be specified explicitly.
            last_activation (nn.functional): activation function of the
                additional layer specified by ``last_layer_param``. Note that if
                ``last_layer_param`` is not None, ``last_activation`` has to be
                specified explicitly.

            Args for the generator
            ====================================================================
            noise_dim (int): dimension of noise
            hidden_layers (tuple): size of hidden layers.
            use_fc_bn (bool): whether use batnch normalization for fc layers.
            particles (int): number of sampling particles
            entropy_regularization (float): weight of entropy regularization
            kernel_sharpness (float): Used only for entropy_regularization > 0.
                We calcualte the kernel in SVGD as:
                    :math:`\exp(-kernel_sharpness * reduce_mean(\frac{(x-y)^2}{width}))`
                where width is the elementwise moving average of :math:`(x-y)^2`

            Args for training and testing
            ====================================================================
            loss_type (str): loglikelihood type for the generated functions,
                types are [``classification``, ``regression``]
            loss_func (Callable): loss_func(outputs, targets)   
            voting (str): types of voting results from sampled functions,
                types are [``soft``, ``hard``]
            optimizer (torch.optim.Optimizer): The optimizer for training.
            logging_network (bool): whether logging the archetectures of networks.
            name (str):
        """
        param_net = ParamNetwork(
            input_tensor_spec=input_tensor_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation=activation,
            last_layer_param=last_layer_param,
            last_activation=last_activation)

        gen_output_dim = param_net.param_length
        noise_spec = TensorSpec(shape=(noise_dim, ))
        net = EncodingNetwork(
            noise_spec,
            fc_layer_params=hidden_layers,
            use_fc_bn=use_fc_bn,
            last_layer_size=gen_output_dim,
            last_activation=math_ops.identity,
            name="Generator")

        if logging_network:
            logging.info("Generated network")
            logging.info("-" * 68)
            logging.info(param_net)

            logging_info("Generator network")
            logging_info("-" * 68)
            logging_info(net)

        self._generator = Generator(
            gen_output_dim,
            noise_dim=noise_dim,
            net=net,
            entropy_regularization=entropy_regularization,
            kernel_sharpness=kernel_sharpness,
            par_vi=par_vi,
            optimizer=optimizer,
            name=name)

        self._param_net = param_net
        self._particles = particles
        self._train_loader = None
        self._test_loader = None
        self._loss_func = loss_func
        self._par_vi = par_vi
        self._use_fc_bn = use_fc_bn
        self._loss_type = loss_type
        assert (voting in ['soft', 'hard'
                           ]), ("voting only supports \"soft\" and \"hard\"")
        self._voting = voting
        if loss_type == 'classification':
            self._compute_loss = self._classification_loss
            self._vote = self._classification_vote
            if self._loss_func is None:
                self._loss_func = F.cross_entropy
        elif loss_type == 'regression':
            self._compute_loss = self._regression_loss
            self._vote = self._regression_vote
            if self._loss_func is None:
                self._loss_func = functools.partial(
                    F.mse_loss, reduction='sum')
        else:
            assert ValueError(
                "loss_type only supports \"classification\" and \"regression\""
            )
        if par_vi == 'gfsf':
            self._grad_func = self._stein_grad
        elif par_vi == 'svgd':
            self._grad_func = self._svgd_grad
        else:
            assert ValueError("par_vi only supports \"gfsf\" and \"svgd\"")

    def set_data_loader(self, train_loader, test_loader=None):
        """Set data loadder for training and testing."""
        self._train_loader = train_loader
        self._test_loader = test_loader

    def set_particles(self, particles):
        """Set the number of particles to sample through one forward
        pass of the hypernetwork. """
        self._particles = particles

    @property
    def particles(self):
        return self._particles

    def sample_parameters(self, noise=None, particles=None, training=True):
        "Sample parameters for an ensemble of networks." ""
        if noise is None and particles is None:
            particles = self.particles
        generator_step = self._generator.predict_step(
            noise=noise, batch_size=particles, training=training)
        return generator_step.output

    def predict_step(self, inputs, params=None, particles=None, state=None):
        """Predict ensemble outputs for inputs using the hypernetwork model.
        
        Args:
            inputs (Tensor): inputs to the ensemble of networks.
            params (Tensor): parameters of the ensemble of networks,
                if None, will resample.
            particles (int): size of sampled ensemble.
            state: not used.

        Returns:
            AlgorithmStep: outputs with shape (batch_size, output_dim)
        """
        if params is None:
            params = self.sample_parameters(particles=particles)
        self._param_net.set_parameters(params)
        outputs, _ = self._param_net(inputs)
        return AlgStep(output=outputs, state=(), info=())

    def train_iter(self, particles=None, loss_func=None, state=None):
        """Perform one epoch (iteration) of training."""

        assert self._train_loader is not None, "Must set data_loader first."
        with record_time("time/train"):
            loss = 0.
            if self._loss_type == 'classification':
                avg_acc = []
            params = None
            # if not self._regenerate_for_each_batch:
            #     params = self.sample_parameters(particles=particles)
            for batch_idx, (data, target) in enumerate(self._train_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                alg_step = self.train_step((data, target),
                                           params=params,
                                           particles=particles,
                                           loss_func=loss_func,
                                           state=state)
                self.update_with_gradient(alg_step.info)
                loss += alg_step.info.extra.loss
                if self._loss_type == 'classification':
                    avg_acc.append(alg_step.info.extra.extra)
        if self._loss_type == 'classification':
            acc = torch.as_tensor(avg_acc)
            logging.info("Avg acc: {}".format(acc.mean() * 100))
        logging.info("Cum loss: {}".format(loss))

        return batch_idx + 1

    def train_step(self,
                   inputs,
                   params=None,
                   loss_func=None,
                   particles=None,
                   state=None):
        """Perform one batch of training computation.

        Args:
            inputs (nested Tensor): input training data. 
            params (Tensor): sampled parameter for param_net, if None,
                will re-sample.
            loss_func (Callable): loss_func([outputs, inputs])
                (loss_func(outputs) if inputs is None) returns a Tensor with
                shape [batch_size] as a loss for optimizing the generator
            particles (int): number of sampled particles. 
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        # def _neglogprob(inputs, params)
        if loss_func is None:
            loss_func = self._loss_func
        # if self._regenerate_for_each_batch:
        params = self.sample_parameters(particles=particles)

        train_info, loss_propagated = self._grad_func(inputs, params,
                                                      loss_func)
        return AlgStep(
            output=params,
            state=(),
            info=LossInfo(loss=loss_propagated, extra=train_info))

    def _stein_grad(self, inputs, params, loss_func):
        """Compute particle gradients via gfsf (stein estimator). """
        data, target = inputs
        particles = params.shape[0]
        self._param_net.set_parameters(params)
        output, _ = self._param_net(data)  # [B, N, D]
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        loss, extra = self._compute_loss(output, target, loss_func)

        loss_grad = torch.autograd.grad(loss.sum(), params)[0]
        logq_grad = self._score_func(params)
        grad = loss_grad - logq_grad
        # grad = loss_grad

        train_info = HyperNetworkLossInfo(loss=loss, extra=extra)
        loss_propagated = torch.sum(grad.detach() * params, dim=-1)

        return train_info, loss_propagated

    def _svgd_grad(self, inputs, params, loss_func):
        """Compute particle gradients via svgd. """
        data, target = inputs
        particles = params.shape[0] // 2
        params_i, params_j = torch.split(params, particles, dim=0)
        self._param_net.set_parameters(params_j)
        output, _ = self._param_net(data)  # [B, N/2, D]
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        loss, extra = self._compute_loss(output, target, loss_func)

        loss_grad = torch.autograd.grad(loss.sum(), params_j)[0]  # [Nj, W]
        # q_i = params_i + torch.rand_like(params_i) * 1e-8
        # q_j = params_j + torch.rand_like(params_j) * 1e-8
        # kappa, kappa_grad = self._rbf_func(q_j, q_i)  # [Nj, Ni], [Nj, Ni, W]
        kappa, kappa_grad = self._rbf_func(params_j,
                                           params_i)  # [Nj, Ni], [Nj, Ni, W]
        Nj = kappa.shape[0]
        kernel_logp = torch.matmul(kappa.t(), loss_grad) / Nj
        grad = (kernel_logp - kappa_grad.mean(0))  # [Ni, W]

        train_info = HyperNetworkLossInfo(loss=loss, extra=extra)
        loss_propagated = torch.sum(grad.detach() * params_i, dim=-1)

        return train_info, loss_propagated

    def _classification_loss(self, output, target, loss_func):
        pred = output.max(-1)[1]
        acc = pred.eq(target).float().mean(0)
        avg_acc = acc.mean()
        loss = loss_func(output.transpose(1, 2), target)
        return loss, avg_acc

    def _regression_loss(self, output, target, loss_func):
        out_shape = output.shape[-1]
        assert (target.shape[-1] == out_shape), (
            "feature dimension of output and target does not match.")
        loss = loss_func(
            output.reshape(-1, out_shape), target.reshape(-1, out_shape))
        return loss, ()

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
        h = self._median_width(dist_sq)
        h = torch.max(h, torch.as_tensor([h_min]))

        kappa = torch.exp(-dist_sq / h)  # [Nx, Nx]
        kappa_grad = torch.einsum('ij,ijk->ijk', kappa,
                                  -2 * diff / h)  # [Nx, Ny, W]

        return kappa, kappa_grad

    def _score_func(self, x, alpha=1e-6, h_min=1e-3):
        r"""Compute the stein estimator of the score function 
            :math:`\nabla\log q = -(K + \alpha I)^{-1}\nabla K`

        Args:
            x (Tensor): set of N particles, shape (N x D), where D is the 
                dimenseion of each particle
            alpha (float): weight of regularization for inverse kernel

        Returns:
            :math:`\nabla\log q` (Tensor): the score function of shape (N x D)
            
        """
        N, D = x.shape
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # [N, N, D]
        dist_sq = torch.sum(diff**2, -1)  # [N, N]

        # compute the kernel width
        h = self._median_width(dist_sq)
        h = torch.max(h, torch.as_tensor([h_min]))
        # h =.1

        kappa = torch.exp(-dist_sq / h)  # [N, N]
        kappa_inv = torch.inverse(kappa + alpha * torch.eye(N))  # [N, N]
        kappa_grad = torch.einsum('ij,ijk->jk', kappa, -2 * diff / h)  # [N, D]

        return kappa_inv @ kappa_grad

    def evaluate(self, loss_func=None, particles=None):
        """Evaluate on a randomly drawn network. """

        assert self._test_loader is not None, "Must set test_loader first."
        if loss_func is None:
            loss_func = self._loss_func
        if self._use_fc_bn:
            self._generator.eval()
        params = self.sample_parameters(particles=particles)
        self._param_net.set_parameters(params)
        if self._use_fc_bn:
            self._generator.train()
        with record_time("time/test"):
            if self._loss_type == 'classification':
                test_acc = 0.
            test_loss = 0.
            for i, (data, target) in enumerate(self._test_loader):
                data = data.to(alf.get_default_device())
                target = target.to(alf.get_default_device())
                output, _ = self._param_net(data)  # [B, N, D]
                loss, extra = self._vote(output, target, loss_func)
                if self._loss_type == 'classification':
                    test_acc += extra.item()
                test_loss += loss.item()

        if self._loss_type == 'classification':
            test_acc /= len(self._test_loader.dataset)
            logging.info("Test acc: {}".format(test_acc * 100))
        logging.info("Test loss: {}".format(test_loss))

    def _classification_vote(self, output, target, loss_func):
        """ensmeble the ooutputs from sampled classifiers."""
        particles = output.shape[1]
        probs = F.softmax(output, dim=-1)  # [B, N, D]
        if self._voting == 'soft':
            pred = probs.mean(1).cpu()  # [B, D]
            vote = pred.argmax(-1)
        elif self._voting == 'hard':
            pred = probs.argmax(-1).cpu()  # [B, N, 1]
            vote = pred.mode(1)[0]  # [B, 1]
        correct = vote.eq(target.cpu().view_as(vote)).float().cpu().sum()
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        loss = loss_func(output.transpose(1, 2), target)
        return loss, correct

    def _regression_vote(self, output, target, loss_func):
        """ensemble the outputs for sampled regressors."""
        particles = output.shape[1]
        pred = output.mean(1)  # [B, D]
        loss = loss_func(pred, target)
        target = target.unsqueeze(1).expand(*target.shape[:1], particles,
                                            *target.shape[1:])
        total_loss = loss_func(output, target)
        return loss, total_loss

    def summarize_train(self, experience, train_info, loss_info, params):
        """Generate summaries for training & loss info after each gradient update.
        The default implementation of this function only summarizes params
        (with grads) and the loss. An algorithm can override this for additional
        summaries. See ``RLAlgorithm.summarize_train()`` for an example.

        Args:
            experience (nested Tensor): samples used for the most recent
                ``update_with_gradient()``. By default it's not summarized.
            train_info (nested Tensor): ``AlgStep.info`` returned by either
                ``rollout_step()`` (on-policy training) or ``train_step()``
                (off-policy training). By default it's not summarized.
            loss_info (LossInfo): loss
            params (list[Parameter]): list of parameters with gradients
        """
        if self._config.summarize_grads_and_vars:
            summary_utils.summarize_variables(params)
            summary_utils.summarize_gradients(params)
        if self._debug_summaries:
            summary_utils.summarize_loss(loss_info)
