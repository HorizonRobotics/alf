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
"""A generic generator."""

import gin
import numpy as np
import torch

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.mi_estimator import MIEstimator
from alf.data_structures import AlgStep, LossInfo, namedtuple
import alf.nest as nest
from alf.networks import Network, EncodingNetwork, ReluMLP
from alf.tensor_specs import TensorSpec
from alf.utils import common, math_ops
from alf.utils.averager import AdaptiveAverager

GeneratorLossInfo = namedtuple("GeneratorLossInfo",
                               ["generator", "mi_estimator"])


@gin.configurable
class CriticAlgorithm(Algorithm):
    """
    Wrap a critic network as an Algorithm for flexible gradient updates
    called by the Generator when par_vi is 'minmax'.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_dim=None,
                 hidden_layers=(3, 3),
                 activation=torch.relu_,
                 net: Network = None,
                 use_relu_mlp=False,
                 use_bn=True,
                 optimizer=None,
                 name="CriticAlgorithm"):
        """Create a CriticAlgorithm.

        Args:
            input_tensor_spec (TensorSpec): spec of inputs. 
            output_dim (int): dimension of output, default value is input_dim.
            hidden_layers (tuple): size of hidden layers.
            activation (nn.functional): activation used for all critic layers.
            net (Network): network for predicting outputs from inputs.
                If None, a default one with hidden_layers will be created
            use_relu_mlp (bool): whether use ReluMLP as default net constrctor.
                Diagonals of Jacobian can be explicitly computed for ReluMLP.
            use_bn (bool): whether use batch norm for each critic layers.
            optimizer (torch.optim.Optimizer): (optional) optimizer for training.
            name (str): name of this CriticAlgorithm.
        """
        if optimizer is None:
            optimizer = alf.optimizers.Adam(lr=1e-3)
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)

        self._use_relu_mlp = use_relu_mlp
        self._output_dim = output_dim
        if output_dim is None:
            self._output_dim = input_tensor_spec.shape[0]
        if net is None:

            if use_relu_mlp:
                net = ReluMLP(
                    input_tensor_spec=input_tensor_spec,
                    hidden_layers=hidden_layers,
                    activation=activation)
            else:
                net = EncodingNetwork(
                    input_tensor_spec=input_tensor_spec,
                    fc_layer_params=hidden_layers,
                    use_fc_bn=use_bn,
                    activation=activation,
                    last_layer_size=self._output_dim,
                    last_activation=math_ops.identity,
                    last_use_fc_bn=use_bn,
                    name='Critic')
        self._net = net

    def reset_net_parameters(self):
        for fc in self._net._fc_layers:
            fc.reset_parameters()

    def predict_step(self, inputs, state=None, requires_jac_diag=False):
        """Predict for one step of inputs.

        Args:
            inputs (torch.Tensor): inputs for prediction.
            state: not used.
            requires_jac_trace (bool): whether outputs diagonals of Jacobian.

        Returns:
            AlgStep:
            - output (torch.Tensor): predictions or (predictions, diag_jacobian)
                if requires_jac_diag is True.
            - state: not used.
        """
        if self._use_relu_mlp:
            outputs = self._net(inputs, requires_jac_diag=requires_jac_diag)[0]
        else:
            outputs = self._net(inputs)[0]

        return AlgStep(output=outputs, state=(), info=())


@gin.configurable
class Generator(Algorithm):
    r"""Generator

    Generator generates outputs given `inputs` (can be None) by transforming
    a random noise and input using `net`:

        outputs = net([noise, input]) if input is not None
                  else net(noise)

    The generator is trained to minimize the following objective:

        :math:`E(loss\_func(net([noise, input]))) - entropy\_regulariztion \cdot H(P)`

    where P is the (conditional) distribution of outputs given the inputs
    implied by `net` and H(P) is the (conditional) entropy of P.

    If the loss is the (unnormalized) negative log probability of some
    distribution Q and the ``entropy_regularization`` is 1, this objective is
    equivalent to minimizing :math:`KL(P||Q)`.

    It uses two different ways to optimize `net` depending on
    ``entropy_regularization``:

    * ``entropy_regularization`` = 0: the minimization is achieved by simply
      minimizing loss_func(net([noise, inputs]))

    * entropy_regularization > 0: the minimization is achieved using amortized
      particle-based variational inference (ParVI), in particular, three ParVI
      methods are implemented:

        1. amortized Stein Variational Gradient Descent (SVGD):

        Feng et al "Learning to Draw Samples with Amortized Stein Variational
        Gradient Descent" https://arxiv.org/pdf/1707.06626.pdf

        2. amortized Wasserstein ParVI with Smooth Functions (GFSF):

        Liu, Chang, et al. "Understanding and accelerating particle-based 
        variational inference." International Conference on Machine Learning. 2019.

        3. amortized Fisher Neural Sampler with Hutchinson's estimator (MINMAX):

        Hu et at. "Stein Neural Sampler." https://arxiv.org/abs/1810.03545, 2018.

    It also supports an additional optional objective of maximizing the mutual
    information between [noise, inputs] and outputs by using mi_estimator to
    prevent mode collapse. This might be useful for ``entropy_regulariztion`` = 0
    as suggested in section 5.1 of the following paper:

    Hjelm et al `Learning Deep Representations by Mutual Information Estimation
    and Maximization <https://arxiv.org/pdf/1808.06670.pdf>`_
    """

    def __init__(self,
                 output_dim,
                 noise_dim=32,
                 input_tensor_spec=None,
                 hidden_layers=(256, ),
                 net: Network = None,
                 net_moving_average_rate=None,
                 entropy_regularization=0.,
                 mi_weight=None,
                 mi_estimator_cls=MIEstimator,
                 par_vi=None,
                 critic_input_dim=None,
                 critic_hidden_layers=(100, 100),
                 critic_l2_weight=10.,
                 critic_iter_num=2,
                 critic_relu_mlp=False,
                 critic_use_bn=True,
                 minmax_resample=True,
                 critic_optimizer=None,
                 optimizer=None,
                 name="Generator"):
        r"""Create a Generator.

        Args:
            output_dim (int): dimension of output
            noise_dim (int): dimension of noise
            input_tensor_spec (nested TensorSpec): spec of inputs. If there is
                no inputs, this should be None.
            hidden_layers (tuple): sizes of hidden layers.
            net (Network): network for generating outputs from [noise, inputs]
                or noise (if inputs is None). If None, a default one with
                hidden_layers will be created
            net_moving_average_rate (float): If provided, use a moving average
                version of net to do prediction. This has been shown to be
                effective for GAN training (arXiv:1907.02544, arXiv:1812.04948).
            entropy_regularization (float): weight of entropy regularization.
            mi_weight (float): weight of mutual information loss.
            mi_estimator_cls (type): the class of mutual information estimator
                for maximizing the mutual information between [noise, inputs]
                and [outputs, inputs].
            par_vi (string): ParVI methods, options are
                [``svgd``, ``svgd2``, ``svgd3``, ``gfsf``, ``minmax``],
                * svgd: empirical expectation of SVGD is evaluated by a single 
                    resampled particle. The main benefit of this choice is it 
                    supports conditional case, while all other options do not.
                * svgd2: empirical expectation of SVGD is evaluated by splitting
                    half of the sampled batch. It is a trade-off between 
                    computational efficiency and convergence speed.
                * svgd3: empirical expectation of SVGD is evaluated by 
                    resampled particles of the same batch size. It has better
                    convergence but involves resampling, so less efficient
                    computaionally comparing with svgd2.
                * gfsf: wasserstein gradient flow with smoothed functions. It 
                    involves a kernel matrix inversion, so computationally most
                    expensive, but in some case the convergence seems faster 
                    than svgd approaches.
                * minmax: Fisher Neural Sampler, optimal descent direction of
                    the Stein discrepancy is solved by an inner optimization
                    procedure in the space of L2 neural networks.
            critic_input_dim (int): dimension of critic input, used for ``minmax``.
            critic_hidden_layers (tuple): sizes of hidden layers of the critic,
                used for ``minmax``.
            critic_l2_weight (float): weight of L2 regularization in training 
                the critic, used for ``minmax``.
            critic_iter_num (int): number of critic updates for each generator
                train_step, used for ``minmax``.
            critic_relu_mlp (bool): whether use ReluMLP as the critic constructor,
                used for ``minmax``.
            critic_use_bn (book): whether use batch norm for each layers of the
                critic, used for ``minmax``.
            minmax_resample (bool): whether resample the generator for each 
                critic update, used for ``minmax``.
            critic_optimizer (torch.optim.Optimizer): Optimizer for training the
                critic, used for ``minmax``.
            optimizer (torch.optim.Optimizer): (optional) optimizer for training
            name (str): name of this generator
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)
        self._output_dim = output_dim
        self._noise_dim = noise_dim
        self._entropy_regularization = entropy_regularization
        self._par_vi = par_vi
        if entropy_regularization == 0:
            self._grad_func = self._ml_grad
        else:
            if par_vi == 'gfsf':
                self._grad_func = self._gfsf_grad
            elif par_vi == 'svgd':
                self._grad_func = self._svgd_grad
            elif par_vi == 'svgd2':
                self._grad_func = self._svgd_grad2
            elif par_vi == 'svgd3':
                self._grad_func = self._svgd_grad3
            elif par_vi == 'minmax':
                if critic_input_dim is None:
                    critic_input_dim = output_dim
                self._grad_func = self._minmax_grad
                self._critic_iter_num = critic_iter_num
                self._critic_l2_weight = critic_l2_weight
                self._critic_relu_mlp = critic_relu_mlp
                self._minmax_resample = minmax_resample
                self._critic = CriticAlgorithm(
                    TensorSpec(shape=(critic_input_dim, )),
                    hidden_layers=critic_hidden_layers,
                    use_relu_mlp=critic_relu_mlp,
                    use_bn=critic_use_bn,
                    optimizer=critic_optimizer)
            else:
                raise ValueError("Unsupported par_vi method: %s" % par_vi)

            self._kernel_width_averager = AdaptiveAverager(
                tensor_spec=TensorSpec(shape=()))

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
        return ["_predict_net", "_critic"]

    @property
    def noise_dim(self):
        return self._noise_dim

    def _predict(self, inputs=None, noise=None, batch_size=None,
                 training=True):
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

    def predict_step(self,
                     inputs=None,
                     noise=None,
                     batch_size=None,
                     training=False,
                     state=None):
        """Generate outputs given inputs.

        Args:
            inputs (nested Tensor): if None, the outputs is generated only from
                noise.
            noise (Tensor): input to the generator.
            batch_size (int): batch_size. Must be provided if inputs is None.
                Its is ignored if inputs is not None
            training (bool): whether train the generator.
            state: not used

        Returns:
            AlgorithmStep: outputs with shape (batch_size, output_dim)
        """
        outputs, _ = self._predict(
            inputs=inputs,
            noise=noise,
            batch_size=batch_size,
            training=training)
        return AlgStep(output=outputs, state=(), info=())

    def train_step(self,
                   inputs,
                   loss_func,
                   batch_size=None,
                   transform_func=None,
                   entropy_regularization=None,
                   state=None):
        """
        Args:
            inputs (nested Tensor): if None, the outputs is generated only from
                noise.
            loss_func (Callable): loss_func([outputs, inputs])
                (loss_func(outputs) if inputs is None) returns a Tensor or namedtuple
                of tensors with field `loss`, which is a Tensor of
                shape [batch_size] a loss term for optimizing the generator.
            batch_size (int): batch_size. Must be provided if inputs is None.
                Its is ignored if inputs is not None.
            transform_func (Callable): transform function on generator's outputs.
                Used in function value based par_vi (currently supported
                by [``svgd2``, ``svgd3``, ``gfsf``]) for evaluating the network(s)
                parameterized by the generator's outputs (given by self._predict) 
                on the training batch (predefined with transform_func). 
                It can be called in two ways 
                - transform_func(params): params is a tensor of parameters for a 
                    network, of shape ``[D]`` or ``[B, D]``
                    - ``B``: batch size
                    - ``D``: length of network parameters
                    In this case, transform_func first samples additional data besides
                    the predefined training batch and then evaluate the network(s)
                    parameterized by ``params`` on the training batch plus additional
                    sampled data.
                - transform_func((params, extra_samples)): params is the same as
                    above case and extra_samples is the tensor of additional sampled
                    data.
                    In this case, transform_func evaluates the network(s) parameterized 
                    by ``params`` on predefined training batch plus ``extra_samples``. 
                It returns three tensors
                - outputs: outputs of network parameterized by params evaluated 
                    on predined training batch.
                - density_outputs: outputs of network parameterized by params 
                    evaluated on additional sampled data. 
                - extra_samples: additional sampled data, same as input
                    extra_samples if called as transform_func((params, extra_samples))
            entropy_regularization (float): weight of entropy regularization.
            state: not used

        Returns:
            AlgorithmStep:
                outputs: Tensor with shape (batch_size, dim)
                info: LossInfo
        """
        outputs, gen_inputs = self._predict(inputs, batch_size=batch_size)
        if entropy_regularization is None:
            entropy_regularization = self._entropy_regularization
        loss, loss_propagated = self._grad_func(
            inputs, outputs, loss_func, entropy_regularization, transform_func)
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

    def _ml_grad(self,
                 inputs,
                 outputs,
                 loss_func,
                 entropy_regularization=None,
                 transform_func=None):
        assert transform_func is None, (
            "function value based vi is not supported for ml_grad")
        loss_inputs = outputs if inputs is None else [outputs, inputs]
        loss = loss_func(loss_inputs)

        grad = torch.autograd.grad(loss.sum(), outputs)[0]
        loss_propagated = torch.sum(grad.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def _kernel_width(self, dist):
        """Update kernel_width averager and get latest kernel_width. """
        if dist.ndim > 1:
            dist = torch.sum(dist, dim=-1)
            assert dist.ndim == 1, "dist must have dimension 1 or 2."
        width, _ = torch.median(dist, dim=0)
        width = width / np.log(len(dist))
        self._kernel_width_averager.update(width)

        return self._kernel_width_averager.get()

    def _rbf_func(self, x, y):
        """Compute RGF kernel, used by svgd_grad. """
        d = (x - y)**2
        d = torch.sum(d, -1)
        h = self._kernel_width(d)
        w = torch.exp(-d / h)

        return w

    def _rbf_func2(self, x, y):
        r"""
        Compute the rbf kernel and its gradient w.r.t. first entry 
        :math:`K(x, y), \nabla_x K(x, y)`, used by svgd_grad2 and svgd_grad3. 

        Args:
            x (Tensor): set of N particles, shape (Nx, ...), where Nx is the 
                number of particles.
            y (Tensor): set of N particles, shape (Ny, ...), where Ny is the 
                number of particles.

        Returns:
            :math:`K(x, y)` (Tensor): the RBF kernel of shape (Nx x Ny)
            :math:`\nabla_x K(x, y)` (Tensor): the derivative of RBF kernel of shape (Nx x Ny x D)

        """
        Nx = x.shape[0]
        Ny = y.shape[0]
        x = x.view(Nx, -1)
        y = y.view(Ny, -1)
        Dx = x.shape[1]
        Dy = y.shape[1]
        assert Dx == Dy
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [Nx, Ny, D]
        dist_sq = torch.sum(diff**2, -1)  # [Nx, Ny]
        h = self._kernel_width(dist_sq.view(-1))

        kappa = torch.exp(-dist_sq / h)  # [Nx, Nx]
        kappa_grad = kappa.unsqueeze(-1) * (-2 * diff / h)  # [Nx, Ny, D]
        return kappa, kappa_grad

    def _score_func(self, x, alpha=1e-5):
        r"""
        Compute the stein estimator of the score function 
        :math:`\nabla\log q = -(K + \alpha I)^{-1}\nabla K`,
        used by gfsf_grad. 

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
        h, _ = torch.median(dist_sq.view(-1), dim=0)
        h = h / np.log(N)

        kappa = torch.exp(-dist_sq / h)  # [N, N]
        kappa_inv = torch.inverse(kappa + alpha * torch.eye(N))  # [N, N]
        kappa_grad = torch.einsum('ij,ijk->jk', kappa, -2 * diff / h)  # [N, D]

        return -kappa_inv @ kappa_grad

    def _svgd_grad(self,
                   inputs,
                   outputs,
                   loss_func,
                   entropy_regularization,
                   transform_func=None):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by a single resampled particle. 
        """
        outputs2, _ = self._predict(inputs, batch_size=outputs.shape[0])
        assert transform_func is None, (
            "function value based vi is not supported for svgd_grad")
        kernel_weight = self._rbf_func(outputs, outputs2)
        weight_sum = entropy_regularization * kernel_weight.sum()

        kernel_grad = torch.autograd.grad(weight_sum, outputs2)[0]

        loss_inputs = outputs2 if inputs is None else [outputs2, inputs]
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        weighted_loss = kernel_weight.detach() * neglogp

        loss_grad = torch.autograd.grad(weighted_loss.sum(), outputs2)[0]
        grad = loss_grad - kernel_grad
        loss_propagated = torch.sum(grad.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def _svgd_grad2(self,
                    inputs,
                    outputs,
                    loss_func,
                    entropy_regularization,
                    transform_func=None):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by splitting half of the sampled batch. 
        """
        assert inputs is None, '"svgd2" does not support conditional generator'
        if transform_func is not None:
            outputs, extra_outputs, _ = transform_func(outputs)
            aug_outputs = torch.cat([outputs, extra_outputs], dim=-1)
        else:
            aug_outputs = outputs
        num_particles = outputs.shape[0] // 2
        outputs_i, outputs_j = torch.split(outputs, num_particles, dim=0)
        aug_outputs_i, aug_outputs_j = torch.split(
            aug_outputs, num_particles, dim=0)

        loss_inputs = outputs_j
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(),
                                        loss_inputs)[0]  # [Nj, D]

        # [Nj, Ni], [Nj, Ni, D']
        kernel_weight, kernel_grad = self._rbf_func2(aug_outputs_j.detach(),
                                                     aug_outputs_i.detach())
        kernel_logp = torch.matmul(kernel_weight.t(),
                                   loss_grad) / num_particles  # [Ni, D]

        loss_prop_kernel_logp = torch.sum(
            kernel_logp.detach() * outputs_i, dim=-1)
        loss_prop_kernel_grad = torch.sum(
            -entropy_regularization * kernel_grad.mean(0).detach() *
            aug_outputs_i,
            dim=-1)
        loss_propagated = loss_prop_kernel_logp + loss_prop_kernel_grad

        return loss, loss_propagated

    def _svgd_grad3(self,
                    inputs,
                    outputs,
                    loss_func,
                    entropy_regularization,
                    transform_func=None):
        """
        Compute particle gradients via SVGD, empirical expectation
        evaluated by resampled particles of the same batch size. 
        """
        assert inputs is None, '"svgd3" does not support conditional generator'
        num_particles = outputs.shape[0]
        outputs2, _ = self._predict(inputs, batch_size=num_particles)
        if transform_func is not None:
            outputs, extra_outputs, samples = transform_func(outputs)
            outputs2, extra_outputs2, _ = transform_func((outputs2, samples))
            aug_outputs = torch.cat([outputs, extra_outputs], dim=-1)
            aug_outputs2 = torch.cat([outputs2, extra_outputs2], dim=-1)
        else:
            aug_outputs = outputs  # [N, D']
            aug_outputs2 = outputs2  # [N2, D']
        loss_inputs = outputs2
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(),
                                        loss_inputs)[0]  # [N2, D]

        # [N2, N], [N2, N, D']
        kernel_weight, kernel_grad = self._rbf_func2(aug_outputs2.detach(),
                                                     aug_outputs.detach())
        kernel_logp = torch.matmul(kernel_weight.t(),
                                   loss_grad) / num_particles  # [N, D]

        loss_prop_kernel_logp = torch.sum(
            kernel_logp.detach() * outputs, dim=-1)
        loss_prop_kernel_grad = torch.sum(
            -entropy_regularization * kernel_grad.mean(0).detach() *
            aug_outputs,
            dim=-1)
        loss_propagated = loss_prop_kernel_logp + loss_prop_kernel_grad

        return loss, loss_propagated

    def _gfsf_grad(self,
                   inputs,
                   outputs,
                   loss_func,
                   entropy_regularization,
                   transform_func=None):
        """Compute particle gradients via GFSF (Stein estimator). """
        assert inputs is None, '"gfsf" does not support conditional generator'
        if transform_func is not None:
            outputs, extra_outputs, _ = transform_func(outputs)
            aug_outputs = torch.cat([outputs, extra_outputs], dim=-1)
        else:
            aug_outputs = outputs
        score_inputs = aug_outputs.detach()
        loss_inputs = outputs
        loss = loss_func(loss_inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(),
                                        loss_inputs)[0]  # [N2, D]
        logq_grad = self._score_func(score_inputs) * entropy_regularization

        loss_prop_neglogp = torch.sum(loss_grad.detach() * outputs, dim=-1)
        loss_prop_logq = torch.sum(logq_grad.detach() * aug_outputs, dim=-1)
        loss_propagated = loss_prop_neglogp + loss_prop_logq

        return loss, loss_propagated

    def _jacobian_trace(self, fx, x):
        """Hutchinson's trace Jacobian estimator O(1) call to autograd,
            used by "\"minmax\" method"""
        assert fx.shape[-1] == x.shape[-1], (
            "Jacobian is not square, no trace defined.")
        eps = torch.randn_like(fx)
        jvp = torch.autograd.grad(
            fx, x, grad_outputs=eps, retain_graph=True, create_graph=True)[0]
        tr_jvp = torch.einsum('bi,bi->b', jvp, eps)
        return tr_jvp

    def _critic_train_step(self, inputs, loss_func, entropy_regularization=1.):
        """
        Compute the loss for critic training.
        """
        loss = loss_func(inputs)
        if isinstance(loss, tuple):
            neglogp = loss.loss
        else:
            neglogp = loss
        loss_grad = torch.autograd.grad(neglogp.sum(), inputs)[0]  # [N, D]

        if self._critic_relu_mlp:
            critic_step = self._critic.predict_step(
                inputs, requires_jac_diag=True)
            outputs, jac_diag = critic_step.output
            tr_gradf = jac_diag.sum(-1)  # [N]
        else:
            outputs = self._critic.predict_step(inputs).output
            tr_gradf = self._jacobian_trace(outputs, inputs)  # [N]

        f_loss_grad = (loss_grad.detach() * outputs).sum(1)  # [N]
        loss_stein = f_loss_grad - entropy_regularization * tr_gradf  # [N]

        l2_penalty = (outputs * outputs).sum(1).mean() * self._critic_l2_weight
        critic_loss = loss_stein.mean() + l2_penalty

        return critic_loss

    def _minmax_grad(self,
                     inputs,
                     outputs,
                     loss_func,
                     entropy_regularization,
                     transform_func=None):
        """
        Compute particle gradients via minmax svgd (Fisher Neural Sampler). 
        """
        assert inputs is None, '"minmax" does not support conditional generator'

        # optimize the critic using resampled particles
        assert transform_func is None, (
            "function value based vi is not supported for minmax_grad")
        num_particles = outputs.shape[0]

        for i in range(self._critic_iter_num):

            if self._minmax_resample:
                critic_inputs, _ = self._predict(
                    inputs, batch_size=num_particles)
            else:
                critic_inputs = outputs.detach().clone()
                critic_inputs.requires_grad = True

            critic_loss = self._critic_train_step(critic_inputs, loss_func,
                                                  entropy_regularization)
            self._critic.update_with_gradient(LossInfo(loss=critic_loss))

        # compute amortized svgd
        loss = loss_func(outputs.detach())
        critic_outputs = self._critic.predict_step(outputs.detach()).output
        loss_propagated = torch.sum(-critic_outputs.detach() * outputs, dim=-1)

        return loss, loss_propagated

    def after_update(self, training_info):
        if self._predict_net:
            self._predict_net_updater()
