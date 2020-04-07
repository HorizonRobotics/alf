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
                 optimizer=None,
                 name="Generator"):
        """Create a Generator.

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
            self._grad_func = self._stein_grad
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
            common.get_target_updater(self._net, self._predict_net, tau=1.0)

    def _trainable_attributes_to_ignore(self):
        return ["_predict_net"]

    def _predict(self, inputs, batch_size=None, training=True):
        if inputs is None:
            assert self._input_tensor_spec is None
            assert batch_size is not None
        else:
            nest.assert_same_structure(inputs, self._input_tensor_spec)
            batch_size = nest.flatten(inputs)[0].shape[0]
        noise = torch.randn(batch_size, self._noise_dim)
        gen_inputs = noise if inputs is None else [noise, inputs]
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
        outputs, _ = self._predict(inputs, batch_size, training=False)
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
        outputs, gen_inputs = self._predict(inputs, batch_size)
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

    def _kernel_func(self, x, y):
        d = (x - y)**2
        self._kernel_width_averager.update(torch.mean(d, dim=0))
        d = torch.mean(d / self._kernel_width_averager.get(), dim=-1)
        w = torch.exp(-self._kernel_sharpness * d)
        return w

    def _stein_grad(self, inputs, outputs, loss_func):
        outputs2, _ = self._predict(inputs, batch_size=outputs.shape[0])
        kernel_weight = self._kernel_func(outputs, outputs2)
        weight_sum = self._entropy_regularization * kernel_weight.sum()

        kernel_grad = torch.autograd.grad(weight_sum, outputs2)[0]

        loss_inputs = outputs2 if inputs is None else [outputs2, inputs]
        loss = loss_func(loss_inputs)
        weighted_loss = kernel_weight.detach() * loss

        loss_grad = torch.autograd.grad(weighted_loss.sum(), outputs2)[0]

        return loss, loss_grad - kernel_grad

    def after_update(self, training_info):
        if self._predict_net:
            common.get_target_updater(
                self._net,
                self._predict_net,
                tau=self._net_moving_average_rate)
