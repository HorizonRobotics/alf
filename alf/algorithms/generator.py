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

from collections import namedtuple

import gin
import tensorflow as tf

from tf_agents.networks.network import Network

from alf.algorithms.algorithm import Algorithm, AlgorithmStep, LossInfo
from alf.algorithms.mi_estimator import MIEstimator
from alf.utils import common
from alf.utils.averager import AdaptiveAverager
from alf.utils.encoding_network import EncodingNetwork

GeneratorLossInfo = namedtuple("GeneratorLossInfo",
                               ["generator", "mi_estimator"])


@gin.configurable
class Generator(Algorithm):
    """Generator

    Generator generates outputs given `inputs` (can be None) by transforming
    a random noise and input using `net`:
        outputs = net([noise, inputs]) if input is not None
                  else net(noise)

    It has two modes:
    ML: net is trained to minized loss_func([outputs, inputs])
    STEIN: net is trained to generate outputs to match the distribution whose
        (unnormalized) negative probability is given by loss_func([outputs, inputs]).
        The matching is achieved by using amortized Stein variational gradient
        descent (SVGD). See the following paper for reference
        Feng et al "Learning to Draw Samples with Amortized Stein Variational
        Gradient Descent" https://arxiv.org/pdf/1707.06626.pdf

    It also supports an additional optional objective of maximizing the mutual
    information between [inputs, noise] and outputs by using mi_estimator
    """

    def __init__(self,
                 output_dim,
                 noise_dim=32,
                 input_tensor_spec=None,
                 hidden_layers=(256, ),
                 net: Network = None,
                 mode='ML',
                 kernel_sharpness=2.,
                 mi_weight=None,
                 mi_estimator_cls=MIEstimator,
                 optimizer: tf.optimizers.Optimizer = None,
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
            mode (str): one of ('ML', 'STEIN')
            kernel_sharpness (float): Used only for mode 'STEIN'. The kernel
                for SVGD is calcualted as:
                    exp(-kernel_sharpness * reduce_mean((x-y)^2/width)),
                where width is the elementwise moving average of (x-y)^2
            mi_estimator_cls (type): the class of mutual information estimator
                for maximizing the mutual information between [noise, inputs]
                and [outputs, inputs].
            optimizer (tf.optimizers.Optimizer): optimizer (optional)
            name (str): name of this generator
        """
        super().__init__(train_state_spec=(), optimizer=optimizer, name=name)
        self._noise_dim = noise_dim
        if mode == 'ML':
            self._grad_func = self._ml_grad
        elif mode == 'STEIN':
            self._grad_func = self._stein_grad
            self._kernel_width_averager = AdaptiveAverager(
                tensor_spec=tf.TensorSpec(shape=(output_dim, )))
            self._kernel_sharpness = kernel_sharpness
        else:
            raise ValueError("Unsupported mode %s" % mode)

        noise_spec = tf.TensorSpec(shape=[noise_dim])

        if net is None:
            net_input_spec = noise_spec
            if input_tensor_spec is not None:
                net_input_spec = [net_input_spec, input_tensor_spec]
            net = EncodingNetwork(
                name="Generator",
                input_tensor_spec=net_input_spec,
                fc_layer_params=hidden_layers,
                last_layer_size=output_dim)

        self._mi_estimator = None
        self._input_tensor_spec = input_tensor_spec
        if mi_weight is not None:
            x_spec = noise_spec
            y_spec = tf.TensorSpec((output_dim, ))
            if input_tensor_spec is not None:
                x_spec = [x_spec, input_tensor_spec]
            self._mi_estimator = mi_estimator_cls(
                x_spec, y_spec, sampler='shift')
            self._mi_weight = mi_weight
        self._net = net

    def _predict(self, inputs, batch_size=None):
        if inputs is None:
            assert self._input_tensor_spec is None
            assert batch_size is not None
        else:
            tf.nest.assert_same_structure(inputs, self._input_tensor_spec)
            batch_size = tf.shape(tf.nest.flatten(inputs)[0])[0]
        shape = common.concat_shape([batch_size], [self._noise_dim])
        noise = tf.random.normal(shape=shape)
        gen_inputs = noise if inputs is None else [noise, inputs]
        outputs = self._net(gen_inputs)[0]
        return outputs, gen_inputs

    def predict(self, inputs, batch_size=None, state=None):
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
        outputs, _ = self._predict(inputs, batch_size)
        return AlgorithmStep(outputs=outputs)

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
        loss_propagated = tf.reduce_sum(
            tf.stop_gradient(grad) * outputs, axis=-1)

        mi_loss = ()
        if self._mi_estimator is not None:
            mi_step = self._mi_estimator.train_step([gen_inputs, outputs])
            mi_loss = mi_step.info.loss
            loss_propagated += self._mi_weight * mi_loss

        return AlgorithmStep(
            outputs=outputs,
            state=(),
            info=LossInfo(
                loss=loss_propagated,
                extra=GeneratorLossInfo(generator=loss, mi_estimator=mi_loss)))

    def _ml_grad(self, inputs, outputs, loss_func):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(outputs)
            loss_inputs = outputs if inputs is None else [outputs, inputs]
            loss = loss_func(loss_inputs)
            scalar_loss = tf.reduce_sum(loss)
        grad = tape.gradient(scalar_loss, outputs)
        return loss, grad

    def _kernel_func(self, x, y):
        d = tf.square(x - y)
        self._kernel_width_averager.update(tf.reduce_mean(d, axis=0))
        d = tf.reduce_mean(d / self._kernel_width_averager.get(), axis=-1)
        w = tf.math.exp(-self._kernel_sharpness * d)
        return w

    def _stein_grad(self, inputs, outputs, loss_func):
        outputs2, _ = self._predict(inputs, batch_size=tf.shape(outputs)[0])
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(outputs2)
            kernel_weight = self._kernel_func(outputs, outputs2)
            weight_sum = tf.reduce_sum(kernel_weight)

        kernel_grad = tape.gradient(weight_sum, outputs2)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(outputs2)
            loss_inputs = outputs2 if inputs is None else [outputs2, inputs]
            loss = loss_func(loss_inputs)
            weighted_loss = tf.stop_gradient(kernel_weight) * loss
            scalar_loss = tf.reduce_sum(weighted_loss)

        loss_grad = tape.gradient(scalar_loss, outputs2)
        return loss, loss_grad - kernel_grad
