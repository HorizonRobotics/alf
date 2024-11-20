# Copyright (c) 2024 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import numpy as np

import torch
from typing import Optional, Tuple, Callable, Union

import alf
from alf.data_structures import AlgStep, namedtuple
from alf.networks.encoding_networks import EncodingNetwork
from alf.algorithms.algorithm import Algorithm
from alf.utils.dist_utils import Beta
from alf.utils import losses

FlowMatchingInfo = namedtuple(
    'FlowMatchingInfo', ['loss', 'denoise_vec', 'pred_denoise_vec'],
    default_value=())


class FlowMatchingAlgorithm(Algorithm):
    """Implement flow matching with a simple linear Gaussian (or optimal transport)
    probability path, as described in

        π0: A Vision-Language-Action Flow Model for General Robot Control, Physical Intelligence, 2024.

    The general idea is to assume a corruption process of the form

    .. math::

        I^t = t*I + (1-t)*e

    where :math:`I` is the clean image, :math:`e` is the noise from :math:`N(0,1)`,
    and :math:`t` is the time step from 0 to 1.

    By taking the derivative of the above equation with respect to time, we get
    the vector field

    .. math::

        u^t = I - e

    which is a constant independent of the time step.

    The purpose of this algorithm is to learn a network :math:`v(I^t,t)` to regress
    this vector field given the corrupted image :math:`I^t` and the time step :math:`t`.

    When generating a new output, we first sample a noise as :math:`I^0\sim N(0,1)`,
    then use Euler integration to generate a denoised output as

    .. math::

        I^{t+dt} = I^t + v(I^t,t) dt

    The vector field network also supports conditional inputs, in a scenario where
    the generated output is a function of the conditional inputs (e.g., actions
    conditioned on robot observations).
    """

    def __init__(self,
                 output_tensor_spec: alf.TensorSpec,
                 noise_tensor_spec: Optional[alf.TensorSpec] = None,
                 cond_input_tensor_spec: alf.NestedTensorSpec = None,
                 vector_field_network_ctor: Callable = EncodingNetwork,
                 tau_beta_paras: Tuple[float] = (1., 1.),
                 noise_std: float = 1.,
                 integration_steps: int = 10,
                 integration_type: str = 'euler',
                 loss_fn: Callable = losses.element_wise_huber_loss,
                 name: str = "FlowMatchingAlgorithm"):
        """
        Args:
            output_tensor_spec: the tensor spec for the output to be denoised. It
                should be a single tensor spec with shape ``[N,D1,D2,...]`` where
                ``N`` is the feature dim and ``D1,D2,...`` are the spatial dims.
            noise_tensor_spec: if provided, must have the same number of dims with
                ``output_tensor_spec``. Also the feature dim ``N`` should be the
                same with ``output_tensor_spec``. Potentially it can have smaller
                tailing dims than ``output_tensor_spec``.
                If None, ``output_tensor_spec`` will be used.
            cond_input_tensor_spec: nested tensor spec for arbitrary conditional
                inputs.
            vector_field_network_ctor: constructor for the vector field network.
                The constructor is expected to take a pair ``(noisy_output, tau)``
                if ``cond_input_tensor_spec`` is None, or a triplet
                ``(noisy_output, tau, cond_input)`` if ``cond_input_tensor_spec``
                is not None, where ``tau`` is a [0,1] scalar of shape ``[B]``.
            tau_beta_paras: beta distribution parameters a and b for sampling tau
                which indicates the denoising time step. A bigger ``a`` puts more
                emphasis on training earlier time steps (denoising noisier outputs).
            noise_std: the standard deviation of the noise to construct corrupted
                outputs. A larger std will make the denoising process slower (i.e.,
                more steps on noisier outputs).
            integration_steps: the number of integration steps when generating a
                new output.
            integration_type: either "euler" or "midpoint".
            loss_fn: loss function for predicting the vector field.
            name: the name of the algorithm.
        """
        super().__init__(name=name)
        self._tau_beta = Beta(
            torch.tensor(tau_beta_paras[0]), torch.tensor(tau_beta_paras[1]))
        self._int_steps = integration_steps
        assert integration_type in ('euler', 'midpoint')
        self._int_type = integration_type
        self._noise_spec = noise_tensor_spec or output_tensor_spec
        self._noise_std = noise_std
        self._output_spec = output_tensor_spec
        assert (len(self._noise_spec.shape) == len(self._output_spec.shape)
                and self._noise_spec.shape[0] == self._output_spec.shape[0])
        tau_spec = alf.TensorSpec(shape=(), dtype=torch.float32)
        net_input_spec = (output_tensor_spec, tau_spec)
        if cond_input_tensor_spec is not None:
            net_input_spec += (cond_input_tensor_spec, )
        self._vector_field_net = vector_field_network_ctor(
            input_tensor_spec=net_input_spec)
        self._loss_fn = loss_fn

    def _get_random_noise(self, batch_size):
        # Sample a random noise at t=0
        noise = self._noise_spec.randn(
            outer_dims=(batch_size, )) * self._noise_std
        return self._resize_noise(noise)

    def _resize_noise(self, noise):
        if self._noise_spec is not self._output_spec:
            # TODO: might be incompatible with TensorRT
            noise = torch.nn.functional.interpolate(
                noise, size=self._output_spec.shape[1:], mode='bilinear')
        return noise

    def train_step(self,
                   inputs: Union[Tuple[alf.nest.NestedTensor, torch.
                                       Tensor], alf.nest.NestedTensor],
                   state=()):
        """Perform a training step of the flow matching algorithm.

        Args:
            inputs: either a tuple of ``(output, cond_input)`` or a single
                ``output``.
        """
        if isinstance(inputs, tuple):
            output, cond_input = inputs
        else:
            output, cond_input = inputs, None
        batch_size = alf.nest.get_nest_batch_size(output)

        # Construct corrupted output
        noise = self._get_random_noise(batch_size)
        tau = self._tau_beta.sample((batch_size, ))
        tau_ = tau.reshape(-1, *([1] * (output.ndim - 1)))
        noisy_output = output * tau_ + noise * (1 - tau_)
        # Cmpute the vector field
        denoising_vec = output - noise
        # Predict the vector field
        inputs = (
            noisy_output,
            tau,
        )
        if cond_input is not None:
            inputs += (cond_input, )
        pred_denoising_vec = self._vector_field_net(inputs)[0]

        loss = self._loss_fn(denoising_vec, pred_denoising_vec)
        loss = loss.sum(list(range(1, loss.ndim)))
        return AlgStep(
            info=FlowMatchingInfo(
                loss=loss,
                denoise_vec=denoising_vec,
                pred_denoise_vec=pred_denoising_vec))

    def generate(self,
                 cond_input: alf.nest.NestedTensor = None,
                 batch_size: int = 1,
                 return_intermediate_steps: bool = False):
        """Generate new outputs.

        Args:
            cond_input: the conditional input. If provided, its batch size will
                be used instead of ``batch_size``.
            batch_size: the batch size of the generated outputs.
            return_intermediate_steps: whether to return the intermediate outputs.
                If True, return all ``self._int_steps`` outputs in a list; otherwise
                return the final output.
        """
        if cond_input is not None:
            batch_size = alf.nest.get_nest_batch_size(cond_input)
        output = self._get_random_noise(batch_size)
        outputs = [output]
        delta = 1. / self._int_steps

        def _time_forward(x0, x, t, dt):
            """Compute x' = x0 + v(x, t)dt
            """
            inputs = (x, t)
            if cond_input is not None:
                inputs += (cond_input, )
            return x0 + dt * self._vector_field_net(inputs)[0]

        for t in np.arange(0, 1, delta):
            tau = torch.full((batch_size, ), t)
            if self._int_type == 'midpoint':
                output_mid = _time_forward(output, output, tau, delta / 2)
                output = _time_forward(output, output_mid, tau + delta / 2,
                                       delta)
            else:
                output = _time_forward(output, output, tau, delta)
            outputs.append(output)

        if return_intermediate_steps:
            return outputs
        return outputs[-1]
