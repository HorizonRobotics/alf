# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Diffusion-like generative models driven by SDEs."""

import math
import torch
import alf
from alf.utils import summary_utils
from alf.utils.dist_utils import TruncatedNormal
from alf.nest import get_nest_batch_size
from alf.utils.common import expand_dims_as


class SDE:
    r"""Base stochastic differential equation interface.

    .. math::

        dx = -\beta(t) x dt + g(t) dw

    Sub-classes provide closed-form coefficients for specific SDE families.

    """

    def pt0(self, t):
        """Return the closed-form :math:`p(x_t | x_0)` parameters.

        Args:
            t: Tensor of time values at which to evaluate the marginal.

        Returns:
            Tuple ``(alpha, sigma)`` representing the linear coefficient and
            standard deviation of the conditional distribution.
            :math:`p(x_t | x_0) = Normal(alpha * x_0, sigma^2 I)`.
        """
        raise NotImplementedError

    def pt0_dot(self, t):
        """Return the derivatives of pt0 with respect to time.

        Args:
            t: Tensor of time values to differentiate at.

        Returns:
            Tuple ``(alpha_dot, sigma_dot)`` corresponding to the time
            derivatives of the parameters of :math:`p(x_t | x_0)`.
        """
        raise NotImplementedError

    def diffusion_coeff(self, t):
        r"""Return the drift and diffusion coefficients :math:`(\beta, g)`.

        Args:
            t: Tensor of time values at which to evaluate the coefficients.

        Returns:
            Tuple ``(beta, g)`` describing the SDE drift and diffusion terms.
        """
        raise NotImplementedError


class OTSDE(SDE):
    # codespell:ignore-begin
    r"""Optimal transport SDE with closed-form linear coefficients.

    This corresponds to commonly used flow matching model ("FM/OT" in https://arxiv.org/abs/2210.02747).

    .. math::

        dx = -\frac{1}{1-t} x dt + \sqrt{\frac{2t}{1-t}} dw

    """

    # codespell:ignore-end

    def pt0(self, t):
        return 1 - t, t

    def pt0_dot(self, t):
        return -1, 1

    def diffusion_coeff(self, t):
        return 1 / (1 - t), (2 * t / (1 - t)).sqrt()


class RTSDE(SDE):
    r"""SDE with trigonometric drift and diffusion.

    .. math::

        dx = - \frac{\pi}{2}\tan(\frac{\pi}{2}t)xdt + \sqrt{\pi\tan(\frac{\pi}{2}t)} dw

    """

    def pt0(self, t):
        angle = 0.5 * math.pi * t
        return torch.cos(angle), torch.sin(angle)

    def pt0_dot(self, t):
        angle = 0.5 * math.pi * t
        return -0.5 * math.pi * torch.sin(angle), 0.5 * math.pi * torch.cos(
            angle)

    def diffusion_coeff(self, t):
        beta = 0.5 * math.pi * torch.tan(0.5 * math.pi * t)
        return 0.5 * beta, (2 * beta)**0.5


class SDEGenerator(torch.nn.Module):
    """Base class for diffusion-like generators driven by an SDE."""

    def __init__(self,
                 input_spec,
                 output_spec,
                 model_ctor,
                 sde: SDE,
                 mean_flow=False,
                 time_sampler=torch.rand,
                 steps=5):
        """Initialize the generator and underlying neural model.

        Args:
            input_spec: Spec describing conditional inputs.
            output_spec: Spec for generated data.
            model_ctor: Factory creating the predictor network. The callable
                should accept ``(input_spec, output_spec, mean_flow)`` and
                return an :class:`alf.networks.Network` that consumes
                ``(x_t, inputs, t[, h])``.
            sde: Stochastic differential equation describing the generative
                process.
            mean_flow: If ``True``, the model will receive an additional input
                encoding the look-ahead horizon for mean flow training.
            time_sampler: Callable that samples the training time ``t``.
            steps: Number of Euler steps used during sampling.
        """
        super().__init__()
        self._model = model_ctor(input_spec, output_spec, mean_flow=mean_flow)
        self._sde = sde
        self._output_spec = output_spec
        self._steps = steps
        if isinstance(output_spec, alf.BoundedTensorSpec):
            self._min = torch.tensor(output_spec.minimum)
            self._max = torch.tensor(output_spec.maximum)
        self._time_sampler = time_sampler

    def calc_loss(self, inputs, samples, f_neg_energy=None, sample_mask=None):
        """Compute per-sample losses for training.

        Args:
            inputs: Conditional input nest consumed by the model during
                training.
            samples: Observed ``x_0`` tensors. When ``None`` the implementation
                will draw ``x_0`` from the prior distribution.
            f_neg_energy: Optional callable ``f(x0, inputs) -> Tensor`` that
                returns per-sample negative energies used to importance weight
                the loss.
            sample_mask: Optional broadcastable tensor used to zero out losses
                of masked samples.
        """
        raise NotImplementedError

    def sample(self, inputs, steps):
        """Generate samples given the conditioning inputs.

        Args:
            inputs: Conditional inputs used during generation.
            steps: Number of Euler integration steps to use.

        Returns:
            Generated samples matching ``output_spec``.
        """
        raise NotImplementedError

    @property
    def state_spec(self):
        return ()

    def _apply_sample_mask(self, diff, sample_mask):
        """Apply a mask to the loss tensor if provided.

        Args:
            diff: Tensor containing per-element loss values.
            sample_mask: Optional mask tensor broadcastable to ``diff``.

        Returns:
            Masked loss tensor.
        """
        if sample_mask is not None:
            if sample_mask.ndim == 1:
                sample_mask = expand_dims_as(sample_mask, diff)
            diff = diff * sample_mask
        return diff

    def _calc_weights(self, x0, alpha, sigma, inputs, f_neg_energy):
        """Compute importance weights based on the energy function.

        Args:
            x0: Tensor of ``x_0`` samples.
            alpha: Scaling factor from ``p(x_t | x_0)``.
            sigma: Standard deviation from ``p(x_t | x_0)``.
            inputs: Conditional inputs associated with each ``x_0`` sample.
            f_neg_energy: Callable returning negative energies for importance
                weighting.

        Returns:
            Weights representing the likelihood of each ``x_0`` under the
            energy function.
        """
        with torch.no_grad():
            neg_energy = f_neg_energy(x0, inputs)
            return neg_energy.exp()


def expand_dims(x, ndim):
    """Reshape ``x`` to add ``ndim`` singleton dimensions at the end.

    Args:
        x: Tensor to reshape.
        ndim: Number of singleton dimensions to append.

    Returns:
        Reshaped tensor with additional singleton dimensions.
    """
    return x.reshape(x.shape[0], *((1, ) * ndim))


@alf.configurable
class FlowMatching(SDEGenerator):
    """Flow matching objective that regresses the true velocity field."""

    def _get_x0_xt_x1(self, samples, batch_size, alpha, sigma):
        """Sample ``(x_0, x_t, x_1)`` triplets for flow matching.

        Args:
            samples: Optional tensor of ground-truth ``x_0`` values.
            batch_size: Number of samples to draw.
            alpha: Scaling factor from the SDE marginal.
            sigma: Standard deviation from the SDE marginal.

        Returns:
            Tuple ``(x0, xt, x1)`` consistent with the SDE marginals.
        """
        if samples is None:
            if isinstance(self._output_spec, alf.BoundedTensorSpec):
                # For bounded data, we use a p1(.) that is uniform within the bounds.
                xt = self._output_spec.sample((batch_size, ))
                # x1 = self._output_spec.randn((batch_size,))

                x1_max = (xt - alpha * self._min) / sigma
                x1_min = (xt - alpha * self._max) / sigma
                dist = TruncatedNormal(loc=torch.zeros_like(x1_min),
                                       scale=torch.ones_like(x1_min),
                                       lower_bound=x1_min,
                                       upper_bound=x1_max)
                x1 = dist.sample()

                # x1_max = x1_max.minimum(self._max)
                # x1_min = x1_min.maximum(self._min)
                # x1 = torch.rand((batch_size,) + self._output_spec.shape) * (x1_max - x1_min) + x1_min

            else:
                xt = self._output_spec.randn((batch_size, ))
                x1 = self._output_spec.randn((batch_size, ))
            x0 = (xt - sigma * x1) / alpha
        else:
            x0 = samples
            x1 = torch.randn((batch_size, ) + self._output_spec.shape,
                             device=samples.device)
            xt = alpha * x0 + sigma * x1

        return x0, xt, x1

    def calc_loss(self, inputs, samples, f_neg_energy=None, sample_mask=None):
        """Compute the squared error between predicted and true velocities.

        Args:
            inputs: Conditional input nest.
            samples: Optional tensor of ground-truth ``x_0`` values.
            f_neg_energy: Optional energy function for importance weighting.
            sample_mask: Optional mask applied to the per-element losses.

        Returns:
            Tensor of per-sample velocity regression losses.
        """
        leaf = alf.nest.extract_any_leaf_from_nest(inputs)
        batch_size = leaf.shape[0]
        device = leaf.device
        t = self._time_sampler(batch_size, device=device)
        alpha, sigma = self._sde.pt0(expand_dims(t, self._output_spec.ndim))
        x0, xt, x1 = self._get_x0_xt_x1(samples, batch_size, alpha, sigma)
        vt = self._model((xt, inputs, t))[0]
        alpha_dot, sigma_dot = self._sde.pt0_dot(
            expand_dims(t, self._output_spec.ndim))
        cvt = alpha_dot * x0 + sigma_dot * x1
        diff = (vt - cvt)**2
        diff = self._apply_sample_mask(diff, sample_mask)
        loss = diff.reshape(batch_size, -1).sum(-1)

        if f_neg_energy is not None:
            p0 = self._calc_weights(x0, alpha, sigma, inputs, f_neg_energy)
            loss = p0 * loss

        return loss

    def sample(self, inputs):
        """Sample by integrating the learned velocity field.

        Args:
            inputs: Conditional inputs used for generation.

        Returns:
            Tensor of generated samples.
        """
        batch_size = get_nest_batch_size(inputs)
        if isinstance(self._output_spec, alf.BoundedTensorSpec):
            noise = self._output_spec.sample((batch_size, ))
        else:
            noise = self._output_spec.randn((batch_size, ))
        noise = noise * self._sde.pt0(torch.tensor(1))[1]
        dt = 1 / self._steps
        with torch.no_grad():
            for step in range(0, self._steps):
                t = torch.full((batch_size, ), 1 - step * dt)
                vt = self._model((noise, inputs, t))[0]
                noise = noise - vt * dt

        return noise
