# Copyright (c) 2022 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import torch
import torch.nn as nn
from typing import Any

import alf
from alf.utils.averager import ScalarEMAverager
from alf.utils.tensor_utils import global_norm, clip_by_global_norm


def get_opt_arg(p: nn.Parameter, argname: str, default: Any = None):
    """Get parameter specific optimizer arguments.

    Args:
        p: the parameter
        argname: name of the argument
        default: the default value
    Returns:
        The parameter specific value if it is found, otherwise default
    """
    opt_args = getattr(p, 'opt_args', None)
    if opt_args is None:
        return default
    value = opt_args.get(argname, None)
    return default if value is None else value


@alf.configurable
class GradientNoiseScaleEstimator(nn.Module):
    r"""Implement the simple Gradient Noise Scale estimator as detailed in
    Appendix A, "An Empirical Model of Large-Batch Training", McCandlish et al.,
    `arXiv <https://arxiv.org/pdf/1812.06162.pdf>`_, 2018.

    The simplified GNS is defined as:

    .. math::

        B_{simple} = \frac{tr(\Sigma(\theta))}{|G(\theta)|^2},

    where :math:`\Sigma` is the per-sample covariance matrix defined as

    .. math::

        \Sigma(\theta) = cov_{x\sim p} (\Nabla_{\theta} L_x(\theta)),

    and :math:`G(\theta)` is the true gradient given the entire data distribution.

    Generally, GNS indicates the noise-to-signal value of SGD. The authors suggest
    that we should choose a batch size close to GNS in order to average out the
    noise in the gradient. In other words, GNS is positively correlated to the
    current gradient descent difficulty. We would expect a high GNS for a difficult
    learning task, especially when different training samples generate opposite
    gradient directions.

    .. note::

        You can turn on this estimator in ``TrainerConfig``. However, this will
        increase the back-propagation overhead.

        Note that the *expectation* of the estimated GNS is independent with
        the batch size in theory, but does depend on the learning rate. A good
        practice of using this estimator given a learning rate is to make sure:

        1. the learning rate is reasonable. If it's too large, then GNS is unstable.
        1. that the batch size is large enough (smaller variance), and
        2. the batch data can represent samples from the true data distribution.
           For example, if your batch is too large but the replay buffer is too
           small, then the estimate won't make sense (consider increasing the
           ``initial_collect_steps``).

    We also provide an alternative way of estimating GNS. Given the gradients of
    two sampled batches :math:`G_{est1}` and :math:`G_{est2}`, we have

    .. math::

        \begin{array}{l}
            \alpha\triangleq \mathbb{E}[<G_{est1}\circ G_{est2}>] = |G|^2 \\
            \beta\triangleq\mathbb{E}[\frac{|G_{est1}|^2 + |G_{est2}|^2}{2}] = \frac{1}{B}tr[\Sigma] + |G|^2 \\
        \end{array}

    Then we can maintain a moving average of :math:`\bar{\alpha}` and :math:`\bar{\beta}`,
    and use :math:`(\frac{\bar{\beta}}{\bar{\alpha}}-1)B` as the estimated GNS.
    """

    def __init__(self,
                 batch_size_ratio: float = 0.1,
                 update_rate: float = 0.001,
                 gradient_norm_clip: float = None,
                 mode: str = "alternative",
                 name: str = "GNSEstimator"):
        """
        Args:
            batch_size_ratio: the portion of a batch to be used as a "smaller"
                batch. In theory, another smaller batch should be sampled *independently*
                from the data distribution. However, for simplicity, this estimator
                samples the smaller batch from a batch and uses the remaining as
                the larger batch. So this ratio should be small (<0.5). If the
                ratio is too small, the calculated smaller batch size will be
                clipped at 1.
            update_rate: the update rate for computing moving averages of the
                quantities needed by GNS. Generally, a smaller value (slower update)
                makes the estimated GNS more biased (because quantities at different
                training steps are averaged) while a larger value (quicker
                update) makes it have more variances.
            gradient_norm_clip: a clipping value for global gradient norm. If
                None, no clipping is performed. Usually, a clipping value is
                required for a stable GNS estimate. Depending on how stable the GNS
                is estimated, this value could also suggest a clipping norm for
                the optimizer.
            mode: either "paper" or "alternative". "paper" uses the calculation
                in the paper. "alternative" is the default mode as its calculation
                is easier to understand.
            name:
        """
        super().__init__()
        self._name = name
        assert mode in ["paper", "alternative"]
        if mode == "paper":
            assert 0 < batch_size_ratio < 0.5
        else:
            batch_size_ratio = 0.5
        self._mode = mode
        self._batch_size_ratio = batch_size_ratio
        self._grad_norm_clip = gradient_norm_clip
        self._gradient_norm_averager = ScalarEMAverager(
            update_rate=update_rate)
        self._var_trace_averager = ScalarEMAverager(update_rate=update_rate)
        self.register_buffer('_last_valid_gns', torch.zeros(()))

    def _calculate_gradient_norm(self, loss: torch.Tensor,
                                 tensors: alf.nest.NestedTensor):
        grads = alf.nest.utils.grad(tensors, loss.mean(), retain_graph=True)
        if self._grad_norm_clip is not None:
            grads, _ = clip_by_global_norm(grads, self._grad_norm_clip)
        norm2 = global_norm(grads)
        grads = torch.cat([g.reshape(-1) for g in alf.nest.flatten(grads)])
        return norm2**2, grads

    def forward(self, loss: torch.Tensor, tensors: alf.nest.NestedTensor):
        """Given a loss tensor and a nest of tensors, return the estimated GNS.

        Args:
            loss: a loss tensor *before* taking the mean. Each entry of the tensor
                represents an individual loss on a single training sample. Ideally,
                the samples used for computing these losses should be sampled *with*
                replacement independently. The loss can have a shape of either
                ``[T,B]`` or ``[B]``. The estimate will be more stable if ``B``
                is large and the batch could represent samples from the data
                distribution well.
            tensors: a nest of tensors whose gradients are considered

        Returns:
            gns: the estimated gradient noise scale (a scalar). A smaller value
                means more effective grad steps.
        """
        assert loss.ndim in [1, 2], "loss must be a rank-1 or -2 tensor!"

        B = loss.shape[-1]
        shuffled_loss = loss[..., torch.randperm(B)]

        b = max(1, int(B * self._batch_size_ratio))
        B -= b

        B_norm2, B_grads = self._calculate_gradient_norm(
            shuffled_loss[..., b:], tensors)
        b_norm2, b_grads = self._calculate_gradient_norm(
            shuffled_loss[..., :b], tensors)

        if self._mode == "paper":
            gradient_norm = (B * B_norm2 - b * b_norm2) / (B - b)
            var_trace = (b_norm2 - B_norm2) / (1. / b - 1. / B)
        else:
            assert B == b, "Check if the batch size is even!"
            var_trace = (B_norm2 + b_norm2) / 2.
            gradient_norm = (B_grads * b_grads).sum()

        avg_grad_norm = self._gradient_norm_averager.average(gradient_norm)
        avg_var_trace = self._var_trace_averager.average(var_trace)
        simple_noise_scale = avg_var_trace / (avg_grad_norm + 1e-8)

        if self._mode == "alternative":
            simple_noise_scale = (simple_noise_scale - 1) * B

        if simple_noise_scale < 0:
            # In theory GNS should be non-negative. If the current estimate is
            # negative, then we simply reuse the last estimate.
            simple_noise_scale = self._last_valid_gns
        else:
            self._last_valid_gns = torch.clone(simple_noise_scale)

        return simple_noise_scale.detach()
