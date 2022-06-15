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
from alf.utils.tensor_utils import global_norm


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
    """Implement the simple Gradient Noise Scale estimator as detailed in
    Appendix A, "An Empirical Model of Large-Batch Training", McCandlish et al.,
    `arXiv <https://arxiv.org/pdf/1812.06162.pdf>`_, 2018.

    Generally, GNS indicates the noise-to-signal value of SGD. The authors suggest
    that we should choose a batch size close to GNS in order to average out the
    noise in the gradient.

    .. note::

        You can turn on this estimator in ``TrainerConfig``. However, this will
        increase the back-propagation overhead.
    """

    def __init__(self,
                 batch_size_ratio: float = 0.1,
                 update_rate: float = 0.01,
                 gradient_norm_clip: float = 10.,
                 name: str = "GNSEstimator"):
        """
        Args:
            batch_size_ratio: the portion of a batch to be used as a "smaller"
                batch. In theory, another smaller batch should be sampled *independently*
                from the data distribution. However, for simplicity, this estimator
                samples the smaller batch from a batch and uses the remaining as
                the larger batch. So this ratio should be small (<0.5).
            update_rate: the update rate for computing moving averages of the
                quantities needed by GNS. Generally, a smaller value (slower update)
                makes the estimated GNS more biased (because quantities at different
                training steps are averaged) while a larger value (quicker
                update) makes it have more variances.
            gradient_norm_clip: a clipping value for gradient norm outliers
        """
        super().__init__()
        self._name = name
        assert 0 < batch_size_ratio < 0.5
        self._batch_size_ratio = batch_size_ratio
        self._grad_norm_clip = gradient_norm_clip
        self._gradient_norm_averager = ScalarEMAverager(
            update_rate=update_rate)
        self._var_trace_averager = ScalarEMAverager(update_rate=update_rate)

    def _calculate_gradient_norm(self, loss: torch.Tensor,
                                 tensors: alf.nest.NestedTensor):
        grads = alf.nest.utils.grad(tensors, loss.mean(), retain_graph=True)
        norm2 = torch.clamp(global_norm(grads), max=self._grad_norm_clip)
        return norm2**2

    def forward(self, loss: torch.Tensor, tensors: alf.nest.NestedTensor):
        """Given a loss tensor and a nest of tensors, return the estimated GNS.

        Args:
            loss: a loss tensor *before* taking the mean. Each entry of the tensor
                represents an individual loss on a single training sample. Ideally,
                the samples used for computing these losses should be sampled *with*
                replacement independently. So on-policy algorithms might violate
                this assumption if the unroll length is large.
            tensors: a nest of tensors whose gradients are considered

        Returns:
            gns: the estimated gradient noise scale (a scalar). A smaller value
                means more effective grad steps.
        """
        assert loss.ndim > 0, "loss must be a batched tensor!"
        loss = loss.reshape(-1)
        B = loss.shape[0]
        shuffled_loss = loss[torch.randperm(B)]

        b = int(B * self._batch_size_ratio)
        B -= b

        B_norm2 = self._calculate_gradient_norm(shuffled_loss[b:], tensors)
        b_norm2 = self._calculate_gradient_norm(shuffled_loss[:b], tensors)

        gradient_norm = (B * B_norm2 - b * b_norm2) / (B - b)
        var_trace = (b_norm2 - B_norm2) / (1. / b - 1. / B)

        avg_grad_norm = self._gradient_norm_averager.average(gradient_norm)
        avg_var_trace = self._var_trace_averager.average(var_trace)
        simple_noise_scale = avg_var_trace / avg_grad_norm
        return simple_noise_scale
