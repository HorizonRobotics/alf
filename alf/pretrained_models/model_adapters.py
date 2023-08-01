# Copyright (c) 2023 Horizon Robotics and Hobot Contributors. All Rights Reserved.
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

import types
import copy
from functools import partial
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import alf


class LoRA(nn.Module):
    r"""Base class for LoRA (Low-Rank Adaptation).

    For any model layer expressed as a matrix multiplication of the form
    :math:`h=W_0x`, it performs a reparameterization such that:

    .. math::

        h = W_0x + \frac{\alpha}{r} BAx

    where :math:`A\in\mathbb{R}^{r\times k}` and :math:`B\in\mathbb{R}^{d\times r}`
    are the decomposition matrices and :math:`r` is the low-dim rank of the
    decomposition.

    The original LoRA paper doesn't adapt biases: "We leave the empirical
    investigation of adapting the MLP layers, LayerNorm layers, and biases
    to a future work."

    If a module has a weight matrix with dimensionality greater than 2 (e.g., conv),
    we might need to first reinterpret it as a 2D matrix ``W_0``. This reinterpretation
    will differ for different modules.
    """

    def __init__(self,
                 m: nn.Module,
                 rank: int = 16,
                 weight: float = 1.,
                 name: str = 'LoRA'):
        """
        Args:
            m: the module to be adapted
            rank: the low rank
            weight: weight for the low-rank weight matrix
        """
        super().__init__()

        for para in m.parameters():
            para.requires_grad = False

        self._name = name
        self._r = rank
        self._alpha = weight
        # place ``m`` in a list to avoid pytorch recursive tracking
        self._m = [m]
        self._merged = False

        # LoRA weight mats
        self._wA = None
        self._wB = None  # optional; only present if m.weight is huge
        nrows, ncols = self._decompose_weight_dims(m)
        # original size: nrows * ncols
        # adapter size: (nrows + ncols) * r
        # need low-rank adaptation: (nrows + ncols) * r < nrows * ncols
        if rank > ncols * nrows / (nrows + ncols):
            self._wA = nn.Parameter(torch.Tensor(nrows, ncols))
        else:
            self._wA = nn.Parameter(torch.Tensor(rank, ncols))
            self._wB = nn.Parameter(torch.Tensor(nrows, rank))
        self.reset_parameters()
        self._adapt(m)

    def reset_parameters(self):
        if self._wB is None:
            nn.init.zeros_(self._wA)
        else:
            nn.init.xavier_uniform_(self._wA)
            nn.init.zeros_(self._wB)

    def _adapter_weight(self):
        w = self._wA
        if self._wB is not None:
            w = self._wB @ w
        w = w.reshape(self._m[0].weight.shape)
        scaling = self._alpha
        if self._wB is not None:
            # If wB is used, wB @ wA will introduce a scale of self._r, so we
            # need to scale the weight back.
            scaling /= self._r
        return scaling * w

    def _adapt(self, m: nn.Module):
        """Adapt a module *in place*.

        After this, ``m.forward()`` will be computed with the adapter weights.
        """
        assert not hasattr(m, '_forward0'), (
            "The module has already been adapted! You need to first remove the "
            "adapter.")
        forward0 = m.forward
        adapter_forward = self.forward

        def _unmerged_forward(self, input):
            """For unmerged forward, the base model and adapter forward separately,
            and their results are weighted combined.

            This forward way should be used for training.
            """
            return forward0(input) + adapter_forward(input)

        m.forward = types.MethodType(_unmerged_forward, m)
        m._unmerged_forward = m.forward
        m._forward0 = forward0

    def merge(self):
        """Merge the adapter and base module weights.

        This operation will change the base module weight in place and restore
        its original ``forward()``. It should only be called in the inference mode.
        """
        if self._merged:
            return
        m = self._m[0]
        m.weight.data.add_(self._adapter_weight())
        m.forward = m._forward0
        self._merged = True

    def unmerge(self):
        """Unmerge the adapter weights. The base module is still adapted, and
        the adapter weights can be trained after unmerge.
        """
        if not self._merged:
            return
        m = self._m[0]
        m.weight.data.add_(-self._adapter_weight())
        m.forward = m._unmerged_forward
        self._merged = False

    def detach(self):
        """Detach from and recover the base module.

        Note that this operation is *irreversible*. Once detached, there is no
        way to add the adapter back to the module.

        We still keep the adapter weights after detachment.
        """
        if self._merged:
            self.unmerge()
        m = self._m[0]
        m.forward = m._forward0
        del m._forward0
        del m._unmerged_forward

    @classmethod
    def can_adapt(cls, m: nn.Module) -> bool:
        """Check if the adapter class can adapt a given module.
        """
        raise NotImplementedError()

    @classmethod
    def _decompose_weight_dims(cls, m: nn.Module) -> Tuple[int]:
        """Return two (reshaped) dimensions ``(nrows, ncols)`` so that
        ``nrows*ncols == np.prod(m.weight.shape)``.

        The reinterpreted shape will be used to create LoRA weights ``self._wA``
        and ``self._wB``.

        Returns:
            tuple: a pair of ints ``(nrows, ncols)``.
        """
        raise NotImplementedError()

    def forward(self, input):
        raise NotImplementedError()


@alf.configurable
class LinearAdapter(LoRA):
    """Adapter for linear layers.
    """

    @classmethod
    def can_adapt(cls, m: nn.Module) -> bool:
        return isinstance(m, nn.Linear)

    @classmethod
    def _decompose_weight_dims(cls, m: nn.Module):
        return m.out_features, m.in_features

    def forward(self, input):
        return F.linear(input, self._adapter_weight())


@alf.configurable
class Conv2dAdapter(LoRA):
    """Adapter for Conv2d layers.

    The most natural way of a LoRA decomposition for Conv2d is

    .. code-block:: python

        (rank, kernel_size[0] * kernel_size[1] * in_channels) x (out_channels, rank)

    However, since ``out_channels`` is usually small, this decomposition is not
    low-rank actually and it won't save much memory.

    We can first reinterpret the weight matrix as a shape of

    .. code-block:: python

        (out_channels * kernel_size[0], in_channels * kernel_size[1])

    to make the in- and out-dimensions balanced, and then decompose.
    """

    @classmethod
    def can_adapt(cls, m: nn.Module) -> bool:
        return isinstance(m, nn.Conv2d)

    @classmethod
    def _decompose_weight_dims(cls, m: nn.Module):
        kernel_size = m.kernel_size
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, ) * 2
        nrows = m.out_channels // m.groups * kernel_size[0]
        ncols = m.in_channels * kernel_size[1]
        return nrows, ncols

    def forward(self, input):
        m = self._m[0]
        return F.conv2d(
            input,
            self._adapter_weight(),
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups)
