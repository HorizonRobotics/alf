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
        dtype = next(m.parameters()).dtype
        self._merged = False

        # LoRA weight mats
        self._wA = None
        self._wB = None  # optional; only present if m.weight is huge
        nrows, ncols = self._decompose_weight_dims(m)
        # original size: nrows * ncols
        # adapter size: (nrows + ncols) * r
        # need low-rank adaptation: (nrows + ncols) * r < nrows * ncols
        if rank > ncols * nrows / (nrows + ncols):
            self._wA = nn.Parameter(torch.zeros(nrows, ncols, dtype=dtype))
        else:
            self._wA = nn.Parameter(torch.zeros(rank, ncols, dtype=dtype))
            self._wB = nn.Parameter(torch.zeros(nrows, rank, dtype=dtype))
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
        return w * self.scaling

    @property
    def scaling(self) -> float:
        scaling = self._alpha
        if self._wB is not None:
            # If wB is used, wB @ wA will increase the output magnitude. LoRA
            # compensates this by dividing the result by ``self._r``.
            scaling /= self._r
        return scaling

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
class EmbeddingAdapter(LoRA):
    """Adapter for embedding layers.
    """

    @classmethod
    def can_adapt(cls, m: nn.Module) -> bool:
        return isinstance(m, nn.Embedding)

    @classmethod
    def _decompose_weight_dims(cls, m: nn.Module):
        return m.num_embeddings, m.embedding_dim

    def forward(self, input):
        m = self._m[0]
        if self._wB is not None:
            embedding_table = self._wB
        else:
            embedding_table = self._wA
        input = F.embedding(input, embedding_table, m.padding_idx, m.max_norm,
                            m.norm_type, m.scale_grad_by_freq, m.sparse)
        if self._wB is not None:
            input = input @ self._wA
        return input * self.scaling


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
        input = input @ self._wA.t()
        if self._wB is not None:
            input = input @ self._wB.t()
        return input * self.scaling


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
        # conv2d weight has a shape: ``(out_channels, in_channels // groups, *kernel_size)``
        nrows = m.out_channels * kernel_size[0]
        ncols = m.in_channels // m.groups * kernel_size[1]
        return nrows, ncols

    def forward(self, input):
        m = self._m[0]
        if isinstance(m.padding, str) or m.groups > 1 or self._wB is None:
            # Three scenarios when two-stage convolution is difficult:
            # 1. m.groups > 1: r has to be divisible by m.groups in order to
            #    preserve the correct input-output mapping
            # 2. m.padding is a string: torch will compute paddings on the fly,
            #    so we won't know its values in advance.
            # 3. low-rank decomposition is not available
            return F.conv2d(
                input,
                self._adapter_weight(),
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                groups=m.groups)
        else:
            input = F.conv2d(
                input,
                self._wA.reshape(self._r, m.in_channels, 1, m.kernel_size[1]),
                stride=(1, m.stride[1]),
                padding=(0, m.padding[1]),
                dilation=(1, m.dilation[1]))
            output = F.conv2d(
                input,
                self._wB.reshape(m.out_channels, self._r, m.kernel_size[0], 1),
                stride=(m.stride[0], 1),
                padding=(m.padding[0], 0),
                dilation=(m.dilation[0], 1))
            return output * self.scaling

    def _adapter_weight(self):
        m = self._m[0]
        if self._wB is None:
            w = self._wA.reshape(m.weight.shape)
        else:
            # This weight tensor has to be consistent with the two-stage conv in
            # ``self.forward()``
            wa = self._wA.reshape(self._r, m.in_channels // m.groups,
                                  m.kernel_size[1])
            wb = self._wB.reshape(m.out_channels, self._r, m.kernel_size[0])
            w = torch.einsum('rik,org->oigk', wa, wb)
        return w * self.scaling
