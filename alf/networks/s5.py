# Copyright (c) 2023 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

from typing import Callable, Tuple
import math
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import alf
import alf.layers
from alf.initializers import variance_scaling_init
from alf.networks import Network
"""
Implement the S5 networks described in

J. T.H. Smith et. al. Simplified State Space Layers for Sequence Modeling
https://openreview.net/forum?id=Ai8Hw3AXqks

The original JAX implementation is at https://github.com/lindermanlab/S5


The following modules are implemented:

- S5SSM: corresponds to s5.ssm.S5SSM. The core diagonal state space model
- S5Block: corresponds to s5.layers.SequenceLayer. A
- create_stacked_s5_encoder: corresponds to s5.seq_model.StackedEncoderModel

An example of sMNIST training is in s5_train.py
"""


def _view_as_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: real tensor with shape [..., N]
    Returns:
        complex tensor with shape [..., N // 2]
    """
    return torch.view_as_complex(x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2))


def _view_as_real(x):
    """
    Args:
        x: complex tensor with shape [..., N]
    Returns:
        real tensor with shape [..., 2*N]
    """
    return torch.view_as_real(x).reshape(*x.shape[:-1], x.shape[-1] * 2)


def _as_real_to_complex_matrix(x: torch.Tensor) -> torch.Tensor:
    """

    .. code-block:: python

        B = _as_real_to_complex(A)
        assert (B @ x == _view_as_real(A @ x.to(B.dtype))).all()

    Args:
        x (torch.Tensor): complex matrix with shape [M, N]
    Returns:
        torch.Tensor: real matrix with shape [2*M, N]

    """
    return torch.view_as_real(x).mT.reshape(2 * x.shape[0], x.shape[1])


def _as_complex_to_real_matrix(x: torch.Tensor) -> torch.Tensor:
    """

    .. code-block:: python

        B = _as_complex_to_real_matrix(A)
        assert (B @ _view_as_real(x) == (A @ x).real).all()

    Args:
        x: complex matrix with shape [M, N]
    Returns:
        real matrix with shape [M, 2*N]
    """
    return _view_as_real(x.conj().resolve_conj())


class S5SSM(Network):
    r"""SSM with diagonal state transition matrix.

    Corresponds to s5.ssm.S5SSM

    The model is defined as the following in continuous time:

    .. math::

        ds/dt = As + Bu
        A = V \Lambda V^{-1}
        ds/dt = V \Lambda V^{-1}s + Bu
        dV^{-1}s/dt = \Lambda V^{-1}s + V^{-1}Bu
        y = C x + D u
        y = CV V^{-1}s + Du

    where :math:`s` is the state, :math:`u` is the input, :math:`y` is the output,
    :math:`A` is the state transition matrix, :math:`B` is the input matrix,
    :math:`C` is the output matrix, :math:`D` is the feedforward matrix,
    :math:`\Lambda` is the diagonal matrix of eigenvalues of :math:`A`,
    and :math:`V` is the matrix of eigenvectors of :math:`A`.

    The discrete time version is:

    .. math::

        x_t = B * input_t
        s_t = \Lambda * s_{t-1} + x_t
        output_t = C s_t + D * input_t

    Note: this implementation corresponds to (C_init='lecun_normal' and conj_sym=True in s5.ssm.S5SSM)

    Args:
        data_dim: dimension of input (H)
        state_dim: dimension of state (2*P)
        dt_min: minimum value to draw timescale values from when initializing log_step
        dt_max: maximum value to draw timescale values from when initializing log_step
        step_rescale: allows for uniformly changing the timescale parameter, e.g. after training
            on a different resolution for the speech commands benchmark

    """

    def __init__(self,
                 data_dim,
                 state_dim,
                 num_blocks,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 step_rescale: float = 1.0,
                 name="S5"):
        super().__init__(
            input_tensor_spec=alf.TensorSpec((data_dim, )),
            state_spec=alf.TensorSpec((state_dim, )),
            name=name)
        assert state_dim % 2 == 0
        assert state_dim // 2 % num_blocks == 0
        cstate_dim = state_dim // 2
        self._data_dim = data_dim
        self._state_dim = state_dim
        self._B = nn.Parameter(torch.zeros((cstate_dim, data_dim, 2)))
        self._Lambda = nn.Parameter(torch.zeros((cstate_dim, 2)))
        self._C = nn.Parameter(torch.zeros((data_dim, state_dim)))
        self._log_step = nn.Parameter(torch.zeros(cstate_dim))
        self._D = nn.Parameter(torch.zeros(data_dim))
        self._step_rescale = step_rescale
        self._dt_min = dt_min
        self._dt_max = dt_max
        self._num_blocks = num_blocks
        self.reset()

    @property
    def data_dim(self):
        return self._data_dim

    def reset(self):
        """

        """
        block_size = self._state_dim // 2 // self._num_blocks
        Lambda, _, _, V, _ = make_DPLR_HiPPO(2 * block_size)
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Lambda = np.tile(Lambda, self._num_blocks)
        Vc = V.conj().T
        # V: complex [2 * cstate_dim, cstate_dim]
        V = scipy.linalg.block_diag(*([V] * self._num_blocks))
        # Vinv: complex [cstate_dim, 2 * cstate_dim]
        Vinv = scipy.linalg.block_diag(*([Vc] * self._num_blocks))

        Lambda = torch.as_tensor(Lambda).to(torch.complex64)
        V = torch.as_tensor(V).to(torch.complex64)
        Vinv = torch.as_tensor(Vinv).to(torch.complex64)

        self._Lambda.data.copy_(torch.view_as_real(Lambda))

        B = torch.zeros((self._state_dim, self._data_dim))
        variance_scaling_init(B)
        B = Vinv @ B.to(Vinv.dtype)
        self._B.data.copy_(torch.view_as_real(B))

        C = torch.zeros((self._data_dim, 2 * self._state_dim))
        variance_scaling_init(C)
        C = _view_as_complex(C) @ V
        self._C.data.copy_(_as_complex_to_real_matrix(C))

        torch.nn.init.normal_(self._D.data)

        self._log_step.data.uniform_(
            math.log(self._dt_min), math.log(self._dt_max))

    def discretize(self):
        """
        Returns:
            Lambda_bar (torch.Tensor): complex tensor with shape [state_dim // 2]
            B_bar (torch.Tensor): real tensor with shape [state_dim, data_dim]
        """
        step = self._step_rescale * self._log_step.exp()
        self._Lambda.data[:, 0].clip_(
            max=-1e-4)  # clip real part to be negative
        Lambda = torch.view_as_complex(self._Lambda)
        Lambda_bar = torch.exp(Lambda * step)
        B_tilde = torch.view_as_complex(self._B)
        B_bar = ((1 / Lambda) * (Lambda_bar - 1))[:, None] * B_tilde
        B_bar = _as_real_to_complex_matrix(B_bar)
        return Lambda_bar, B_bar

    def forward(self, input, state):
        if input.ndim == 3:
            return self.forward_sequence(input, state)
        Lambda_bar, B_bar = self.discretize()
        cstate = _view_as_complex(state)
        state = _view_as_real(Lambda_bar * cstate) + input @ B_bar.T
        output = 2 * state @ self._C.T
        output = torch.addcmul(output, self._D, input)
        return output, state

    def forward_sequence(self, inputs, state):
        r"""

        Args:
            inputs (torch.Tensor): shape is [T, B, data_dim]
            state (torch.Tensor): :math:`s_{-1}`. The shape is [batch_size, state_dim]
        Returns:
            outputs (torch.Tensor): shape is [T, B, data_dim]
            state (torch.Tensor): :math:`s_{T-1}`. The shape is [batch_size, state_dim]
        """
        Lambda_bar, B_bar = self.discretize()
        x = _view_as_complex(inputs @ B_bar.T)
        cstate = _view_as_complex(state)
        cstates = diag_ssm_forward(cstate, x, Lambda_bar)
        outputs = 2 * _view_as_real(cstates) @ self._C.T
        outputs = torch.addcmul(outputs, self._D, inputs)
        return outputs, _view_as_real(cstates[-1])


@triton.jit
def diag_ssm_forward_kernel(s_ptr, x_ptr, lambda_ptr, y_ptr, length,
                            batch_size, dim, BLOCK_SIZE: tl.constexpr):
    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim
    s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)
    for t in range(length):
        offsets = t * batch_size * dim + col_offsets
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        s = s * Lambda + x
        tl.store(y_ptr + offsets, s, mask=mask)


@triton.jit
def diag_ssm_backward_kernel(s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr,
                             grad_lambda_ptr, grad_y_ptr, length, batch_size,
                             dim, BLOCK_SIZE: tl.constexpr):

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    Lambda = tl.load(lambda_ptr + col_offsets % dim, mask=mask, other=0)

    # Initialize gradients to zero
    grad_s = tl.zeros_like(Lambda)
    grad_Lambda = tl.zeros_like(Lambda)

    for i in range(length):
        # range(length - 1, -1, -1) is not correctly implemented by Triton
        t = length - 1 - i
        offsets = t * batch_size * dim + col_offsets

        grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
        if t > 0:
            s = tl.load(y_ptr + offsets - batch_size * dim, mask=mask, other=0)
        else:
            s = tl.load(s_ptr + col_offsets, mask=mask, other=0)

        grad_s = grad_y + grad_s
        grad_x = grad_s
        grad_Lambda += grad_s * s
        grad_s = grad_s * Lambda

        tl.store(grad_x_ptr + offsets, grad_x, mask=mask)

    tl.store(grad_s_ptr + col_offsets, grad_s, mask=mask)
    tl.store(grad_lambda_ptr + col_offsets, grad_Lambda, mask=mask)


@triton.jit
def diag_ssm_forward_kernel_complex(s_ptr, x_ptr, y_ptr, lambda_ptr, length,
                                    batch_size, dim, BLOCK_SIZE: tl.constexpr):
    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    # Load real and imaginary parts of 's' and 'Lambda'
    s_real = tl.load(s_ptr + col_offsets * 2, mask=mask, other=0)
    s_imag = tl.load(s_ptr + col_offsets * 2 + 1, mask=mask, other=0)
    lambda_real = tl.load(
        lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(
        lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    for t in range(length):
        offsets = (t * batch_size * dim + col_offsets) * 2
        # Load real and imaginary parts of 'x'
        x_real = tl.load(x_ptr + offsets, mask=mask, other=0)
        x_imag = tl.load(x_ptr + offsets + 1, mask=mask, other=0)

        # Complex multiplication and addition
        new_s_real = s_real * lambda_real - s_imag * lambda_imag + x_real
        new_s_imag = s_real * lambda_imag + s_imag * lambda_real + x_imag

        # Store the updated real and imaginary parts
        tl.store(y_ptr + offsets, new_s_real, mask=mask)
        tl.store(y_ptr + offsets + 1, new_s_imag, mask=mask)

        # Update s for the next iteration
        s_real, s_imag = new_s_real, new_s_imag


@triton.jit
def diag_ssm_backward_kernel_complex(
        s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr,
        grad_y_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr):

    # autograd for complex numbers calculates \partial f / \partial z^*
    # so we need to take conjugate during the calculation.
    # https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    # Load real and imaginary parts of 's' and 'Lambda'
    lambda_real = tl.load(
        lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(
        lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    # Initialize gradients to zero
    grad_s_real = tl.zeros_like(lambda_real)
    grad_s_imag = tl.zeros_like(lambda_imag)
    grad_lambda_real = tl.zeros_like(lambda_real)
    grad_lambda_imag = tl.zeros_like(lambda_imag)

    for i in range(length):
        # range(length - 1, -1, -1) is not correctly implemented by Triton
        t = length - 1 - i
        offsets = (t * batch_size * dim + col_offsets) * 2

        grad_y_real = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
        grad_y_imag = -tl.load(grad_y_ptr + offsets + 1, mask=mask, other=0)
        if t > 0:
            s_real = tl.load(
                y_ptr + offsets - 2 * batch_size * dim, mask=mask, other=0)
            s_imag = tl.load(
                y_ptr + offsets - 2 * batch_size * dim + 1, mask=mask, other=0)
        else:
            s_real = tl.load(s_ptr + 2 * col_offsets, mask=mask, other=0)
            s_imag = tl.load(s_ptr + 2 * col_offsets + 1, mask=mask, other=0)

        grad_s_real = grad_y_real + grad_s_real
        grad_s_imag = grad_y_imag + grad_s_imag
        grad_x_real = grad_s_real
        grad_x_imag = grad_s_imag
        grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
        grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
        grad_s_real = grad_x_real * lambda_real - grad_x_imag * lambda_imag
        grad_s_imag = grad_x_real * lambda_imag + grad_x_imag * lambda_real

        tl.store(grad_x_ptr + offsets, grad_x_real, mask=mask)
        tl.store(grad_x_ptr + offsets + 1, -grad_x_imag, mask=mask)

    # Store the final gradients for s and Lambda
    tl.store(grad_s_ptr + col_offsets * 2, grad_s_real, mask=mask)
    tl.store(grad_s_ptr + col_offsets * 2 + 1, -grad_s_imag, mask=mask)
    tl.store(grad_lambda_ptr + col_offsets * 2, grad_lambda_real, mask=mask)
    tl.store(
        grad_lambda_ptr + col_offsets * 2 + 1, -grad_lambda_imag, mask=mask)


class _ssm_forward(torch.autograd.Function):
    BLOCK_SIZE = 128

    @staticmethod
    def forward(ctx, s, x, Lambda):
        assert s.is_contiguous() and x.is_contiguous(
        ) and Lambda.is_contiguous()
        length, batch_size, dim = x.shape
        n = batch_size * dim
        y = torch.zeros_like(x)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )

        if Lambda.dtype == torch.complex64:
            diag_ssm_forward_kernel_complex[grid](torch.view_as_real(s),
                                                  torch.view_as_real(x),
                                                  torch.view_as_real(y),
                                                  torch.view_as_real(Lambda),
                                                  length, batch_size, dim,
                                                  _ssm_forward.BLOCK_SIZE)
        else:
            diag_ssm_forward_kernel[grid](s, x, Lambda, y, length, batch_size,
                                          dim, _ssm_forward.BLOCK_SIZE)
        ctx.save_for_backward(s, y, Lambda)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        s, y, Lambda = ctx.saved_tensors
        length, batch_size, dim = y.shape
        grad_y = grad_y.contiguous()
        n = batch_size * dim
        grad_s = torch.empty_like(s)
        grad_x = torch.empty_like(grad_y)
        grad_lambda = torch.empty_like(s)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
        if Lambda.dtype == torch.complex64:
            diag_ssm_backward_kernel_complex[grid](
                torch.view_as_real(s), torch.view_as_real(Lambda),
                torch.view_as_real(y), torch.view_as_real(grad_s),
                torch.view_as_real(grad_x), torch.view_as_real(grad_lambda),
                torch.view_as_real(grad_y), length, batch_size, dim,
                _ssm_forward.BLOCK_SIZE)
        else:
            diag_ssm_backward_kernel[grid](
                s, Lambda, y, grad_s, grad_x, grad_lambda, grad_y, length,
                batch_size, dim, _ssm_forward.BLOCK_SIZE)
        return grad_s, grad_x, grad_lambda.sum(dim=0)


diag_ssm_forward_triton = _ssm_forward.apply


def diag_ssm_forward(s, x, Lambda):
    r"""Diagonal SSM forward pass

    Calculate :math:`y_t = Lambda * y_{t-1} + x_t` for t > 0
    and :math:`y_0 = Lambda * s + x_0`

    Args:
        s (torch.Tensor): shape is [batch_size, state_dim]
        x (torch.Tensor): shape is [length, batch_size, data_dim]
        Lambda (torch.Tensor): shape is [state_dim]
    Returns:
        torch.Tensor: y in the above equation. The shape is
            [length, batch_size, state_dim]
    """
    if x.is_cuda:
        return diag_ssm_forward_triton(s, x, Lambda)
    else:
        return diag_ssm_forward_slow(s, x, Lambda)


def diag_ssm_forward_slow(s, x, Lambda):
    length = x.shape[0]
    cstates = []
    for i in range(length):
        s = torch.addcmul(x[i], Lambda, s)
        cstates.append(s)
    cstates = torch.stack(cstates)
    return cstates


class BatchNorm1dChannelLast(nn.BatchNorm1d):
    """Batch normalization layer for channel-last tensor
    """

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape is [T, B, C] or [B, C]
        """
        if x.ndim == 2:
            return super().forward(x)
        elif x.ndim == 3:
            x = x.transpose(1, 2)
            x = super().forward(x)
            return x.transpose(1, 2)
        else:
            raise ValueError("Only support 2d or 3d tensor")


class Dropout(nn.Module):
    """Dropout layer supporting shared dropout mask across given dimension

    Args:
        dropout (float): dropout rate
        broadcast_dims (tuple): dimensions to share dropout mask
    """

    def __init__(self, dropout: float, broadcast_dims: Tuple[int, ...] = ()):
        super().__init__()
        self._broadcast_dims = broadcast_dims
        self._dropout = dropout

    def forward(self, x):
        if self.training:
            mask_shape = list(x.shape)
            for d in self._broadcast_dims:
                mask_shape[d] = 1
            mask = torch.rand(mask_shape, device=x.device) > self._dropout
            return x * mask
        else:
            return x * (1 - self._dropout)


class S5Block(Network):
    """
    Corresponds to s5.layers.SequenceLayer

    Note: for sequence input, the input is assumed to be [T, B, C]

    """

    def __init__(self,
                 ssm_ctor: Callable[..., nn.Module],
                 dropout: float,
                 activation: str = "half_glu2",
                 prenorm: bool = False,
                 batchnorm: bool = False,
                 bn_momentum: float = 0.10,
                 step_rescale: float = 1.0,
                 name="S5Block"):

        ssm = ssm_ctor(step_rescale=step_rescale)
        d_model = ssm.data_dim
        super().__init__(
            input_tensor_spec=alf.TensorSpec((d_model, )),
            state_spec=ssm.state_spec,
            name=name)

        if activation in ["full_glu"]:
            self._out1 = alf.layers.FC(d_model, d_model)
            self._out2 = alf.layers.FC(d_model, d_model)
        elif activation in ["half_glu1", "half_glu2"]:
            self._out2 = alf.layers.FC(d_model, d_model)

        if batchnorm:
            self._norm = BatchNorm1dChannelLast(
                num_features=d_model, momentum=bn_momentum)
        else:
            self._norm = nn.LayerNorm(d_model)

        # s5.layers.SequenceLayer uses shared dropout mask across time dimension
        # torch.nn.Dropout does not support this
        self._drop3d = Dropout(dropout, broadcast_dims=(0, ))
        self._drop2d = nn.Dropout(dropout)

        self._ssm = ssm
        self._activation = activation
        self._prenorm = prenorm

    @property
    def d_model(self):
        return self._ssm.data_dim

    def forward(self, input, state):
        """
        Args:
            input (torch.Tensor): shape is [T, B, C] or [B, C]
            state (torch.Tensor): shape is [B, state_dim]
        """
        if self._prenorm:
            x = self._norm(input)
        else:
            x = input
        x, state = self._ssm(x, state)
        if x.ndim == 3:
            drop = self._drop3d
        else:
            drop = self._drop2d

        if self._activation == "full_glu":
            x = drop(nn.gelu(x))
            x = self._out1(x) * F.sigmoid(self._out2(x))
            x = drop(x)
        elif self._activation == "half_glu1":
            x = drop(F.gelu(x))
            x = x * F.sigmoid(self._out2(x))
            x = drop(x)
        elif self._activation == "half_glu2":
            # Only apply GELU to the gate input
            x1 = drop(F.gelu(x))
            x = x * F.sigmoid(self._out2(x1))
            x = drop(x)
        elif self._activation == "gelu":
            x = drop(F.gelu(x))
        else:
            raise NotImplementedError("Activation: {} not implemented".format(
                self.activation))

        x = input + x
        if not self._prenorm:
            x = self._norm(x)
        return x, state


def create_stacked_s5_encoder(data_dim,
                              ssm_ctor: Callable[..., nn.Module],
                              num_layers,
                              dropout: float = 0.0,
                              activation: str = "half_glu2",
                              prenorm: bool = False,
                              batchnorm: bool = False,
                              bn_momentum: float = 0.10,
                              step_rescale=1.0,
                              name="StackedS5Encoder"):
    """
    Corresponds to s5.seq_model.StackedEncoderModel

    """
    ssms = [
        S5Block(
            ssm_ctor,
            dropout=dropout,
            activation=activation,
            prenorm=prenorm,
            batchnorm=batchnorm,
            bn_momentum=bn_momentum,
            step_rescale=step_rescale) for _ in range(num_layers)
    ]
    encoder = alf.layers.FC(data_dim, ssms[0].d_model)
    return alf.networks.Sequential(encoder, *ssms, name=name)


def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = np.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig
