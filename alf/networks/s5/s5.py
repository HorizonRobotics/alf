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

import alf
import alf.layers
from alf.initializers import variance_scaling_init
from alf.networks import Network

from .utils import make_DPLR_HiPPO, diag_ssm_forward
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
    and :math:`V` is the matrix of eigenvectors of :math:`A`. Note that we use
    diagonal :math:`D`.

    The discrete time version is:

    .. math::

        x_t = B * input_t
        s_t = \Lambda * s_{t-1} + x_t
        output_t = C s_t + D * input_t

    Note: this implementation corresponds to s5.ssm.S5SSM with the following settings:

        C_init='lecun_normal'
        bidirectional=False
        conj_sym=True
        clip_eigs=True
        discretization='zoh'

    Args:
        data_dim: dimension of input and output (H)
        state_dim: dimension of state (2*P)
        num_blocks: number of blocks (J). The state_dim must be divisible by 2 * num_blocks.
            This affect the initial values of B, Lambda, C and D.
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
        # dimension of the complex state
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
    """Corresponds to s5.layers.SequenceLayer.

    It performs the following computation:

    For prenorm=True: output = input + Activation(SSM(Norm(input)))

    For prenorm=False: output = Norm(input + Activation(SSM(input)))

    where Activation included dropout.

    Note: for sequence input, the input is assumed to be [T, B, C]

    Args:
        ssm_ctor: a callable that returns an instance of S5SSM, will be called
            as ``ssm_ctor(step_rescale=step_rescale)``
        dropout: dropout rate
        activation: one of "full_glu", "half_glu1", "half_glu2", "gelu"
        prenorm: whether to apply layer norm before the block
        batchnorm: True to use BatchNorm, False to use LayerNorm
        bn_momentum: momentum for BatchNorm
        step_rescale: allows for uniformly changing the timescale parameter,
            e.g. after training on a different resolution
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
