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

import numpy as np
import torch


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


try:
    import triton
    import triton.language as tl

    @triton.jit
    def diag_ssm_forward_kernel(s_ptr, x_ptr, lambda_ptr, y_ptr, length,
                                batch_size, dim, BLOCK_SIZE: tl.constexpr):
        """
        Args:
            s_ptr: [batch_size, dim]
            x_ptr: [length, batch_size, dim]
            lambda_ptr: [dim]
            y_ptr: [length, batch_size, dim]
        """
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
    def diag_ssm_backward_kernel(
            s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr,
            grad_y_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr):
        """
        Args:
            s_ptr: [batch_size, dim]
            lambda_ptr: [dim]
            y_ptr: [length, batch_size, dim]
            grad_s_ptr: [batch_size, dim]
            grad_x_ptr: [length, batch_size, dim]
            grad_lambda_ptr: [batch_size, dim]. The shape is different from ``grad_s_ptr``
                because we need the caller to sum the gradients after the kernel finish.
                It's more complicated to sum the gradients inside the kernel.
            grad_y_ptr: [length, batch_size, dim]
        """

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
                s = tl.load(
                    y_ptr + offsets - batch_size * dim, mask=mask, other=0)
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
    def diag_ssm_forward_kernel_complex(s_ptr, x_ptr, y_ptr, lambda_ptr,
                                        length, batch_size, dim,
                                        BLOCK_SIZE: tl.constexpr):
        """
        Args:
            s_ptr: [batch_size, dim, 2]
            x_ptr: [length, batch_size, dim, 2]
            lambda_ptr: [dim, 2]
            y_ptr: [length, batch_size, dim, 2]
        """
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
        """
        Args:
            s_ptr: [batch_size, dim, 2]
            lambda_ptr: [dim, 2]
            y_ptr: [length, batch_size, dim, 2]
            grad_s_ptr: [batch_size, dim, 2]
            grad_x_ptr: [length, batch_size, dim, 2]
            grad_lambda_ptr: [batch_size, dim, 2]. The shape is different from ``grad_s_ptr``
                because we need the caller to sum the gradients after the kernel finish.
                It's more complicated to sum the gradients inside the kernel.
            grad_y_ptr: [length, batch_size, dim, 2]
        """

        # autograd for complex numbers calculates \partial f / \partial z^*
        # so we need to take conjugate during the calculation.
        # https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers
        # So in the following code, when we load/store the imaginary part of a gradient,
        # we need to negate it.

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
            grad_y_imag = -tl.load(
                grad_y_ptr + offsets + 1, mask=mask, other=0)
            if t > 0:
                s_real = tl.load(
                    y_ptr + offsets - 2 * batch_size * dim, mask=mask, other=0)
                s_imag = tl.load(
                    y_ptr + offsets - 2 * batch_size * dim + 1,
                    mask=mask,
                    other=0)
            else:
                s_real = tl.load(s_ptr + 2 * col_offsets, mask=mask, other=0)
                s_imag = tl.load(
                    s_ptr + 2 * col_offsets + 1, mask=mask, other=0)

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
        tl.store(
            grad_lambda_ptr + col_offsets * 2, grad_lambda_real, mask=mask)
        tl.store(
            grad_lambda_ptr + col_offsets * 2 + 1,
            -grad_lambda_imag,
            mask=mask)

    class _ssm_forward(torch.autograd.Function):
        # TODO use @triton.autotune to choose the best BLOCK_SIZE
        # BLOCK_SIZE = 128 seems work well for 3090
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
                diag_ssm_forward_kernel_complex[grid](
                    torch.view_as_real(s), torch.view_as_real(x),
                    torch.view_as_real(y), torch.view_as_real(Lambda), length,
                    batch_size, dim, _ssm_forward.BLOCK_SIZE)
            elif Lambda.dtype.is_floating_point:
                diag_ssm_forward_kernel[grid](s, x, Lambda, y, length,
                                              batch_size, dim,
                                              _ssm_forward.BLOCK_SIZE)
            else:
                raise ValueError("Unsupported dtype: %s" % Lambda.dtype)
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
            # Here grad_lambda stores the gradients of Lambda for each sample
            # in the batch. We will sum them up after the kernel finishes.
            grad_lambda = torch.empty_like(s)
            grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
            if Lambda.dtype == torch.complex64:
                diag_ssm_backward_kernel_complex[grid](
                    torch.view_as_real(s), torch.view_as_real(Lambda),
                    torch.view_as_real(y), torch.view_as_real(grad_s),
                    torch.view_as_real(grad_x),
                    torch.view_as_real(grad_lambda),
                    torch.view_as_real(grad_y), length, batch_size, dim,
                    _ssm_forward.BLOCK_SIZE)
            else:
                diag_ssm_backward_kernel[grid](
                    s, Lambda, y, grad_s, grad_x, grad_lambda, grad_y, length,
                    batch_size, dim, _ssm_forward.BLOCK_SIZE)
            return grad_s, grad_x, grad_lambda.sum(dim=0)

    diag_ssm_forward_triton = _ssm_forward.apply

except ImportError:
    from alf.utils.common import warning_once

    def diag_ssm_forward_triton(s, x, Lambda):
        warning_once("Triton is not installed. Using slow diag_ssm_forward.")
        return diag_ssm_forward_slow(s, x, Lambda)


def diag_ssm_forward(s, x, Lambda):
    r"""Diagonal SSM forward pass

    Calculate :math:`y_t = Lambda * y_{t-1} + x_t` for t > 0
    and :math:`y_0 = Lambda * s + x_0`

    Args:
        s (torch.Tensor): shape is [batch_size, state_dim]
        x (torch.Tensor): shape is [length, batch_size, state_dim]
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
