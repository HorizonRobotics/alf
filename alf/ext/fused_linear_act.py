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

import torch
from typing import Any, Literal, Optional
from alf.utils.common import lazy_load_extension
import torch.nn.functional as F
import pathlib
import os
from .act_backward import act_backward

DIR = pathlib.Path(__file__).parent.absolute()
_ext = lazy_load_extension(name="fused_matmul_act",
                           sources=[os.path.join(DIR, "fused_matmul_act.cu")],
                           verbose=True)


class StaticState:
    workspace = {}
    workspace_size = 1024 * 1024 * 8
    bias_g = {}

    @classmethod
    def get(cls, name: str, device: torch.device) -> Any:
        idx = device.index if device.index is not None else 0
        if idx not in cls.workspace:
            cls.workspace[idx] = torch.empty((cls.workspace_size, ),
                                             dtype=torch.uint8,
                                             device=device).cuda(idx)
            cls.bias_g[idx] = torch.tensor([], dtype=torch.float16).cuda(idx)
        if name == "bias":
            return cls.bias_g[idx]
        if name == "workspace":
            return cls.workspace[idx]
        if name == "workspace_size":
            return cls.workspace_size


def fused_matmul_act(a, b, bias, activation):
    if a.is_cuda:
        workspace = StaticState.get("workspace", a.device)
        if bias is None:
            bias = StaticState.get("bias", a.device)
        return _ext.fused_matmul_act(a, b, bias, activation, workspace)
    else:
        if activation == "RELU":
            act = F.relu_
        elif activation == "GELU":
            act = F.gelu
        elif activation == "NONE":
            act = lambda x: x
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        return act(F.linear(a, b.T, bias))


class FusedLinearAct(torch.autograd.Function):
    """Fused linear layer with activation function.

    It performs the following operation:

    .. math::
        output = act(input @ weight^T + bias)

    where :math:`@` is the matrix multiplication operator.
    The activation function can be one of the following:
    - "RELU": ReLU activation
    - "GELU": GELU activation
    - "NONE": No activation (linear layer)
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor,
                bias: Optional[torch.Tensor], act: Literal["RELU", "GELU",
                                                           "NONE"]):
        assert input.ndim >= 2, f"Invalid input shape: {input.shape}"
        assert weight.ndim == 2 and weight.shape[1] == input.shape[
            -1], f"Invalid shape: {input.shape} {weight.shape}"
        assert bias is None or (bias.ndim == 1
                                and bias.shape[0] == weight.shape[0])

        if torch.is_autocast_enabled() and torch.get_autocast_dtype(
                'cuda') == torch.float16:
            input = input.to(torch.float16)
            weight = weight.to(torch.float16)
            if bias is not None:
                bias = bias.to(torch.float16)
        else:
            assert input.dtype == weight.dtype
            assert bias is None or bias.dtype == input.dtype

        output = fused_matmul_act(input.reshape(-1, input.shape[-1]), weight.T,
                                  bias, act)
        output = output.reshape(*input.shape[:-1], weight.shape[0])
        ctx.save_for_backward(input, weight, output if act != "NONE" else None)
        ctx.act = act
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, output = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        output = output.reshape(
            -1, output.shape[-1]) if output is not None else None
        grad = act_backward(output, grad_output, ctx.act)

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = fused_matmul_act(grad, weight, None, "NONE")
            grad_input = grad_input.reshape(input.shape)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = fused_matmul_act(grad.T,
                                           input.reshape(-1, input.shape[-1]),
                                           None, "NONE")

        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = grad.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None


fused_linear_act = FusedLinearAct.apply
