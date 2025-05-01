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
from torch.utils.cpp_extension import load
import pathlib
import os
from packaging.version import parse as parse_version

DIR = pathlib.Path(__file__).parent.absolute()

# Requires torch >= 2.6.0-rc1 to build this extension
# May also set ALF_DISABLE_CUDA_EXT=1 to disable the extension
_ext = None
_min_torch_version = parse_version("2.6.0-rc1")
_torch_version = parse_version(torch.__version__)

_should_load_ext = (torch.cuda.is_available()
                    and _torch_version >= _min_torch_version
                    and os.environ.get('ALF_DISABLE_CUDA_EXT', '0') != '1')

if _should_load_ext:
    try:
        _ext = load(name="act_backward",
                    sources=[os.path.join(DIR, "act_backward.cu")],
                    verbose=True)

        relu_backward_cuda = _ext.relu_backward
    except ImportError:
        # There is a bug in pybind11 currently where pybind11 will
        # incorrectly use the system python instead of the virtualenv python.
        # This can result in a python version mismatch error.
        # For now, we'll just catch this and ignore it.
        # See https://github.com/pybind/pybind11/issues/5626
        pass


def relu_backward(output, grad_output):
    """Computes the gradient of the ReLU activation function.

    If condition is satisfied, it uses the CUDA kernel for the computation.
    Otherwise, it uses the standard PyTorch operation.

    Args:
        output (torch.Tensor): The output of the ReLU activation function.
        grad_output (torch.Tensor): The gradient of the loss with respect to
            the output of the ReLU activation function.
    Returns:
        torch.Tensor: The gradient of the loss with respect to the input of
            the ReLU activation function. It is ```grad_output * (output > 0).float()``
    """
    assert output.ndim == 2
    assert grad_output.ndim == 2
    assert output.shape == grad_output.shape
    assert output.dtype == grad_output.dtype
    assert output.is_cuda == grad_output.is_cuda
    assert output.dtype.is_floating_point
    if output.is_cuda and output.is_contiguous() and grad_output.is_contiguous(
    ):
        return relu_backward_cuda(output, grad_output)
    else:
        return grad_output * (output > 0).float()


def act_backward(output: torch.Tensor, grad_output: torch.Tensor, act: str):
    """Computes the gradient of the activation function.

    Args:
        output (torch.Tensor): The output of the activation function.
        grad_output (torch.Tensor): The gradient of the loss with respect to
            the output of the activation function.
        act (str): The activation function used. One of "NONE", "RELU", "GELU".
    Returns:
        torch.Tensor: The gradient of the loss with respect to the input of
            the activation function.
    """
    if act == "NONE":
        return grad_output
    elif act == "RELU":
        return relu_backward(output, grad_output)
    elif act == "GELU":
        raise NotImplementedError("GELU backward is not implemented yet")
    else:
        raise ValueError(f"Unsupported activation function: {act}")
