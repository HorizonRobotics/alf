// Copyright (c) 2025 Horizon Robotics and ALF Contributors.
// All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

template <typename T>
__global__ void relu_backward_kernel(const T* grad_output,
                                     const T* input,
                                     T* grad_input,
                                     int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    T zero = static_cast<T>(0.0f);
    grad_input[idx] = grad_output[idx] * (input[idx] > zero);
  }
}

template <>
__global__ void relu_backward_kernel<__half>(const __half* grad_output,
                                             const __half* input,
                                             __half* grad_input,
                                             int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    __half zero = __float2half(0.0f);
    grad_input[idx] = __hmul(
        grad_output[idx],
        __hgt(input[idx], zero) ? __float2half(1.0f) : __float2half(0.0f));
  }
}

// Host launcher
template <typename T>
void relu_backward_cuda_launcher(
    const T* grad_output, const T* input, T* grad_input, int rows, int cols) {
  dim3 blockDim(512);
  dim3 gridDim((rows * cols + blockDim.x - 1) / blockDim.x);
  relu_backward_kernel<T>
      <<<gridDim, blockDim>>>(grad_output, input, grad_input, rows * cols);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "Error in relu_backward_kernel: "
        "%s rows=%d cols=%d gridDim=(%d,%d,%d)\n",
        cudaGetErrorString(err),
        rows,
        cols,
        gridDim.x,
        gridDim.y,
        gridDim.z);
  }
}

// PyTorch wrapper function.
torch::Tensor relu_backward(const torch::Tensor input,
                            const torch::Tensor grad_output) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(grad_output.dim() == 2, "grad_output must be a 2D matrix");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D matrix");
  TORCH_CHECK(input.scalar_type() == grad_output.scalar_type(),
              "Input and grad_output must have the same dtype");

  int rows = input.size(0);
  int cols = input.size(1);

  auto grad_input = at::empty({rows, cols}, input.options());

  if (input.scalar_type() == at::kFloat) {
    relu_backward_cuda_launcher<float>(grad_output.data_ptr<float>(),
                                       input.data_ptr<float>(),
                                       grad_input.data_ptr<float>(),
                                       rows,
                                       cols);
  } else if (input.scalar_type() == at::kHalf) {
    relu_backward_cuda_launcher<__half>(
        reinterpret_cast<const __half*>(grad_output.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()),
        rows,
        cols);
  } else {
    TORCH_CHECK(false, "relu_backward only supports float32 and float16");
  }

  return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_backward",
        py::overload_cast<torch::Tensor, torch::Tensor>(&relu_backward),
        "ReLU backward pass CUDA kernel (grad_output * (input > 0))",
        py::arg("input"),
        py::arg("grad_output"));
}
