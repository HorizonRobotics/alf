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
__device__ __forceinline__ T relu_grad(T x, T go) {
  return (x > T(0)) ? go : T(0);
}

template <>
__device__ __forceinline__ __half relu_grad(__half x, __half go) {
  return __hgt(x, __float2half(0.f)) ? go : __float2half(0.f);
}

template <>
__device__ __forceinline__ __nv_bfloat16 relu_grad(__nv_bfloat16 x,
                                                   __nv_bfloat16 go) {
  return __hgt(x, __float2bfloat16(0.f)) ? go : __float2bfloat16(0.f);
}

template <typename T>
__global__ void relu_backward_kernel(const T* grad_output,
                                     const T* input,
                                     T* grad_input,
                                     int n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    grad_input[idx] = relu_grad(input[idx], grad_output[idx]);
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
    throw std::runtime_error("Error in relu_backward_kernel: " +
                             std::string(cudaGetErrorString(err)));
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
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, input.scalar_type(), "relu_backward", ([&] {
        relu_backward_cuda_launcher<scalar_t>(grad_output.data_ptr<scalar_t>(),
                                              input.data_ptr<scalar_t>(),
                                              grad_input.data_ptr<scalar_t>(),
                                              rows,
                                              cols);
      }));

  return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_backward",
        py::overload_cast<torch::Tensor, torch::Tensor>(&relu_backward),
        "ReLU backward pass CUDA kernel (grad_output * (input > 0))",
        py::arg("input"),
        py::arg("grad_output"));
}
