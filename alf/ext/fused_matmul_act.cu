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
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define DEFAULT_WORKSPACE_SIZE (8 * 1024 * 1024)

// From aten/src/ATen/cuda/EmptyTensor.cpp
namespace at::detail {

TensorBase empty_cuda(IntArrayRef size,
                      ScalarType dtype,
                      std::optional<Device> device_opt,
                      std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_cuda());
  const DeviceGuard device_guard(device);
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  return at::detail::empty_generic(
      size, allocator, cuda_dks, dtype, memory_format_opt);
}

TensorBase empty_cuda(IntArrayRef size,
                      std::optional<ScalarType> dtype_opt,
                      std::optional<Layout> layout_opt,
                      std::optional<Device> device_opt,
                      std::optional<bool> pin_memory_opt,
                      std::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt,
              "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) ==
                                   Layout::Strided);

  const auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_cuda(size, dtype, device_opt, memory_format_opt);
}

TensorBase empty_cuda(IntArrayRef size, const TensorOptions& options) {
  return at::detail::empty_cuda(size,
                                optTypeMetaToScalarType(options.dtype_opt()),
                                options.layout_opt(),
                                options.device_opt(),
                                options.pinned_memory_opt(),
                                options.memory_format_opt());
}
}  // namespace at::detail

#define checkCudaStatus(call)                               \
  {                                                         \
    cudaError_t status = call;                              \
    if (status != cudaSuccess) {                            \
      printf("%s:%d: cuda API failed with status %d: %s\n", \
             __FILE__,                                      \
             __LINE__,                                      \
             status,                                        \
             cudaGetErrorString(status));                   \
      throw std::logic_error("cuda API failed");            \
    }                                                       \
  }

#define checkCublasStatus(call)                           \
  {                                                       \
    cublasStatus_t status = call;                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                \
      printf("%s:%d: CUBLAS API failed with status %d\n", \
             __FILE__,                                    \
             __LINE__,                                    \
             status);                                     \
      throw std::logic_error("cuBLAS API failed");        \
    }                                                     \
  }

cudaDataType_t convertTensorDtypeToCudaDataType(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat:
      return CUDA_R_32F;
    case torch::kHalf:
      return CUDA_R_16F;
    case torch::kBFloat16:
      return CUDA_R_16BF;
    case torch::kDouble:
      return CUDA_R_64F;
    default:
      throw std::invalid_argument("Unsupported tensor data type");
  }
}

union MixedScalar {
  at::Half f16;
  at::BFloat16 bf16;
  float f32;
  double f64;
};

void getComputeTypeAndScaleType(cudaDataType_t data_type,
                                cublasComputeType_t* compute_type,
                                cudaDataType_t* scale_type) {
  // See https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
  // for all valid combinations of compute type and data type
  switch (data_type) {
    case CUDA_R_16F:
      *compute_type = CUBLAS_COMPUTE_16F;
      *scale_type = CUDA_R_16F;
      break;
    case CUDA_R_16BF:
      *compute_type = CUBLAS_COMPUTE_32F;
      *scale_type = CUDA_R_32F;
      break;
    case CUDA_R_32F:
      *compute_type = CUBLAS_COMPUTE_32F;
      *scale_type = CUDA_R_32F;
      break;
    case CUDA_R_64F:
      *compute_type = CUBLAS_COMPUTE_64F;
      *scale_type = CUDA_R_64F;
      break;
    default:
      throw std::invalid_argument("Unsupported data type");
  }
}

MixedScalar convertToMixedScalar(double number, cudaDataType_t dtype) {
  MixedScalar result;
  switch (dtype) {
    case CUDA_R_32F:
      result.f32 = number;
      break;
    case CUDA_R_64F:
      result.f64 = number;
      break;
    case CUDA_R_16F:
      result.f16 = at::Half(number);
      break;
    case CUDA_R_16BF:
      result.bf16 = at::BFloat16(number);
      break;
    default:
      throw std::invalid_argument("Unsupported tensor data type");
  }
  return result;
}

cublasLtMatrixLayout_t createMatrixLayout(torch::Tensor tensor,
                                          cublasOperation_t* trans) {
  cublasLtMatrixLayout_t layout = nullptr;

  int64_t m;
  int64_t n;

  if (tensor.stride(0) != 1 && tensor.stride(1) != 1) {
    throw std::invalid_argument("Unsupported tensor layout");
  }
  cudaDataType_t dataType =
      convertTensorDtypeToCudaDataType(tensor.scalar_type());

  int64_t ld;
  if (tensor.stride(1) == 1) {
    ld = tensor.stride(0);
    *trans = CUBLAS_OP_N;
    m = tensor.size(1);
    n = tensor.size(0);
  } else {
    ld = tensor.stride(1);
    *trans = CUBLAS_OP_T;
    m = tensor.size(0);
    n = tensor.size(1);
  }
  checkCublasStatus(cublasLtMatrixLayoutCreate(&layout, dataType, m, n, ld));
  return layout;
}

void printMatrixLayout(cublasLtMatrixLayout_t layout) {
  int64_t rows, cols, ld;
  size_t size;
  checkCublasStatus(cublasLtMatrixLayoutGetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows), &size));
  checkCublasStatus(cublasLtMatrixLayoutGetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols), &size));
  checkCublasStatus(cublasLtMatrixLayoutGetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld), &size));

  printf("Matrix Layout: rows=%ld, cols=%ld, ld=%ld\n", rows, cols, ld);
}

torch::Tensor fused_matmul_act(torch::Tensor a,
                               torch::Tensor b,
                               torch::Tensor bias = {},
                               std::string epilogue_str = "NONE",
                               torch::Tensor workspace = {},
                               int64_t workspaceSize = DEFAULT_WORKSPACE_SIZE) {
  // Somehow setting matrix layout as row-major doesn't work out. So we
  // have to use column-major order, but torch uses row-major, so we use
  // out^T = a^T * b^T + bias to get the correct result.

  c10::cuda::CUDAGuard device_guard(a.device());

  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  bool has_bias = bias.numel() > 0;

  if (workspace.numel() == 0) {
    // Allocate workspace if not provided
    workspace = torch::empty(
        workspaceSize,
        at::TensorOptions().dtype(torch::kUInt8).device(a.device()));
  } else {
    workspaceSize = workspace.numel();
  }

  if (a.dim() != 2 || b.dim() != 2) {
    throw std::invalid_argument("Input tensors must be 2D");
  }
  if (a.size(1) != b.size(0)) {
    throw std::invalid_argument("Matrix dimensions do not match");
  }
  if (has_bias && bias.dim() != 1) {
    throw std::invalid_argument("Bias tensor must be 1D");
  }
  if (has_bias && bias.size(0) != b.size(1)) {
    throw std::invalid_argument(
        "Bias tensor size does not match input tensor size");
  }
  if (has_bias && bias.scalar_type() != a.scalar_type()) {
    throw std::invalid_argument(
        "Bias tensor must be of the same type as input tensors");
  }
  if (epilogue_str != "NONE" && epilogue_str != "RELU" &&
      epilogue_str != "GELU") {
    throw std::invalid_argument("Invalid epilogue type");
  }
  if (a.scalar_type() != b.scalar_type()) {
    throw std::invalid_argument("Input tensors must be same");
  }

  cudaDataType_t cublasBiasDataType =
      convertTensorDtypeToCudaDataType(bias.scalar_type());

  // Allocate output tensor. It's important to use empty_cuda().
  // For some unknown reason, torch::empty() makes the matmul call significantly
  // slow.
  at::Tensor out =
      at::detail::empty_cuda({a.sizes()[0], b.sizes()[1]}, a.options());
  // torch::Tensor out = torch::empty({a.size(0), b.size(1)}, a.options());

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasOperation_t transa;
  cublasOperation_t transb;
  cublasOperation_t transout;
  cublasLtMatrixLayout_t aDesc = createMatrixLayout(a, &transa);
  cublasLtMatrixLayout_t bDesc = createMatrixLayout(b, &transb);
  cublasLtMatrixLayout_t outDesc = createMatrixLayout(out, &transout);
  // printMatrixLayout(aDesc);
  // printMatrixLayout(bDesc);
  // printMatrixLayout(outDesc);
  // printf("transa: %d, transb: %d, transout: %d\n",
  //   transa, transb, transout);

  cublasLtMatmulPreference_t preference = nullptr;

  // Handle transposition
  cudaDataType_t data_type = convertTensorDtypeToCudaDataType(a.scalar_type());
  cudaDataType_t scale_type;
  cublasComputeType_t compute_type;
  getComputeTypeAndScaleType(data_type, &compute_type, &scale_type);

  // Create matmul operation descriptor
  checkCublasStatus(
      cublasLtMatmulDescCreate(&operationDesc, compute_type, scale_type));

  // Set transposition attributes
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transb, sizeof(transb)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transa, sizeof(transa)));

  // Parse epilogue type
  cublasLtEpilogue_t epilogue;

  // Set epilogue attributes
  if (epilogue_str == "NONE") {
    epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
  } else if (epilogue_str == "RELU") {
    epilogue = has_bias ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_RELU;
  } else if (epilogue_str == "GELU") {
    epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU;
  } else {
    throw std::invalid_argument("Invalid epilogue type");
  }
  // Set epilogue attribute in operation descriptor
  checkCublasStatus(
      cublasLtMatmulDescSetAttribute(operationDesc,
                                     CUBLASLT_MATMUL_DESC_EPILOGUE,
                                     &epilogue,
                                     sizeof(epilogue)));

  // Set bias attributes if bias is provided
  if (has_bias) {
    const void* bias_ref = bias.const_data_ptr();
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc,
                                       CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &bias_ref,
                                       sizeof(bias_ref)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc,
                                       CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                       &cublasBiasDataType,
                                       sizeof(cublasBiasDataType)));
  }

  // Create preference descriptor & set workspace size
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
      preference,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceSize,
      sizeof(workspaceSize)));

  /*
  cublasStatus_t cublasLtMatmul(
      cublasLtHandle_t               lightHandle,
      cublasLtMatmulDesc_t           computeDesc,
      const void                    *alpha,
      const void                    *A,
      cublasLtMatrixLayout_t         Adesc,
      const void                    *B,
      cublasLtMatrixLayout_t         Bdesc,
      const void                    *beta,
      const void                    *C,
      cublasLtMatrixLayout_t         Cdesc,
      void                          *D,
      cublasLtMatrixLayout_t         Ddesc,
      const cublasLtMatmulAlgo_t    *algo,
      void                          *workspace,
      size_t                         workspaceSizeInBytes,
      cudaStream_t                   stream);
  */

  const MixedScalar alpha = convertToMixedScalar(1.0, scale_type);
  const MixedScalar beta = convertToMixedScalar(0.0, scale_type);

  checkCublasStatus(
      cublasLtMatmul(ltHandle,
                     operationDesc,
                     &alpha,
                     b.const_data_ptr(),
                     bDesc,
                     a.const_data_ptr(),
                     aDesc,
                     &beta,
                     out.mutable_data_ptr(),
                     outDesc,
                     out.mutable_data_ptr(),
                     outDesc,
                     nullptr,
                     workspace.mutable_data_ptr(),
                     workspaceSize,
                     at::cuda::getCurrentCUDAStream(a.device().index())));

  // Clean up
  checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(outDesc));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(bDesc));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(aDesc));
  checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_matmul_act",
        py::overload_cast<torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          std::string,
                          torch::Tensor,
                          int64_t>(&fused_matmul_act),
        "Fused Matmul+Activation (act(a * b + bias))",
        py::arg("a"),
        py::arg("b"),
        py::arg("bias") = torch::empty({}),
        py::arg("epilogue_str") = "NONE",
        py::arg("workspace") = torch::empty({}),
        py::arg("workspaceSize") = 1024 * 1024 * 8);
}
