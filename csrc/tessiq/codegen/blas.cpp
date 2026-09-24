// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/blas.h"

#ifdef TESSIQ_WITH_CUDA

#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include <cuda_runtime.h>

#include "tessiq/codegen/gemm.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq::codegen {

namespace {

constexpr std::string_view kLogModule = "blas";

Error blas_error(cublasStatus_t status, const std::string& what) {
  return external_error(what + ": " + cublasGetStatusName(status));
}

std::mutex& handles_mutex() {
  static std::mutex mutex;
  return mutex;
}

/// Held from choosing a handle's stream until the call on it is queued.
///
/// A handle is shared by every launch on its device and carries one stream at a time, so two
/// threads launching on different streams could each set the stream and then queue on the other
/// one's. Queuing does not wait for the device, so this is held for microseconds.
std::mutex& launch_mutex() {
  static std::mutex mutex;
  return mutex;
}

/// One cuBLAS handle per device, created on first use and kept.
///
/// Creating a handle allocates workspace and is far more expensive than the multiplies a model
/// does with it, so this is a cache rather than a per-call resource.
Result<cublasHandle_t> handle_for(int ordinal) {
  static std::unordered_map<int, cublasHandle_t> handles;

  const std::scoped_lock guard(handles_mutex());
  if (const auto it = handles.find(ordinal); it != handles.end()) {
    return it->second;
  }

  if (const cudaError_t status = cudaSetDevice(ordinal); status != cudaSuccess) {
    return external_error(std::string("cudaSetDevice: ") + cudaGetErrorName(status));
  }

  cublasHandle_t handle = nullptr;
  if (const cublasStatus_t status = cublasCreate(&handle); status != CUBLAS_STATUS_SUCCESS) {
    return blas_error(status, "cublasCreate");
  }
  TESSIQ_LOG(kDebug, kLogModule) << "created a cublas handle for device " << ordinal;
  handles.emplace(ordinal, handle);
  return handle;
}

} // namespace

bool blas_available() { return true; }

Result<void> zero_fill(void* data, std::size_t bytes, int ordinal, Stream stream) {
  if (data == nullptr) {
    return invalid_argument("zero_fill was given a null buffer");
  }
  if (const cudaError_t status = cudaSetDevice(ordinal); status != cudaSuccess) {
    return external_error(std::string("cudaSetDevice: ") + cudaGetErrorName(status));
  }
  if (const cudaError_t status =
          cudaMemsetAsync(data, 0, bytes, static_cast<cudaStream_t>(stream.handle));
      status != cudaSuccess) {
    return external_error(std::string("cudaMemsetAsync: ") + cudaGetErrorName(status));
  }
  return ok();
}

Result<void> launch_gemm(const GemmKernel& kernel,
                         const GemmExtents& extents,
                         const GemmBuffers& buffers,
                         int ordinal,
                         Stream stream) {
  if (kernel.dtype != ir::DType::kF32) {
    return unsupported("cublas gemm is float32 only for now");
  }
  if (buffers.lhs == nullptr || buffers.rhs == nullptr || buffers.output == nullptr) {
    return invalid_argument("gemm was given a null buffer");
  }
  if (!extents.are_usable()) {
    return invalid_argument("gemm " + kernel.output + " was given no usable extents");
  }

  TESSIQ_ASSIGN_OR_RETURN(const cublasHandle_t handle, handle_for(ordinal));
  if (const cudaError_t status = cudaSetDevice(ordinal); status != cudaSuccess) {
    return external_error(std::string("cudaSetDevice: ") + cudaGetErrorName(status));
  }

  // cuBLAS reads column-major and the IR is row-major. A row-major matrix read as column-major is
  // its own transpose, so asking for C^T = B^T A^T in column-major computes C = A B in row-major
  // with the operands swapped and nothing copied. That is why n and m are the other way round
  // below, and why the output's leading dimension is n.
  const auto op = [](const GemmOperand& operand) {
    return operand.transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  };
  const float alpha = 1.0F;
  const float beta = 1.0F;

  const auto m = static_cast<int>(extents.m);
  const auto n = static_cast<int>(extents.n);
  const auto k = static_cast<int>(extents.k);
  const auto lda = static_cast<int>(extents.lhs_leading);
  const auto ldb = static_cast<int>(extents.rhs_leading);

  auto* c = static_cast<float*>(buffers.output);

  // Named for what cuBLAS sees rather than for what the IR called them, because they are the
  // other way round: the first matrix it multiplies is our second one.
  const cublasOperation_t first_op = op(kernel.rhs);
  const cublasOperation_t second_op = op(kernel.lhs);
  const auto* first = static_cast<const float*>(buffers.rhs);
  const auto* second = static_cast<const float*>(buffers.lhs);
  const int first_ld = ldb;
  const int second_ld = lda;

  const std::scoped_lock guard(launch_mutex());
  if (const cublasStatus_t status =
          cublasSetStream(handle, static_cast<cudaStream_t>(stream.handle));
      status != CUBLAS_STATUS_SUCCESS) {
    return blas_error(status, "cublasSetStream");
  }

  if (extents.batch == 1) {
    if (const cublasStatus_t status = cublasSgemm(handle,
                                                  first_op,
                                                  second_op,
                                                  n,
                                                  m,
                                                  k,
                                                  &alpha,
                                                  first,
                                                  first_ld,
                                                  second,
                                                  second_ld,
                                                  &beta,
                                                  c,
                                                  n);
        status != CUBLAS_STATUS_SUCCESS) {
      return blas_error(status, "cublasSgemm");
    }
    return ok();
  }

  // Batched, with each matrix following the last. The strides are in elements.
  const auto first_stride = static_cast<long long>(extents.k) * extents.n;
  const auto second_stride = static_cast<long long>(extents.m) * extents.k;
  const auto out_stride = static_cast<long long>(extents.m) * extents.n;
  if (const cublasStatus_t status = cublasSgemmStridedBatched(handle,
                                                              first_op,
                                                              second_op,
                                                              n,
                                                              m,
                                                              k,
                                                              &alpha,
                                                              first,
                                                              first_ld,
                                                              first_stride,
                                                              second,
                                                              second_ld,
                                                              second_stride,
                                                              &beta,
                                                              c,
                                                              n,
                                                              out_stride,
                                                              static_cast<int>(extents.batch));
      status != CUBLAS_STATUS_SUCCESS) {
    return blas_error(status, "cublasSgemmStridedBatched");
  }
  return ok();
}

} // namespace tessiq::codegen

#else // TESSIQ_WITH_CUDA

namespace tessiq::codegen {

bool blas_available() { return false; }

Result<void> zero_fill(void* /*data*/, std::size_t /*bytes*/, int /*ordinal*/, Stream /*stream*/) {
  return unsupported("this build has no CUDA toolkit");
}

Result<void> launch_gemm(const GemmKernel& /*kernel*/,
                         const GemmExtents& /*extents*/,
                         const GemmBuffers& /*buffers*/,
                         int /*ordinal*/,
                         Stream /*stream*/) {
  return unsupported("this build has no CUDA toolkit, so there is no BLAS to call");
}

} // namespace tessiq::codegen

#endif // TESSIQ_WITH_CUDA
