// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "tessiq/codegen/gemm.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// Whether this build can call a BLAS library on the device.
[[nodiscard]] bool blas_available();

/// Device pointers for one matrix multiply, in the order the kernel description names them.
struct GemmBuffers {
  const void* lhs = nullptr;
  const void* rhs = nullptr;
  void* output = nullptr;
};

/// Writes `bytes` zero bytes at `data` on device `ordinal`.
///
/// What a plain matmul accumulates into. A device memset rather than a tensor operation, because
/// zeroing a tessiq tensor through torch reaches the fallback and costs a host round trip.
Result<void> zero_fill(void* data, std::size_t bytes, int ordinal);

/// Runs `kernel` on device `ordinal`, writing into `buffers.output`.
///
/// The output must already hold what the multiply is added to: zero for a plain matmul, the bias
/// broadcast over every row for an addmm. BLAS scales the destination by beta and adds, which is
/// the only way it takes an accumulator, so filling it is the caller's job.
///
/// Asynchronous, like a kernel launch. Nothing here synchronises.
Result<void> launch_gemm(const GemmKernel& kernel, const GemmBuffers& buffers, int ordinal);

} // namespace tessiq::codegen
