// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "tessiq/codegen/gemm.h"
#include "tessiq/codegen/stream.h"
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

/// Writes `bytes` zero bytes at `data` on device `ordinal`, queued on `stream`.
///
/// Clears a plain matmul's output before the multiply adds into it. A device memset, since
/// zeroing through torch would reach the fallback. Queued on the multiply's stream, which orders
/// the two.
Result<void> zero_fill(void* data, std::size_t bytes, int ordinal, Stream stream);

/// Runs `kernel` at `extents` on device `ordinal`, writing into `buffers.output`.
///
/// The extents are passed separately because a description of a graph traced at more than one
/// shape does not hold them. `kernel` supplies the transpose flags and the element type.
///
/// The output must already hold what the product is added to: zero for a plain matmul, the bias
/// broadcast over every row for an addmm.
///
/// Asynchronous on `stream`.
Result<void> launch_gemm(const GemmKernel& kernel,
                         const GemmExtents& extents,
                         const GemmBuffers& buffers,
                         int ordinal,
                         Stream stream);

} // namespace tessiq::codegen
