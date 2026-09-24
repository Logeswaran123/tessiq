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
/// What a plain matmul accumulates into. A device memset rather than a tensor operation, because
/// zeroing a tessiq tensor through torch reaches the fallback and costs a host round trip. It goes
/// on the same stream as the multiply that reads it, which is what orders the two.
Result<void> zero_fill(void* data, std::size_t bytes, int ordinal, Stream stream);

/// Runs `kernel` at `extents` on device `ordinal`, writing into `buffers.output`.
///
/// The extents are passed rather than read off `kernel`, because a description built for a graph
/// traced at more than one shape does not hold them. The caller works them out from the tensors
/// it was handed; `kernel` still says which operand is transposed and what the element type is,
/// neither of which changes with the shape.
///
/// The output must already hold what the multiply is added to: zero for a plain matmul, the bias
/// broadcast over every row for an addmm. BLAS scales the destination by beta and adds, which is
/// the only way it takes an accumulator, so filling it is the caller's job.
///
/// Queued on `stream`, and asynchronous like a kernel launch. Nothing here synchronises.
Result<void> launch_gemm(const GemmKernel& kernel,
                         const GemmExtents& extents,
                         const GemmBuffers& buffers,
                         int ordinal,
                         Stream stream);

} // namespace tessiq::codegen
