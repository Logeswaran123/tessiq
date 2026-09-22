// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <string>
#include <vector>

#include "tessiq/codegen/gemm.h"
#include "tessiq/codegen/program.h"
#include "tessiq/ir/graph.h"
#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// One fusion group, compiled and ready to run.
///
/// Names are IR value names, which the importer took from the FX nodes, so the caller can find the
/// tensors a group reads and the one it writes in the graph it started with.
struct FusedKernel {
  std::int64_t group = 0;

  /// Values the group reads, in the order the kernel takes them.
  std::vector<std::string> inputs;

  /// The value the group writes.
  std::string output;

  /// Every value the group computes, the output included. The caller skips these when walking the
  /// graph, because the kernel computes all of them in one pass.
  std::vector<std::string> members;

  /// Elements every buffer holds. Fusion refuses a shape change, so there is one count.
  std::int64_t elements = 0;

  /// Loaded and owned by the codegen layer, which keeps it for the life of the process.
  const codegen::Program* program = nullptr;

  /// Runs the kernel. `inputs` and `output` must be contiguous tensors on the device this was
  /// compiled for, and `output` must already be the right size.
  ///
  /// Asynchronous, like any launch. The caller orders the result against whatever reads it.
  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out) const;
};

/// Compiles every fusion group in `graph` for CUDA device `ordinal`.
///
/// Returns an empty list rather than an error when there is nothing to compile, when this build
/// has no CUDA toolkit, or when a group uses something the emitter has no code for. None of those
/// is a failure: the caller runs those operations one at a time, which always works.
Result<std::vector<FusedKernel>> compile_fused(const ir::Graph& graph, int ordinal);

/// One compiled reduction, ready to run.
///
/// Separate from FusedKernel because a reduction is its own kernel rather than a step in a loop.
/// It reads one tensor and writes one, and the three extents say how: `outer * inner` outputs,
/// each combining `reduce` elements that sit `inner` apart.
struct ReductionKernel {
  /// The IR name of the value it writes, which is how the executor finds the node it belongs to.
  std::string output;

  /// The IR name of the value it reads.
  std::string input;

  std::int64_t outer = 0;
  std::int64_t reduce = 0;
  std::int64_t inner = 0;

  /// Blocks to launch, which depends on which kernel shape the emitter wrote. Carried rather
  /// than recomputed, so the launch cannot disagree with the source about how the threads are
  /// laid over the problem.
  std::int64_t blocks = 0;

  const codegen::Program* program = nullptr;

  /// How many elements it reads and writes.
  [[nodiscard]] std::int64_t inputs() const { return outer * reduce * inner; }

  [[nodiscard]] std::int64_t outputs() const { return outer * inner; }

  [[nodiscard]] Result<void> launch(const at::Tensor& in, const at::Tensor& out) const;
};

/// Compiles every reduction in `graph` that the emitter can write.
///
/// A reduction it declines is not a failure. It runs on the host, which is what happened to every
/// reduction before this existed.
Result<std::vector<ReductionKernel>> compile_reductions(const ir::Graph& graph, int ordinal);

/// One matrix multiply, ready to run through cuBLAS.
///
/// Unlike the other two this has no compiled code behind it. What it carries is the shape of the
/// call: which operands, whether the library reads either transposed, and the extents.
struct GemmKernel {
  codegen::GemmKernel description;

  /// The CUDA device it runs on, which cuBLAS needs in order to pick its handle.
  int ordinal = 0;

  /// Fills the result with the bias before the multiply, when there is one. BLAS only takes an
  /// accumulator by scaling what is already there, so the bias has to be written first, and doing
  /// that through torch costs a host round trip per multiply.
  const codegen::Program* fill = nullptr;

  /// The IR names of the values read, in the order `launch` expects them: the two matrices, and
  /// the bias last when there is one.
  [[nodiscard]] std::vector<std::string> inputs() const;

  /// The IR name of the value written.
  [[nodiscard]] const std::string& output() const { return description.output; }

  /// Runs it. `tensors` matches `inputs()`.
  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out) const;
};

/// Describes every matrix multiply in `graph` that cuBLAS can run.
Result<std::vector<GemmKernel>> compile_gemms(const ir::Graph& graph, int ordinal);

} // namespace tessiq::torch_backend
