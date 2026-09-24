// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tessiq/codegen/conv.h"
#include "tessiq/codegen/cudnn.h"
#include "tessiq/codegen/gemm.h"
#include "tessiq/codegen/kernel.h"
#include "tessiq/codegen/program.h"
#include "tessiq/codegen/reduction.h"
#include "tessiq/codegen/stream.h"
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

  /// How each input is reached, in the order `inputs` names them. A kernel written for a shape
  /// nobody knew could not fold a broadcast's divisor into its source, so the launch works each
  /// one out from the buffer sizes and passes it beside the element count.
  std::vector<codegen::OperandRead> reads;

  /// Loaded and owned by the codegen layer, which keeps it for the life of the process.
  const codegen::Program* program = nullptr;

  /// Runs the kernel on `stream`. `inputs` and `output` must be contiguous tensors on the device
  /// this was compiled for, and `output` must already be the right size.
  ///
  /// Asynchronous, like any launch. The caller orders the result against whatever reads it.
  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out,
                                    codegen::Stream stream) const;
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

  /// The axes folded and the rank they are folded over, so that a launch given a shape the
  /// compiler never saw can work the three extents out from the tensor itself.
  std::size_t first_axis = 0;
  std::size_t last_axis = 0;
  std::size_t rank = 0;

  /// Whether the folded axes stay in the result as ones, which the executor needs in order to
  /// allocate a tensor the rest of the graph can read.
  bool keeps_dims = false;

  /// Which kernel shape the emitter wrote. Carried rather than inferred, so the launch cannot
  /// disagree with the source about how the threads are laid over the problem.
  codegen::ReductionShape shape = codegen::ReductionShape::kBlockPerOutput;

  const codegen::Program* program = nullptr;

  /// Whether the extents arrive with the tensor rather than being known here.
  [[nodiscard]] bool has_dynamic_extent() const { return !extents().are_usable(); }

  /// What the description knew, with `ir::kDynamic` where it knew nothing.
  [[nodiscard]] codegen::ReductionExtents extents() const {
    codegen::ReductionExtents known;
    known.outer = outer;
    known.reduce = reduce;
    known.inner = inner;
    return known;
  }

  /// How many elements it reads and writes. Meaningless unless the extents are known.
  [[nodiscard]] std::int64_t inputs() const { return outer * reduce * inner; }

  [[nodiscard]] std::int64_t outputs() const { return outer * inner; }

  /// The three extents one launch walks, taken from the tensor it is handed.
  [[nodiscard]] Result<codegen::ReductionExtents> extents_for(const at::Tensor& in) const;

  /// The shape of the tensor it writes, given the one it will read.
  [[nodiscard]] Result<std::vector<std::int64_t>>
  output_shape(const std::vector<at::Tensor>& tensors) const;

  [[nodiscard]] Result<void>
  launch(const at::Tensor& in, const at::Tensor& out, codegen::Stream stream) const;
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

  /// Whether any number the call needs arrives with the tensors rather than being known here.
  [[nodiscard]] bool has_dynamic_extent() const { return description.has_dynamic_extent(); }

  /// The shape of the tensor this writes, given the ones it will read.
  ///
  /// The batch dimensions and the row count come from the left operand and the column count from
  /// the right, which is where they are even when the description has numbers of its own. The
  /// caller needs this because the fake tensor it would otherwise size the result from holds
  /// symbolic extents and has no shape.
  [[nodiscard]] Result<std::vector<std::int64_t>>
  output_shape(const std::vector<at::Tensor>& tensors) const;

  /// Runs it on `stream`. `tensors` matches `inputs()`.
  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out,
                                    codegen::Stream stream) const;
};

/// Describes every matrix multiply in `graph` that cuBLAS can run.
Result<std::vector<GemmKernel>> compile_gemms(const ir::Graph& graph, int ordinal);

/// One convolution, ready to run through cuDNN.
///
/// Like a matmul this has no generated code behind it. What it carries is a plan: the problem
/// described to cuDNN and an algorithm chosen, both done once when the graph was compiled.
struct ConvKernel {
  codegen::ConvKernel description;

  /// Shared rather than owned, because a compiled graph is copied into Python and every copy has
  /// to reach the same cuDNN descriptors and workspace.
  std::shared_ptr<const codegen::ConvPlan> plan;

  /// The IR names of the values read, in the order `launch` expects them.
  [[nodiscard]] std::vector<std::string> inputs() const { return description.inputs(); }

  /// The IR name of the value written.
  [[nodiscard]] const std::string& output() const { return description.output; }

  /// The shape of the tensor this writes. Fixed at compile time, since a convolution whose shapes
  /// were not known has no plan.
  [[nodiscard]] std::vector<std::int64_t> output_shape() const {
    return {description.output_shape.begin(), description.output_shape.end()};
  }

  /// Runs it on `stream`. `tensors` matches `inputs()`, packed NCHW.
  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out,
                                    codegen::Stream stream) const;
};

/// Plans every convolution in `graph` that cuDNN can run.
///
/// A convolution with no plan is not a failure. It runs on the host, which is what every one did
/// before this existed.
Result<std::vector<ConvKernel>> compile_convs(const ir::Graph& graph, int ordinal);

} // namespace tessiq::torch_backend
