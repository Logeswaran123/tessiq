// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
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
#include "tessiq/torch/kernel.h"

namespace tessiq::torch_backend {

/// One fusion group, compiled and ready to run.
class FusedKernel final : public Kernel {
 public:
  struct Parts {
    std::int64_t group = 0;

    /// Values the group reads, in the order the kernel takes them.
    std::vector<std::string> inputs;

    /// The value the group writes.
    std::string output;

    /// Every value the group computes, the output included.
    std::vector<std::string> members;

    /// Elements every buffer holds. Fusion refuses a shape change, so there is one count.
    std::int64_t elements = 0;

    /// How each input is reached, in the order `inputs` names them. A kernel written for a shape
    /// nobody knew could not fold a broadcast's divisor into its source, so the launch works each
    /// one out from the buffer sizes and passes it beside the element count.
    std::vector<codegen::OperandRead> reads;

    /// Loaded and owned by the codegen layer, which keeps it for the life of the process.
    const codegen::Program* program = nullptr;
  };

  explicit FusedKernel(Parts parts) : parts_(std::move(parts)) {}

  [[nodiscard]] const Parts& parts() const { return parts_; }

  [[nodiscard]] std::string_view kind() const override { return "fused"; }

  [[nodiscard]] std::vector<std::string> inputs() const override { return parts_.inputs; }

  [[nodiscard]] const std::string& output() const override { return parts_.output; }

  [[nodiscard]] std::vector<std::string> members() const override { return parts_.members; }

  [[nodiscard]] bool packs_inputs() const override { return true; }

  [[nodiscard]] bool has_dynamic_extent() const override;

  /// The first input's shape. Every buffer in a group has the result's shape, which is what lets
  /// one index walk all of them.
  [[nodiscard]] Result<std::vector<std::int64_t>>
  output_shape(const std::vector<at::Tensor>& tensors) const override;

  /// Every input and the output must be contiguous on the device this was compiled for, and the
  /// output must already be the right size.
  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out,
                                    codegen::Stream stream) const override;

 private:
  Parts parts_;
};

/// One compiled reduction, ready to run.
///
/// A kernel of its own rather than a step in a loop. It reads one tensor and writes one, and the
/// three extents say how: `outer * inner` outputs, each combining `reduce` elements that sit
/// `inner` apart.
class ReductionKernel final : public Kernel {
 public:
  struct Parts {
    /// The value it writes.
    std::string output;

    /// The value it reads.
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
  };

  explicit ReductionKernel(Parts parts) : parts_(std::move(parts)) {}

  [[nodiscard]] const Parts& parts() const { return parts_; }

  [[nodiscard]] std::string_view kind() const override { return "reduction"; }

  [[nodiscard]] std::vector<std::string> inputs() const override { return {parts_.input}; }

  [[nodiscard]] const std::string& output() const override { return parts_.output; }

  [[nodiscard]] bool packs_inputs() const override { return true; }

  [[nodiscard]] bool has_dynamic_extent() const override { return !extents().are_usable(); }

  /// What the description knew, with `ir::kDynamic` where it knew nothing.
  [[nodiscard]] codegen::ReductionExtents extents() const;

  /// How many elements it reads. Meaningless unless the extents are known.
  [[nodiscard]] std::int64_t input_elements() const {
    return parts_.outer * parts_.reduce * parts_.inner;
  }

  /// Blocks one launch at the known extents takes.
  [[nodiscard]] std::int64_t blocks() const;

  /// The three extents one launch walks, taken from the tensor it is handed.
  [[nodiscard]] Result<codegen::ReductionExtents> extents_for(const at::Tensor& in) const;

  [[nodiscard]] Result<std::vector<std::int64_t>>
  output_shape(const std::vector<at::Tensor>& tensors) const override;

  /// `tensors` holds exactly the one tensor it reduces.
  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out,
                                    codegen::Stream stream) const override;

 private:
  Parts parts_;
};

/// One matrix multiply, ready to run through cuBLAS.
///
/// No compiled code behind this. What it carries is the shape of the call: which operands, whether
/// the library reads either transposed, and the extents.
class GemmKernel final : public Kernel {
 public:
  struct Parts {
    codegen::GemmKernel description;

    /// The CUDA device it runs on, which cuBLAS needs in order to pick its handle.
    int ordinal = 0;

    /// Fills the result with the bias before the multiply, when there is one. BLAS only takes an
    /// accumulator by scaling what is already there, so the bias has to be written first, and
    /// doing that through torch costs a host round trip per multiply.
    const codegen::Program* fill = nullptr;
  };

  explicit GemmKernel(Parts parts) : parts_(std::move(parts)) {}

  [[nodiscard]] const Parts& parts() const { return parts_; }

  [[nodiscard]] std::string_view kind() const override { return "gemm"; }

  /// The two matrices, and the bias last when there is one.
  [[nodiscard]] std::vector<std::string> inputs() const override;

  [[nodiscard]] const std::string& output() const override { return parts_.description.output; }

  /// The one kernel that reads a layout rather than requiring one. Packing a transposed operand
  /// before handing it over would move the data and leave the transpose flag describing a layout
  /// that is no longer there.
  [[nodiscard]] bool packs_inputs() const override { return false; }

  [[nodiscard]] bool has_dynamic_extent() const override {
    return parts_.description.has_dynamic_extent();
  }

  /// The batch dimensions and the row count come from the left operand and the column count from
  /// the right, which is where they are even when the description has numbers of its own.
  [[nodiscard]] Result<std::vector<std::int64_t>>
  output_shape(const std::vector<at::Tensor>& tensors) const override;

  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out,
                                    codegen::Stream stream) const override;

 private:
  Parts parts_;
};

/// One convolution, ready to run through cuDNN.
///
/// Like a matmul this has no generated code behind it. What it carries is a plan: the problem
/// described to cuDNN and an algorithm chosen, both done once when the graph was compiled.
class ConvKernel final : public Kernel {
 public:
  struct Parts {
    codegen::ConvKernel description;

    /// Shared rather than owned, because every copy of a compiled graph has to reach the same
    /// cuDNN descriptors and workspace.
    std::shared_ptr<const codegen::ConvPlan> plan;
  };

  explicit ConvKernel(Parts parts) : parts_(std::move(parts)) {}

  [[nodiscard]] const Parts& parts() const { return parts_; }

  [[nodiscard]] std::string_view kind() const override { return "conv"; }

  [[nodiscard]] std::vector<std::string> inputs() const override {
    return parts_.description.inputs();
  }

  [[nodiscard]] const std::string& output() const override { return parts_.description.output; }

  /// cuDNN reads packed NCHW and was told so when the plan was built.
  [[nodiscard]] bool packs_inputs() const override { return true; }

  /// A convolution whose shapes were not known has no plan, so every one here is fixed.
  [[nodiscard]] bool has_dynamic_extent() const override { return false; }

  [[nodiscard]] Result<std::vector<std::int64_t>>
  output_shape(const std::vector<at::Tensor>& tensors) const override;

  /// `tensors` matches `inputs()`, packed NCHW.
  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out,
                                    codegen::Stream stream) const override;

 private:
  Parts parts_;
};

/// Every kernel `graph` can run on CUDA device `ordinal`, of whatever kind.
///
/// Something the compiler declines is not a failure. A group the emitter has no code for, a
/// reduction with no kernel shape, a matmul cuBLAS cannot take and a convolution cuDNN has no plan
/// for all run one operation at a time instead, which always works. So does everything on a build
/// with no CUDA toolkit, where this returns nothing.
Result<std::vector<std::shared_ptr<Kernel>>> compile_kernels(const ir::Graph& graph, int ordinal);

} // namespace tessiq::torch_backend
