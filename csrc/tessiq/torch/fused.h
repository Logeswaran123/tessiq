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

    /// Elements in the output, or `ir::kDynamic`. An input holds as many, or fewer when it
    /// broadcasts.
    std::int64_t elements = 0;

    /// How each input is read, in `inputs` order. A divisor not known at compile time is worked
    /// out at launch from the buffer sizes.
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

  /// The inputs' shapes broadcast together, which is the result's shape: every operation in a
  /// group is elementwise, so its result is the broadcast of its operands.
  [[nodiscard]] Result<std::vector<std::int64_t>>
  output_shape(const std::vector<at::Tensor>& tensors) const override;

  /// Every input and the output must be contiguous on the device this was compiled for, and the
  /// output must already be the right size. For a dynamic kernel that is checked against the
  /// inputs' broadcast shape, since the element count is taken from the output.
  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out,
                                    codegen::Stream stream) const override;

 private:
  Parts parts_;
};

/// One compiled reduction, ready to run.
///
/// Reads one tensor and writes `outer * inner` outputs, each combining `reduce` elements that sit
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

    /// The reduced axes and the input rank, so a launch can derive the extents from the tensor.
    std::size_t first_axis = 0;
    std::size_t last_axis = 0;
    std::size_t rank = 0;

    /// Whether the reduced axes stay in the result with size 1, for sizing the output.
    bool keeps_dims = false;

    /// The kernel shape the emitter wrote, so the launch lays out threads the same way.
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
/// Carries the call's operands, transpose flags and extents; there is no generated code.
class GemmKernel final : public Kernel {
 public:
  struct Parts {
    codegen::GemmKernel description;

    /// The CUDA device it runs on, which cuBLAS needs in order to pick its handle.
    int ordinal = 0;

    /// Writes the bias into every row of the result before the multiply, when there is one. See
    /// `codegen::broadcast_row_source`.
    const codegen::Program* fill = nullptr;
  };

  explicit GemmKernel(Parts parts) : parts_(std::move(parts)) {}

  [[nodiscard]] const Parts& parts() const { return parts_; }

  [[nodiscard]] std::string_view kind() const override { return "gemm"; }

  /// The two matrices, and the bias last when there is one.
  [[nodiscard]] std::vector<std::string> inputs() const override;

  [[nodiscard]] const std::string& output() const override { return parts_.description.output; }

  /// cuBLAS reads the layout it was described. Packing a transposed operand would leave the
  /// transpose flag describing data that moved.
  [[nodiscard]] bool packs_inputs() const override { return false; }

  [[nodiscard]] bool has_dynamic_extent() const override {
    return parts_.description.has_dynamic_extent();
  }

  /// The batch dimensions and rows of the left operand, and the columns of the right.
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
/// Carries a plan built when the graph was compiled; there is no generated code.
class ConvKernel final : public Kernel {
 public:
  struct Parts {
    codegen::ConvKernel description;

    /// Shared, so every copy of a compiled graph uses the same descriptors.
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
/// A node the compiler declines runs one operation at a time instead, which always works. So
/// does everything on a build with no CUDA toolkit, where this returns nothing.
Result<std::vector<std::shared_ptr<Kernel>>> compile_kernels(const ir::Graph& graph, int ordinal);

} // namespace tessiq::torch_backend
