// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/torch/fused.h"

#include <ATen/core/Tensor.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tessiq/codegen/blas.h"
#include "tessiq/codegen/conv.h"
#include "tessiq/codegen/cubin_cache.h"
#include "tessiq/codegen/cudnn.h"
#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/gemm.h"
#include "tessiq/codegen/kernel.h"
#include "tessiq/codegen/program.h"
#include "tessiq/codegen/reduction.h"
#include "tessiq/codegen/stream.h"
#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/kernel.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "fused";

/// The number a broadcast's index expression divides or wraps by, worked out from the buffers.
///
/// Both follow from the element counts alone: an operand read in runs holds one element per run,
/// and one read in cycles holds exactly one cycle.
std::int64_t
divisor_for(codegen::OperandRead::Kind kind, std::int64_t operand, std::int64_t count) {
  return kind == codegen::OperandRead::Kind::kRepeated ? count / operand : operand;
}

std::string shape_string(c10::IntArrayRef sizes) {
  std::string text = "[";
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    text += (i == 0 ? "" : ",") + std::to_string(sizes[i]);
  }
  return text + "]";
}

/// The shapes of `tensors` broadcast together, aligned from the last axis as torch does.
Result<std::vector<std::int64_t>> broadcast_shape(const std::vector<at::Tensor>& tensors,
                                                  const std::string& group) {
  std::vector<std::int64_t> shape;
  for (const at::Tensor& tensor : tensors) {
    const c10::IntArrayRef sizes = tensor.sizes();
    if (sizes.size() > shape.size()) {
      shape.insert(shape.begin(), sizes.size() - shape.size(), 1);
    }
    const std::size_t offset = shape.size() - sizes.size();
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      std::int64_t& extent = shape[offset + i];
      if (sizes[i] == extent || sizes[i] == 1) {
        continue;
      }
      if (extent != 1) {
        return invalid_argument("fusion group " + group + " was given inputs of shapes that do " +
                                "not broadcast, including " + shape_string(sizes));
      }
      extent = sizes[i];
    }
  }
  return shape;
}

/// A dynamic kernel's scalars: the element count, then one divisor per operand that needs one, in
/// the emitter's parameter order.
std::vector<std::int64_t> dynamic_scalars(const std::vector<codegen::OperandRead>& reads,
                                          const std::vector<at::Tensor>& tensors,
                                          std::int64_t count) {
  std::vector<std::int64_t> scalars{count};
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (i < reads.size() && !reads[i].is_known()) {
      scalars.push_back(divisor_for(reads[i].kind, tensors[i].numel(), count));
    }
  }
  return scalars;
}

/// Checks that `out` has the shape `tensors` broadcast to.
///
/// A dynamic kernel takes its element count from `out`, so an output of the wrong shape would pass
/// the launch's other checks whenever every input's size divides its element count.
Result<void> check_broadcast_output(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out,
                                    const std::string& group) {
  TESSIQ_ASSIGN_OR_RETURN(const std::vector<std::int64_t> expected,
                          broadcast_shape(tensors, group));
  if (out.sizes() != c10::IntArrayRef(expected)) {
    return invalid_argument("fusion group " + group + " was given an output of shape " +
                            shape_string(out.sizes()) + "; its inputs broadcast to " +
                            shape_string(expected));
  }
  return ok();
}

} // namespace

bool FusedKernel::has_dynamic_extent() const { return parts_.elements == ir::kDynamic; }

Result<std::vector<std::int64_t>>
FusedKernel::output_shape(const std::vector<at::Tensor>& tensors) const {
  if (tensors.empty()) {
    return invalid_argument("fusion group " + std::to_string(parts_.group) +
                            " cannot size a result from no inputs");
  }
  return broadcast_shape(tensors, std::to_string(parts_.group));
}

Result<void> FusedKernel::launch(const std::vector<at::Tensor>& tensors,
                                 const at::Tensor& out,
                                 codegen::Stream stream) const {
  const std::vector<codegen::OperandRead>& reads = parts_.reads;
  const std::string group = std::to_string(parts_.group);

  /// Whether operand `i` of this many elements can be read for a result of `count`.
  ///
  /// An operand read at `i` has to match exactly. One that broadcasts has to divide the result
  /// evenly, or the last run or cycle would read past the end.
  const auto reaches = [&reads](std::size_t i, std::int64_t operand, std::int64_t count) {
    if (i >= reads.size() || reads[i].kind == codegen::OperandRead::Kind::kSameShape) {
      return operand == count;
    }
    return operand > 0 && count % operand == 0;
  };

  if (parts_.program == nullptr) {
    return internal_error("fusion group " + group + " has no compiled kernel");
  }
  if (tensors.size() != parts_.inputs.size()) {
    return invalid_argument("fusion group " + group + " takes " +
                            std::to_string(parts_.inputs.size()) + " inputs, got " +
                            std::to_string(tensors.size()));
  }

  // A dynamic kernel takes its count from the output it is handed. The IR cannot tell two `?`
  // extents apart, so the checks below are what keep it from reading past a buffer.
  const bool dynamic = has_dynamic_extent();
  const std::int64_t count = dynamic ? out.numel() : parts_.elements;
  if (count == 0) {
    // Only a dynamic kernel meets this; an empty static group is never compiled.
    return ok();
  }

  std::vector<void*> buffers;
  buffers.reserve(tensors.size() + 1);
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    // The kernel indexes every buffer as packed.
    if (!tensor.is_contiguous()) {
      return invalid_argument("fusion group " + group + " input " + std::to_string(i) +
                              " is not contiguous");
    }
    // A broadcast input has fewer elements than the result; none may have more.
    const bool fits = dynamic ? reaches(i, tensor.numel(), count) : tensor.numel() <= count;
    if (!fits || tensor.numel() <= 0) {
      return invalid_argument("fusion group " + group + " was given an input of " +
                              std::to_string(tensor.numel()) + " elements for a result of " +
                              std::to_string(count));
    }
    buffers.push_back(tensor.data_ptr());
  }
  if (!out.is_contiguous() || out.numel() != count) {
    return invalid_argument("fusion group " + group + " was given an output that is not " +
                            std::to_string(count) + " contiguous elements");
  }
  buffers.push_back(out.data_ptr());

  if (!dynamic) {
    return parts_.program->launch(buffers, count, stream);
  }
  TESSIQ_RETURN_IF_ERROR(check_broadcast_output(tensors, out, group));
  const std::vector<std::int64_t> scalars = dynamic_scalars(reads, tensors, count);
  const auto blocks = (count + static_cast<std::int64_t>(codegen::kBlockSize) - 1) /
                      static_cast<std::int64_t>(codegen::kBlockSize);
  return parts_.program->launch(buffers, scalars, blocks, stream);
}

codegen::ReductionExtents ReductionKernel::extents() const {
  codegen::ReductionExtents known;
  known.outer = parts_.outer;
  known.reduce = parts_.reduce;
  known.inner = parts_.inner;
  return known;
}

std::int64_t ReductionKernel::blocks() const {
  return codegen::blocks_for(parts_.shape, extents());
}

/// The three extents this launch walks, taken from the tensor it was handed.
///
/// The reduced axes are fixed; only their extents change. Checked against what the description
/// knew.
Result<codegen::ReductionExtents> ReductionKernel::extents_for(const at::Tensor& in) const {
  if (std::cmp_not_equal(in.dim(), parts_.rank)) {
    return invalid_argument("reduction " + parts_.output + " was compiled for rank " +
                            std::to_string(parts_.rank) + " and handed rank " +
                            std::to_string(in.dim()));
  }
  const std::vector<std::int64_t> sizes(in.sizes().begin(), in.sizes().end());
  const codegen::ReductionExtents found =
      codegen::split_axes(sizes, parts_.first_axis, parts_.last_axis);
  if (!found.are_usable()) {
    return invalid_argument("reduction " + parts_.output + " was handed a shape it cannot walk");
  }

  // Known extents are checked too; a mismatch would read past the buffer.
  const std::array<std::pair<std::int64_t, std::int64_t>, 3> pairs{
      {{parts_.outer, found.outer}, {parts_.reduce, found.reduce}, {parts_.inner, found.inner}}};
  for (const auto& [described, actual] : pairs) {
    if (described != ir::kDynamic && described != actual) {
      return invalid_argument("reduction " + parts_.output + " was compiled for " +
                              std::to_string(described) + " and handed " + std::to_string(actual));
    }
  }
  return found;
}

Result<std::vector<std::int64_t>>
ReductionKernel::output_shape(const std::vector<at::Tensor>& tensors) const {
  if (tensors.size() != 1) {
    return invalid_argument("reduction " + parts_.output + " reads one tensor");
  }
  // The kept axes in order, with the reduced ones as 1 when `keeps_dims`.
  std::vector<std::int64_t> kept;
  const at::Tensor& in = tensors.front();
  for (std::size_t axis = 0; axis < parts_.rank; ++axis) {
    const bool folded = axis >= parts_.first_axis && axis <= parts_.last_axis;
    if (!folded) {
      kept.push_back(in.size(static_cast<std::int64_t>(axis)));
    } else if (parts_.keeps_dims) {
      kept.push_back(1);
    }
  }
  return kept;
}

Result<void> ReductionKernel::launch(const std::vector<at::Tensor>& tensors,
                                     const at::Tensor& out,
                                     codegen::Stream stream) const {
  if (parts_.program == nullptr) {
    return internal_error("reduction " + parts_.output + " has no compiled kernel");
  }
  if (tensors.size() != 1) {
    return invalid_argument("reduction " + parts_.output + " takes one input, got " +
                            std::to_string(tensors.size()));
  }
  const at::Tensor& in = tensors.front();
  TESSIQ_ASSIGN_OR_RETURN(const codegen::ReductionExtents extents, extents_for(in));
  const std::int64_t reads = extents.elements();
  const std::int64_t writes = extents.outputs();
  // The kernel indexes both buffers as packed.
  if (!in.is_contiguous() || in.numel() != reads) {
    return invalid_argument("reduction " + parts_.output + " expects " + std::to_string(reads) +
                            " contiguous input elements, got " + std::to_string(in.numel()));
  }
  if (!out.is_contiguous() || out.numel() != writes) {
    return invalid_argument("reduction " + parts_.output + " expects " + std::to_string(writes) +
                            " contiguous output elements, got " + std::to_string(out.numel()));
  }

  const std::vector<void*> buffers{in.data_ptr(), out.data_ptr()};
  return parts_.program->launch(buffers,
                                {extents.outer, extents.reduce, extents.inner},
                                codegen::blocks_for(parts_.shape, extents),
                                stream);
}

/// The step between rows of `tensor` as stored, which BLAS calls the leading dimension.
///
/// For a transposed operand the rows BLAS reads are the stored columns, so the other stride.
std::int64_t leading_of(const at::Tensor& tensor, bool transposed) {
  const auto rank = tensor.dim();
  const std::int64_t rows = tensor.size(rank - 2);
  const std::int64_t columns = tensor.size(rank - 1);
  // The stride of an axis of extent 1 is arbitrary (torch often reports 1). BLAS needs the packed
  // value, which is the other axis's extent.
  if (transposed) {
    return columns == 1 ? rows : tensor.stride(rank - 1);
  }
  return rows == 1 ? columns : tensor.stride(rank - 2);
}

/// Whether an extent the description knew matches the tensor's.
Result<void> agrees(const std::string& gemm,
                    std::string_view what,
                    std::int64_t described,
                    std::int64_t actual) {
  if (described != ir::kDynamic && described != actual) {
    return invalid_argument("gemm " + gemm + " was compiled for " + std::string(what) + " " +
                            std::to_string(described) + " and handed " + std::to_string(actual));
  }
  return ok();
}

/// The numbers the BLAS call needs, read off the tensors this call was handed.
///
/// A dynamic description does not hold them. When it does, they are checked against the tensors,
/// since a mismatch would make the multiply read the wrong bytes.
Result<codegen::GemmExtents> extents_for(const codegen::GemmKernel& description,
                                         const at::Tensor& lhs,
                                         const at::Tensor& rhs,
                                         const at::Tensor& out) {
  if (lhs.dim() < 2 || rhs.dim() < 2 || out.dim() < 2) {
    return invalid_argument("gemm " + description.output +
                            " was handed something that is not a "
                            "matrix");
  }

  codegen::GemmExtents extents;
  extents.m = out.size(out.dim() - 2);
  extents.n = out.size(out.dim() - 1);
  extents.k = lhs.size(lhs.dim() - 1);
  const std::int64_t plane = extents.m * extents.n;
  extents.batch = plane == 0 ? 0 : out.numel() / plane;
  extents.lhs_leading = leading_of(lhs, description.lhs.transposed);
  extents.rhs_leading = leading_of(rhs, description.rhs.transposed);

  TESSIQ_RETURN_IF_ERROR(agrees(description.output, "m", description.m, extents.m));
  TESSIQ_RETURN_IF_ERROR(agrees(description.output, "n", description.n, extents.n));
  TESSIQ_RETURN_IF_ERROR(agrees(description.output, "k", description.k, extents.k));
  TESSIQ_RETURN_IF_ERROR(
      agrees(description.output, "a batch of", description.batch, extents.batch));
  TESSIQ_RETURN_IF_ERROR(agrees(description.output,
                                "a left operand stepping by",
                                description.lhs.leading,
                                extents.lhs_leading));
  TESSIQ_RETURN_IF_ERROR(agrees(description.output,
                                "a right operand stepping by",
                                description.rhs.leading,
                                extents.rhs_leading));
  return extents;
}

std::vector<std::string> GemmKernel::inputs() const {
  const codegen::GemmKernel& description = parts_.description;
  std::vector<std::string> names{description.lhs.name, description.rhs.name};
  if (description.bias.has_value()) {
    names.push_back(*description.bias);
  }
  return names;
}

Result<std::vector<std::int64_t>>
GemmKernel::output_shape(const std::vector<at::Tensor>& tensors) const {
  if (tensors.size() < 2 || tensors[0].dim() < 2 || tensors[1].dim() < 2) {
    return invalid_argument("gemm " + output() + " cannot size a result from what it was given");
  }
  const at::Tensor& lhs = tensors[0];
  const at::Tensor& rhs = tensors[1];

  std::vector<std::int64_t> shape(lhs.sizes().begin(), lhs.sizes().end());
  shape.back() = rhs.size(rhs.dim() - 1);
  return shape;
}

Result<void> GemmKernel::launch(const std::vector<at::Tensor>& tensors,
                                const at::Tensor& out,
                                codegen::Stream stream) const {
  const codegen::GemmKernel& description = parts_.description;
  const std::vector<std::string> expected = inputs();
  if (tensors.size() != expected.size()) {
    return invalid_argument("gemm " + description.output + " takes " +
                            std::to_string(expected.size()) + " inputs, got " +
                            std::to_string(tensors.size()));
  }
  if (!out.is_contiguous()) {
    // The result is written row by row, so a caller expecting anything else would read it in the
    // wrong order.
    return invalid_argument("gemm " + description.output +
                            " was given an output that is not contiguous");
  }
  TESSIQ_ASSIGN_OR_RETURN(const codegen::GemmExtents extents,
                          extents_for(description, tensors[0], tensors[1], out));

  // The product is added to the output, so write zeros for a plain matmul or the bias in every row
  // for an addmm first.
  const std::int64_t total = extents.output_elements();
  if (description.bias.has_value()) {
    const at::Tensor& bias = tensors.back();
    if (bias.numel() != extents.n) {
      return invalid_argument("gemm " + description.output + " expects a bias of " +
                              std::to_string(extents.n) + ", got " + std::to_string(bias.numel()));
    }
    if (parts_.fill == nullptr) {
      return internal_error("gemm " + description.output +
                            " has a bias and no kernel to lay it "
                            "down");
    }
    const std::vector<void*> buffers{bias.data_ptr(), out.data_ptr()};
    TESSIQ_RETURN_IF_ERROR(
        parts_.fill->launch(buffers,
                            {total, extents.n},
                            (total + static_cast<std::int64_t>(codegen::kBlockSize) - 1) /
                                static_cast<std::int64_t>(codegen::kBlockSize),
                            stream));
  } else {
    TESSIQ_RETURN_IF_ERROR(codegen::zero_fill(out.data_ptr(),
                                              static_cast<std::size_t>(total) *
                                                  static_cast<std::size_t>(out.element_size()),
                                              parts_.ordinal,
                                              stream));
  }

  const codegen::GemmBuffers buffers{.lhs = tensors[0].const_data_ptr(),
                                     .rhs = tensors[1].const_data_ptr(),
                                     .output = out.data_ptr()};
  return codegen::launch_gemm(description, extents, buffers, parts_.ordinal, stream);
}

Result<std::vector<std::int64_t>>
ConvKernel::output_shape(const std::vector<at::Tensor>& /*tensors*/) const {
  // Fixed at compile time, since a convolution whose shapes were not known has no plan.
  const auto& shape = parts_.description.output_shape;
  return std::vector<std::int64_t>(shape.begin(), shape.end());
}

Result<void> ConvKernel::launch(const std::vector<at::Tensor>& tensors,
                                const at::Tensor& out,
                                codegen::Stream stream) const {
  const codegen::ConvKernel& description = parts_.description;
  if (parts_.plan == nullptr) {
    return internal_error("convolution " + description.output + " has no plan");
  }
  const std::size_t expected = description.bias.has_value() ? 3U : 2U;
  if (tensors.size() != expected) {
    return invalid_argument("convolution " + description.output + " takes " +
                            std::to_string(expected) + " inputs, got " +
                            std::to_string(tensors.size()));
  }

  // The plan describes packed NCHW tensors of exactly these extents.
  const auto matches = [](const at::Tensor& tensor, const std::array<std::int64_t, 4>& shape) {
    return tensor.is_contiguous() && tensor.dim() == 4 &&
           std::equal(shape.begin(), shape.end(), tensor.sizes().begin());
  };
  if (!matches(tensors[0], description.input_shape) ||
      !matches(tensors[1], description.weight_shape) || !matches(out, description.output_shape)) {
    return invalid_argument("convolution " + description.output +
                            " was handed tensors that do not match its plan");
  }

  const void* bias = description.bias.has_value() ? tensors[2].data_ptr() : nullptr;
  return parts_.plan->launch(
      tensors[0].data_ptr(), tensors[1].data_ptr(), bias, out.data_ptr(), stream);
}

namespace {

using Kernels = std::vector<std::shared_ptr<Kernel>>;

Result<void> compile_fused(const ir::Graph& graph, int ordinal, Kernels& compiled) {
  if (!codegen::launcher_available()) {
    return ok();
  }
  const auto groups = codegen::fusion_groups(graph);
  if (groups.empty()) {
    return ok();
  }

  TESSIQ_ASSIGN_OR_RETURN(const auto capability, codegen::device_capability(ordinal));

  std::size_t count = 0;
  for (const std::int64_t group : groups) {
    TESSIQ_ASSIGN_OR_RETURN(const codegen::FusedKernel description,
                            codegen::extract_kernel(graph, group));

    const codegen::Emitter& emitter = codegen::cuda_emitter();
    if (!emitter.supports(description)) {
      // Not a failure; the operations run one at a time.
      TESSIQ_LOG(kDebug, kLogModule) << "no cuda code for group " << group << ", leaving it alone";
      continue;
    }

    TESSIQ_ASSIGN_OR_RETURN(const std::string source, emitter.emit(description));
    TESSIQ_ASSIGN_OR_RETURN(
        const codegen::Cubin cubin,
        codegen::compile_cached(source, description.name, capability.first, capability.second));
    TESSIQ_ASSIGN_OR_RETURN(const codegen::Program* program,
                            codegen::load_program(cubin, description.name, ordinal));

    std::vector<std::string> members;
    members.reserve(description.body.size());
    for (const codegen::Step& step : description.body) {
      members.push_back(step.result);
    }
    members.insert(members.end(), description.absorbed.begin(), description.absorbed.end());

    std::vector<codegen::OperandRead> reads;
    reads.reserve(description.inputs.size());
    for (const codegen::KernelInput& input : description.inputs) {
      reads.push_back(input.read);
    }

    compiled.push_back(std::make_shared<FusedKernel>(FusedKernel::Parts{
        .group = group,
        .inputs = description.input_names(),
        .output = description.output,
        .members = std::move(members),
        .elements = description.elements,
        .reads = std::move(reads),
        .program = program,
    }));
    ++count;
  }

  TESSIQ_LOG(kDebug, kLogModule) << "compiled " << count << " of " << groups.size() << " group(s)";
  return ok();
}

/// Compiles one reduction, or nothing when it has no kernel shape or no code the emitter writes.
Result<std::shared_ptr<Kernel>>
compile_one(const ir::Node& node, std::pair<int, int> capability, int ordinal) {
  TESSIQ_ASSIGN_OR_RETURN(const auto description, codegen::extract_reduction(node));
  if (!description.has_value()) {
    TESSIQ_LOG(kDebug, kLogModule) << "no kernel shape for " << node.op() << ", leaving it alone";
    return std::shared_ptr<Kernel>{};
  }

  const codegen::Emitter& emitter = codegen::cuda_emitter();
  if (!emitter.supports(*description)) {
    TESSIQ_LOG(kDebug, kLogModule) << "no cuda code for " << description->name;
    return std::shared_ptr<Kernel>{};
  }

  TESSIQ_ASSIGN_OR_RETURN(const std::string source, emitter.emit(*description));
  TESSIQ_ASSIGN_OR_RETURN(
      const codegen::Cubin cubin,
      codegen::compile_cached(source, description->name, capability.first, capability.second));
  TESSIQ_ASSIGN_OR_RETURN(const codegen::Program* program,
                          codegen::load_program(cubin, description->name, ordinal));

  return std::shared_ptr<Kernel>(std::make_shared<ReductionKernel>(
      ReductionKernel::Parts{.output = description->output,
                             .input = description->input,
                             .outer = description->outer,
                             .reduce = description->reduce,
                             .inner = description->inner,
                             .first_axis = description->first_axis,
                             .last_axis = description->last_axis,
                             .rank = description->rank,
                             .keeps_dims = description->keeps_dims,
                             .shape = description->shape,
                             .program = program}));
}

/// Whether `mark-supported` claimed `node`.
///
/// An unclaimed node runs on the host, and a kernel for it would be launched with host pointers.
/// A node can be declined even when its kind has kernels, such as a float64 matmul.
bool claimed(const ir::Node& node) {
  const ir::Attribute* supported = node.attribute("tessiq.supported");
  const auto* flag = supported == nullptr ? nullptr : std::get_if<bool>(supported);
  return flag != nullptr && *flag;
}

Result<void> compile_reductions(const ir::Graph& graph, int ordinal, Kernels& compiled) {
  if (!codegen::launcher_available()) {
    return ok();
  }
  const std::vector<const ir::Node*> candidates = codegen::reduction_nodes(graph);
  if (candidates.empty()) {
    return ok();
  }

  TESSIQ_ASSIGN_OR_RETURN(const auto capability, codegen::device_capability(ordinal));

  std::size_t count = 0;
  for (const ir::Node* node : candidates) {
    if (!claimed(*node)) {
      continue;
    }
    TESSIQ_ASSIGN_OR_RETURN(auto kernel, compile_one(*node, capability, ordinal));
    if (kernel != nullptr) {
      compiled.push_back(std::move(kernel));
      ++count;
    }
  }

  TESSIQ_LOG(kDebug, kLogModule) << "compiled " << count << " of " << candidates.size()
                                 << " reduction(s)";
  return ok();
}

Result<void> compile_gemms(const ir::Graph& graph, int ordinal, Kernels& compiled) {
  if (!codegen::blas_available()) {
    return ok();
  }
  const std::vector<const ir::Node*> candidates = codegen::gemm_nodes(graph);
  if (candidates.empty()) {
    return ok();
  }
  TESSIQ_ASSIGN_OR_RETURN(const auto capability, codegen::device_capability(ordinal));

  std::size_t count = 0;
  for (const ir::Node* node : candidates) {
    if (!claimed(*node)) {
      continue;
    }
    TESSIQ_ASSIGN_OR_RETURN(const auto description, codegen::extract_gemm(*node));
    if (!description.has_value()) {
      TESSIQ_LOG(kDebug, kLogModule) << "no gemm shape for " << node->op() << ", leaving it alone";
      continue;
    }
    const codegen::Program* fill = nullptr;
    if (description->bias.has_value()) {
      const std::string name = "tessiq_bias_" + description->output;
      const std::string source = codegen::broadcast_row_source(description->dtype, name);
      TESSIQ_ASSIGN_OR_RETURN(
          const codegen::Cubin cubin,
          codegen::compile_cached(source, name, capability.first, capability.second));
      TESSIQ_ASSIGN_OR_RETURN(fill, codegen::load_program(cubin, name, ordinal));
    }
    compiled.push_back(std::make_shared<GemmKernel>(
        GemmKernel::Parts{.description = *description, .ordinal = ordinal, .fill = fill}));
    ++count;
  }

  TESSIQ_LOG(kDebug, kLogModule) << "described " << count << " of " << candidates.size()
                                 << " matmul(s)";
  return ok();
}

Result<void> compile_convs(const ir::Graph& graph, int ordinal, Kernels& compiled) {
  if (!codegen::cudnn_available()) {
    return ok();
  }

  const std::vector<const ir::Node*> candidates = codegen::conv_nodes(graph);
  std::size_t count = 0;
  for (const ir::Node* node : candidates) {
    if (!claimed(*node)) {
      continue;
    }
    TESSIQ_ASSIGN_OR_RETURN(const auto description, codegen::extract_conv(*node));
    if (!description.has_value()) {
      TESSIQ_LOG(kDebug, kLogModule) << "no conv shape for " << node->op() << ", leaving it alone";
      continue;
    }
    auto plan = codegen::make_conv_plan(*description, ordinal);
    if (!plan.has_value()) {
      TESSIQ_LOG(kDebug, kLogModule)
          << "no cuDNN plan for " << description->output << ": " << plan.error().message();
      continue;
    }
    compiled.push_back(std::make_shared<ConvKernel>(ConvKernel::Parts{
        .description = *description,
        .plan = std::shared_ptr<const codegen::ConvPlan>(std::move(plan).value()),
    }));
    ++count;
  }

  TESSIQ_LOG(kDebug, kLogModule) << "planned " << count << " of " << candidates.size()
                                 << " convolution(s)";
  return ok();
}

} // namespace

Result<std::vector<std::shared_ptr<Kernel>>> compile_kernels(const ir::Graph& graph, int ordinal) {
  Kernels compiled;
  TESSIQ_RETURN_IF_ERROR(compile_fused(graph, ordinal, compiled));
  TESSIQ_RETURN_IF_ERROR(compile_reductions(graph, ordinal, compiled));
  TESSIQ_RETURN_IF_ERROR(compile_gemms(graph, ordinal, compiled));
  TESSIQ_RETURN_IF_ERROR(compile_convs(graph, ordinal, compiled));
  return compiled;
}

} // namespace tessiq::torch_backend
