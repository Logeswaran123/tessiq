// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Reductions called outside a compiled graph.

#include "tessiq/torch/eager_reduction.h"

#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/amax_ops.h>
#include <ATen/ops/amin_ops.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mean_ops.h>
#include <ATen/ops/sum_ops.h>
#include <algorithm>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "tessiq/codegen/cubin_cache.h"
#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/program.h"
#include "tessiq/codegen/reduction.h"
#include "tessiq/codegen/stream.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/backend.h"
#include "tessiq/torch/fallback.h"
#include "tessiq/torch/importer.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "eager";

/// One compiled kernel for each combination, element type, thread layout and CUDA device.
using KernelKey = std::tuple<codegen::ReduceOp, at::ScalarType, codegen::ReductionShape, int>;

struct KernelCache {
  std::mutex mutex;
  /// Null for a kernel that did not compile, so it is not tried again. Guarded by `mutex`.
  std::map<KernelKey, const codegen::Program*> programs;
};

KernelCache& kernel_cache() {
  static KernelCache cache;
  return cache;
}

std::string_view op_name(codegen::ReduceOp op) {
  switch (op) {
  case codegen::ReduceOp::kSum:
    return "sum";
  case codegen::ReduceOp::kMean:
    return "mean";
  case codegen::ReduceOp::kMax:
    return "amax";
  case codegen::ReduceOp::kMin:
    return "amin";
  }
  return "?";
}

/// Whether torch gives a reduction of `dtype` the same type back, which the kernel needs. torch
/// sums integers narrower than int64 into int64, and takes the mean of floating point only.
bool keeps_type(codegen::ReduceOp op, at::ScalarType dtype) {
  switch (dtype) {
  case at::kFloat:
  case at::kDouble:
    return true;
  case at::kLong:
    return op != codegen::ReduceOp::kMean;
  case at::kInt:
    return op == codegen::ReduceOp::kMax || op == codegen::ReduceOp::kMin;
  default:
    return false;
  }
}

/// The kernel for `op` over `dtype` with threads laid out as `shape`, on CUDA device `ordinal`.
/// Null when it does not compile.
const codegen::Program* program_for(codegen::ReduceOp op,
                                    at::ScalarType dtype,
                                    codegen::ReductionShape shape,
                                    int ordinal) {
  KernelKey key{op, dtype, shape, ordinal};
  KernelCache& cache = kernel_cache();
  const std::scoped_lock guard(cache.mutex);
  if (const auto found = cache.programs.find(key); found != cache.programs.end()) {
    return found->second;
  }

  const auto compile = [&]() -> Result<const codegen::Program*> {
    codegen::ReductionKernel kernel;
    TESSIQ_ASSIGN_OR_RETURN(kernel.dtype, to_ir_dtype(dtype));
    kernel.name = "tessiq_eager_" + std::string(op_name(op)) + "_" +
                  std::string(ir::dtype_name(kernel.dtype)) +
                  (shape == codegen::ReductionShape::kBlockPerOutput ? "_block" : "_thread");
    kernel.op = op;
    kernel.shape = shape;
    // The extents are kernel arguments, so one kernel serves every shape.
    kernel.outer = ir::kDynamic;
    kernel.reduce = ir::kDynamic;
    kernel.inner = ir::kDynamic;
    kernel.input = "x";
    kernel.output = "y";
    const codegen::Emitter& emitter = codegen::cuda_emitter();
    if (!emitter.supports(kernel)) {
      return unsupported("no cuda code for " + kernel.name);
    }
    TESSIQ_ASSIGN_OR_RETURN(const std::string source, emitter.emit(kernel));
    TESSIQ_ASSIGN_OR_RETURN(const auto capability, codegen::device_capability(ordinal));
    TESSIQ_ASSIGN_OR_RETURN(
        const codegen::Cubin cubin,
        codegen::compile_cached(source, kernel.name, capability.first, capability.second));
    return codegen::load_program(cubin, kernel.name, ordinal);
  };

  const Result<const codegen::Program*> compiled = compile();
  const codegen::Program* program = compiled.has_value() ? *compiled : nullptr;
  if (!compiled.has_value()) {
    TESSIQ_LOG(kDebug, kLogModule)
        << op_name(op) << " has no eager kernel: " << compiled.error().message();
  }
  cache.programs.emplace(key, program);
  return program;
}

/// The reduced axes as an inclusive range, or nothing when they are not one. No axes means all of
/// them, as torch reads `dim=None` and `dim=[]`.
std::optional<std::pair<std::size_t, std::size_t>> axis_range(const at::Tensor& self,
                                                              at::IntArrayRef dims) {
  const auto rank = static_cast<std::size_t>(self.dim());
  if (dims.empty()) {
    return std::pair{std::size_t{0}, rank - 1};
  }
  std::vector<std::size_t> axes;
  for (const std::int64_t dim : dims) {
    if (dim < -self.dim() || dim >= self.dim()) {
      return std::nullopt;
    }
    axes.push_back(static_cast<std::size_t>(at::maybe_wrap_dim(dim, self.dim())));
  }
  std::ranges::sort(axes);
  if (std::ranges::adjacent_find(axes) != axes.end() ||
      axes.back() - axes.front() + 1 != axes.size()) {
    return std::nullopt;
  }
  return std::pair{axes.front(), axes.back()};
}

/// Below this many outputs most of the GPU idles while a few threads or blocks walk the reduced
/// axis, so a long one is split.
constexpr std::int64_t kFewOutputs = 1024;
constexpr std::int64_t kLongAxis = 16384;

/// The chunk to split a reduced axis of `reduce` elements into: its largest divisor from 256 to
/// 4096, or nothing when it has none.
std::optional<std::int64_t> chunk_for(std::int64_t reduce) {
  for (std::int64_t chunk = 4096; chunk >= 256; --chunk) {
    if (reduce % chunk == 0) {
      return chunk;
    }
  }
  return std::nullopt;
}

/// Reduces packed `in`, read as `extents`, into packed `out`. False, with nothing written, when a
/// kernel does not compile.
///
/// A long axis with few outputs is reduced in two passes: each chunk into a partial, then the
/// partials. Equal chunks keep a mean exact, and each thread adds a few thousand elements rather
/// than millions, which is also what keeps a float sum close to the host's.
bool launch(codegen::ReduceOp op,
            int ordinal,
            const at::Tensor& in,
            const at::Tensor& out,
            const codegen::ReductionExtents& extents) {
  if (extents.outputs() < kFewOutputs && extents.reduce >= kLongAxis) {
    if (const std::optional<std::int64_t> chunk = chunk_for(extents.reduce)) {
      const std::int64_t count = extents.reduce / *chunk;
      const codegen::ReductionExtents chunks{
          .outer = extents.outer * count, .reduce = *chunk, .inner = extents.inner};
      const codegen::ReductionExtents partials{
          .outer = extents.outer, .reduce = count, .inner = extents.inner};
      const at::Tensor partial = at::empty({chunks.outputs()}, in.options());
      return launch(op, ordinal, in, partial, chunks) &&
             launch(op, ordinal, partial, out, partials);
    }
  }

  const codegen::ReductionShape shape = extents.inner == 1
                                            ? codegen::ReductionShape::kBlockPerOutput
                                            : codegen::ReductionShape::kThreadPerOutput;
  const codegen::Program* program = program_for(op, in.scalar_type(), shape, ordinal);
  if (program == nullptr) {
    return false;
  }
  const Result<void> launched = program->launch({in.data_ptr(), out.data_ptr()},
                                                {extents.outer, extents.reduce, extents.inner},
                                                codegen::blocks_for(shape, extents),
                                                codegen::kDefaultStream);
  TORCH_CHECK(launched.has_value(), launched.has_value() ? "" : launched.error().to_string());
  return true;
}

/// `op` over `dims` of `self`, or nothing when the call has to run on the host.
std::optional<at::Tensor> reduce(const char* spelling,
                                 codegen::ReduceOp op,
                                 const at::Tensor& self,
                                 at::IntArrayRef dims,
                                 bool keepdim) {
  const passes::OpInfo* info = passes::find_builtin_op(spelling);
  const std::optional<int> ordinal = cuda_ordinal_of(self.device());
  const Result<ir::DType> type = to_ir_dtype(self.scalar_type());
  if (!ordinal.has_value() || info == nullptr || !info->has_kernel || !type.has_value() ||
      !info->accepts(*type) || !keeps_type(op, self.scalar_type()) || self.dim() == 0 ||
      self.numel() == 0 || !self.is_contiguous()) {
    return std::nullopt;
  }
  const std::optional<std::pair<std::size_t, std::size_t>> range = axis_range(self, dims);
  if (!range.has_value()) {
    return std::nullopt;
  }
  const auto [first, last] = *range;
  const std::vector<std::int64_t> sizes(self.sizes().begin(), self.sizes().end());
  const codegen::ReductionExtents extents = codegen::split_axes(sizes, first, last);
  if (!extents.are_usable()) {
    return std::nullopt;
  }
  std::vector<std::int64_t> kept;
  for (std::size_t axis = 0; axis < sizes.size(); ++axis) {
    if (axis < first || axis > last) {
      kept.push_back(sizes[axis]);
    } else if (keepdim) {
      kept.push_back(1);
    }
  }
  at::Tensor out = at::empty(kept, self.options());
  if (!launch(op, *ordinal, self, out, extents)) {
    return std::nullopt;
  }
  return out;
}

at::Tensor sum(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
  if (!dtype.has_value()) {
    if (std::optional<at::Tensor> out =
            reduce("aten.sum.default", codegen::ReduceOp::kSum, self, {}, false)) {
      return *out;
    }
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::sum>::call(self, dtype);
}

at::Tensor sum_dim(const at::Tensor& self,
                   at::OptionalIntArrayRef dim,
                   bool keepdim,
                   std::optional<at::ScalarType> dtype) {
  if (!dtype.has_value()) {
    if (std::optional<at::Tensor> out = reduce("aten.sum.dim_IntList",
                                               codegen::ReduceOp::kSum,
                                               self,
                                               dim.value_or(at::IntArrayRef{}),
                                               keepdim)) {
      return *out;
    }
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::sum_dim_IntList>::call(
      self, dim, keepdim, dtype);
}

at::Tensor mean(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
  if (!dtype.has_value()) {
    if (std::optional<at::Tensor> out =
            reduce("aten.mean.default", codegen::ReduceOp::kMean, self, {}, false)) {
      return *out;
    }
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::mean>::call(self, dtype);
}

at::Tensor mean_dim(const at::Tensor& self,
                    at::OptionalIntArrayRef dim,
                    bool keepdim,
                    std::optional<at::ScalarType> dtype) {
  if (!dtype.has_value()) {
    if (std::optional<at::Tensor> out = reduce("aten.mean.dim",
                                               codegen::ReduceOp::kMean,
                                               self,
                                               dim.value_or(at::IntArrayRef{}),
                                               keepdim)) {
      return *out;
    }
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::mean_dim>::call(
      self, dim, keepdim, dtype);
}

at::Tensor amax(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
  if (std::optional<at::Tensor> out =
          reduce("aten.amax.default", codegen::ReduceOp::kMax, self, dim, keepdim)) {
    return *out;
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::amax>::call(self, dim, keepdim);
}

at::Tensor amin(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
  if (std::optional<at::Tensor> out =
          reduce("aten.amin.default", codegen::ReduceOp::kMin, self, dim, keepdim)) {
    return *out;
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::amin>::call(self, dim, keepdim);
}

} // namespace

Result<void> register_eager_reductions(torch::Library& library) {
  try {
    library.impl("sum", &sum);
    library.impl("sum.dim_IntList", &sum_dim);
    library.impl("mean", &mean);
    library.impl("mean.dim", &mean_dim);
    library.impl("amax", &amax);
    library.impl("amin", &amin);
  } catch (const c10::Error& error) {
    return invalid_argument(error.what_without_backtrace());
  }
  return ok();
}

} // namespace tessiq::torch_backend
