// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Normalisation called outside a compiled graph, one kernel launch each.

#include "tessiq/torch/eager_norm.h"

#include <ATen/core/Tensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/native_batch_norm_ops.h>
#include <ATen/ops/native_layer_norm_ops.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tessiq/codegen/norm.h"
#include "tessiq/codegen/program.h"
#include "tessiq/codegen/stream.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/backend.h"
#include "tessiq/torch/fallback.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "eager";

using Normalised = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

/// A kernel, by its entry point.
enum class Norm {
  /// Batch norm, a block per plane of one channel.
  kBatchByPlane,
  /// Batch norm, a thread per element, for planes narrower than a block.
  kBatchByElement,
  kLayer,
};

/// One compiled kernel for each normalisation, element type and CUDA device.
using KernelKey = std::tuple<Norm, at::ScalarType, int>;

struct KernelCache {
  std::mutex mutex;
  /// Null for a kernel that did not compile, so it is not tried again. Guarded by `mutex`.
  std::map<KernelKey, const codegen::Program*> programs;
};

KernelCache& kernel_cache() {
  static KernelCache cache;
  return cache;
}

/// The kernel for `norm` over `dtype`, a float type, on CUDA device `ordinal`. Null when it does
/// not compile.
const codegen::Program* program_for(Norm norm, at::ScalarType dtype, int ordinal) {
  const KernelKey key{norm, dtype, ordinal};
  KernelCache& cache = kernel_cache();
  const std::scoped_lock guard(cache.mutex);
  if (const auto found = cache.programs.find(key); found != cache.programs.end()) {
    return found->second;
  }
  const ir::DType type = dtype == at::kDouble ? ir::DType::kF64 : ir::DType::kF32;
  const std::string suffix(ir::dtype_name(type));
  std::string source;
  std::string entry;
  if (norm == Norm::kLayer) {
    entry = "tessiq_layer_norm_" + suffix;
    source = codegen::layer_norm_source(type, entry);
  } else {
    // Both entry points come from one source, which the cubin cache compiles once.
    const std::string name = "tessiq_batch_norm_" + suffix;
    entry = name + (norm == Norm::kBatchByPlane ? "_by_plane" : "_by_element");
    source = codegen::batch_norm_source(type, name);
  }
  const Result<const codegen::Program*> compiled = codegen::compile_program(source, entry, ordinal);
  const codegen::Program* program = compiled.has_value() ? *compiled : nullptr;
  if (!compiled.has_value()) {
    TESSIQ_LOG(kDebug, kLogModule) << entry << " does not compile: " << compiled.error().message();
  }
  cache.programs.emplace(key, program);
  return program;
}

/// `eps` as the kernel's element type, as torch converts it.
codegen::KernelArgument epsilon(double eps, at::ScalarType dtype) {
  if (dtype == at::kDouble) {
    return eps;
  }
  return static_cast<float>(eps);
}

/// The CUDA ordinal `input` is on, when it is a packed, non-empty float tensor on a GPU.
std::optional<int> runs_on(const at::Tensor& input) {
  if ((input.scalar_type() != at::kFloat && input.scalar_type() != at::kDouble) ||
      !input.is_contiguous() || input.numel() == 0) {
    return std::nullopt;
  }
  return cuda_ordinal_of(input.device());
}

/// `parameter`, or an undefined tensor when there is none, as torch passes either.
at::Tensor given(const std::optional<at::Tensor>& parameter) {
  return parameter.has_value() ? *parameter : at::Tensor();
}

/// Whether `parameter`, when there is one, is `extent` values beside `input`.
bool fits(const at::Tensor& parameter, const at::Tensor& input, std::int64_t extent) {
  return !parameter.defined() ||
         (parameter.device() == input.device() && parameter.scalar_type() == input.scalar_type() &&
          parameter.numel() == extent);
}

/// `parameter` packed, or a null pointer for the kernel to skip it.
void* data_of(const at::Tensor& parameter) {
  return parameter.defined() ? parameter.data_ptr() : nullptr;
}

void raise_if_error(const Result<void>& result) {
  TORCH_CHECK(result.has_value(), result.has_value() ? "" : result.error().to_string());
}

std::optional<Normalised> batch_norm_on_gpu(const at::Tensor& input,
                                            const at::Tensor& weight,
                                            const at::Tensor& bias,
                                            const at::Tensor& running_mean,
                                            const at::Tensor& running_var,
                                            bool training,
                                            double eps) {
  const std::optional<int> ordinal = runs_on(input);
  if (training || !ordinal.has_value() || input.dim() < 2 || !running_mean.defined() ||
      !running_var.defined()) {
    return std::nullopt;
  }
  const std::int64_t channels = input.size(1);
  for (const at::Tensor* parameter : {&weight, &bias, &running_mean, &running_var}) {
    if (!fits(*parameter, input, channels)) {
      return std::nullopt;
    }
  }
  const std::int64_t planes = input.size(0) * channels;
  const std::int64_t inner = input.numel() / planes;
  const bool by_plane = std::cmp_greater_equal(inner, codegen::kNormBlockSize);
  if ((by_plane && std::cmp_greater(planes, std::numeric_limits<std::uint32_t>::max())) ||
      (!by_plane && input.numel() > std::numeric_limits<std::int32_t>::max())) {
    return std::nullopt;
  }
  const codegen::Program* program = program_for(
      by_plane ? Norm::kBatchByPlane : Norm::kBatchByElement, input.scalar_type(), *ordinal);
  if (program == nullptr) {
    return std::nullopt;
  }

  // The per-channel tensors are small, so packing any that are not costs nothing.
  const at::Tensor packed_weight = weight.defined() ? weight.contiguous() : weight;
  const at::Tensor packed_bias = bias.defined() ? bias.contiguous() : bias;
  const at::Tensor packed_mean = running_mean.contiguous();
  const at::Tensor packed_var = running_var.contiguous();
  at::Tensor out = at::empty(input.sizes(), input.options());
  codegen::LaunchShape shape;
  shape.block[0] = codegen::kNormBlockSize;
  std::vector<codegen::KernelArgument> arguments;
  if (by_plane) {
    shape.grid[0] = static_cast<std::uint32_t>(planes);
    arguments = {channels, inner};
  } else {
    const std::int64_t total = input.numel();
    shape.grid[0] =
        static_cast<std::uint32_t>((total + codegen::kNormBlockSize - 1) / codegen::kNormBlockSize);
    arguments = {static_cast<std::int32_t>(total),
                 static_cast<std::int32_t>(channels),
                 static_cast<std::int32_t>(inner)};
  }
  arguments.push_back(epsilon(eps, input.scalar_type()));
  raise_if_error(program->launch({input.data_ptr(),
                                  data_of(packed_weight),
                                  data_of(packed_bias),
                                  packed_mean.data_ptr(),
                                  packed_var.data_ptr(),
                                  out.data_ptr()},
                                 arguments,
                                 shape,
                                 codegen::kDefaultStream));
  // In eval mode torch saves no statistics, and returns empty tensors in their place.
  return Normalised{out, at::empty({0}, input.options()), at::empty({0}, input.options())};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_batch_norm(const at::Tensor& input,
                  const std::optional<at::Tensor>& weight,
                  const std::optional<at::Tensor>& bias,
                  const std::optional<at::Tensor>& running_mean,
                  const std::optional<at::Tensor>& running_var,
                  bool training,
                  double momentum,
                  double eps) {
  if (std::optional<Normalised> out = batch_norm_on_gpu(input,
                                                        given(weight),
                                                        given(bias),
                                                        given(running_mean),
                                                        given(running_var),
                                                        training,
                                                        eps)) {
    return *out;
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::native_batch_norm>::call(
      input, weight, bias, running_mean, running_var, training, momentum, eps);
}

/// The input read as rows of the normalised extent, one block each. torch's CPU kernel also
/// returns each row's mean and reciprocal deviation, shaped to broadcast against the input, for
/// the backward pass.
std::optional<Normalised> layer_norm_on_gpu(const at::Tensor& input,
                                            c10::SymIntArrayRef normalized_shape,
                                            const at::Tensor& weight,
                                            const at::Tensor& bias,
                                            double eps) {
  const std::optional<int> ordinal = runs_on(input);
  const std::optional<c10::IntArrayRef> shape = c10::asIntArrayRefSlowOpt(normalized_shape);
  if (!ordinal.has_value() || !shape.has_value() || shape->empty() ||
      shape->size() > static_cast<std::size_t>(input.dim()) ||
      input.sizes().slice(static_cast<std::size_t>(input.dim()) - shape->size()) != *shape) {
    return std::nullopt;
  }
  const std::int64_t width = c10::multiply_integers(*shape);
  const std::int64_t rows = input.numel() / width;
  if (!fits(weight, input, width) || !fits(bias, input, width) ||
      std::cmp_greater(rows, std::numeric_limits<std::uint32_t>::max())) {
    return std::nullopt;
  }
  const codegen::Program* program = program_for(Norm::kLayer, input.scalar_type(), *ordinal);
  if (program == nullptr) {
    return std::nullopt;
  }

  const at::Tensor packed_weight = weight.defined() ? weight.contiguous() : weight;
  const at::Tensor packed_bias = bias.defined() ? bias.contiguous() : bias;
  at::Tensor out = at::empty(input.sizes(), input.options());
  std::vector<std::int64_t> statistics(
      input.sizes().begin(), input.sizes().end() - static_cast<std::ptrdiff_t>(shape->size()));
  statistics.resize(static_cast<std::size_t>(input.dim()), 1);
  at::Tensor mean = at::empty(statistics, input.options());
  at::Tensor rstd = at::empty(statistics, input.options());

  codegen::LaunchShape launch;
  launch.grid[0] = static_cast<std::uint32_t>(rows);
  launch.block[0] = codegen::kNormBlockSize;
  raise_if_error(program->launch({input.data_ptr(),
                                  data_of(packed_weight),
                                  data_of(packed_bias),
                                  out.data_ptr(),
                                  mean.data_ptr(),
                                  rstd.data_ptr()},
                                 {width, epsilon(eps, input.scalar_type())},
                                 launch,
                                 codegen::kDefaultStream));
  return Normalised{out, mean, rstd};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
native_layer_norm(const at::Tensor& input,
                  c10::SymIntArrayRef normalized_shape,
                  const std::optional<at::Tensor>& weight,
                  const std::optional<at::Tensor>& bias,
                  double eps) {
  if (std::optional<Normalised> out =
          layer_norm_on_gpu(input, normalized_shape, given(weight), given(bias), eps)) {
    return *out;
  }
  return at::native::call_fallback_fn_symint<&cpu_fallback, at::_ops::native_layer_norm>::call(
      input, normalized_shape, weight, bias, eps);
}

} // namespace

Result<void> register_eager_normalization(torch::Library& library) {
  try {
    library.impl("native_batch_norm", &native_batch_norm);
    library.impl("native_layer_norm", &native_layer_norm);
  } catch (const c10::Error& error) {
    return invalid_argument(error.what_without_backtrace());
  }
  return ok();
}

} // namespace tessiq::torch_backend
