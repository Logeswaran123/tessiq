// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Convolutions called outside a compiled graph.

#include "tessiq/torch/eager_conv.h"

#include <ATen/core/Tensor.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/empty.h>
#include <algorithm>
#include <array>
#include <c10/util/Exception.h>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>

#include "tessiq/codegen/conv.h"
#include "tessiq/codegen/cudnn.h"
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

/// Everything a plan depends on: the shapes, the convolution's parameters, whether it has a bias,
/// and the CUDA device.
using PlanKey = std::tuple<std::array<std::int64_t, 4>,
                           std::array<std::int64_t, 4>,
                           std::array<std::int64_t, 2>,
                           std::array<std::int64_t, 2>,
                           std::array<std::int64_t, 2>,
                           std::int64_t,
                           bool,
                           int>;

struct PlanCache {
  std::mutex mutex;
  /// Null for a convolution cuDNN has no plan for, so it is not asked again. Guarded by `mutex`.
  std::map<PlanKey, std::shared_ptr<const codegen::ConvPlan>> plans;
};

PlanCache& plan_cache() {
  static PlanCache cache;
  return cache;
}

std::array<std::int64_t, 4> four(c10::IntArrayRef sizes) {
  return {sizes[0], sizes[1], sizes[2], sizes[3]};
}

std::array<std::int64_t, 2> two(c10::IntArrayRef values) { return {values[0], values[1]}; }

/// The output's extent along one spatial axis, as torch works it out. Zero when the window does not
/// fit or the stride is not positive, so the convolution goes to the host and torch reports it.
std::int64_t output_extent(std::int64_t input,
                           std::int64_t kernel,
                           std::int64_t stride,
                           std::int64_t padding,
                           std::int64_t dilation) {
  const std::int64_t window = (dilation * (kernel - 1)) + 1;
  const std::int64_t span = input + (2 * padding) - window;
  return span < 0 || stride < 1 ? 0 : (span / stride) + 1;
}

/// The convolution as cuDNN takes it, or nothing when it is not one this runs: a 2-D, untransposed
/// convolution of packed NCHW tensors, all on one GPU, of a type the convolution's ops.toml row
/// lists, whose shapes torch would accept.
std::optional<codegen::ConvKernel> describe(const at::Tensor& input,
                                            const at::Tensor& weight,
                                            const std::optional<at::Tensor>& bias,
                                            at::IntArrayRef stride,
                                            at::IntArrayRef padding,
                                            at::IntArrayRef dilation,
                                            bool transposed,
                                            int64_t groups) {
  const passes::OpInfo* info = passes::find_builtin_op("aten.convolution.default");
  const Result<ir::DType> type = to_ir_dtype(input.scalar_type());
  if (!codegen::cudnn_available() || info == nullptr || !info->has_kernel || !type.has_value() ||
      !info->accepts(*type) || transposed || input.dim() != 4 || weight.dim() != 4 ||
      stride.size() != 2 || padding.size() != 2 || dilation.size() != 2 || groups < 1 ||
      !input.is_contiguous() || weight.device() != input.device() ||
      weight.scalar_type() != input.scalar_type()) {
    return std::nullopt;
  }
  if (bias.has_value() &&
      (bias->device() != input.device() || bias->scalar_type() != input.scalar_type() ||
       bias->dim() != 1 || bias->size(0) != weight.size(0))) {
    return std::nullopt;
  }
  if (input.size(1) != weight.size(1) * groups || weight.size(0) % groups != 0) {
    return std::nullopt;
  }

  codegen::ConvKernel kernel;
  kernel.input = "input";
  kernel.weight = "weight";
  kernel.output = "output";
  if (bias.has_value()) {
    kernel.bias = "bias";
  }
  kernel.dtype = *type;
  kernel.input_shape = four(input.sizes());
  kernel.weight_shape = four(weight.sizes());
  kernel.stride = two(stride);
  kernel.padding = two(padding);
  kernel.dilation = two(dilation);
  kernel.groups = groups;
  kernel.output_shape = {
      input.size(0),
      weight.size(0),
      output_extent(input.size(2), weight.size(2), stride[0], padding[0], dilation[0]),
      output_extent(input.size(3), weight.size(3), stride[1], padding[1], dilation[1])};
  if (std::ranges::any_of(kernel.output_shape, [](std::int64_t extent) { return extent <= 0; })) {
    return std::nullopt;
  }
  return kernel;
}

/// The plan for `kernel` on CUDA device `ordinal`, built once and kept. Null when cuDNN has none.
std::shared_ptr<const codegen::ConvPlan> plan_for(const codegen::ConvKernel& kernel, int ordinal) {
  PlanKey key{kernel.input_shape,
              kernel.weight_shape,
              kernel.stride,
              kernel.padding,
              kernel.dilation,
              kernel.groups,
              kernel.bias.has_value(),
              ordinal};
  PlanCache& cache = plan_cache();
  const std::scoped_lock guard(cache.mutex);
  if (const auto found = cache.plans.find(key); found != cache.plans.end()) {
    return found->second;
  }
  Result<std::unique_ptr<codegen::ConvPlan>> made = codegen::make_conv_plan(kernel, ordinal);
  std::shared_ptr<const codegen::ConvPlan> plan;
  if (made.has_value()) {
    plan = std::move(made).value();
  } else {
    TESSIQ_LOG(kDebug, kLogModule) << "no cuDNN plan for a convolution: " << made.error().message();
  }
  cache.plans.emplace(std::move(key), plan);
  return plan;
}

/// The convolution every backend other than CPU and CUDA is routed to.
///
/// On a GPU it is one cuDNN call. Otherwise it runs PyTorch's CPU convolution on host copies and
/// copies the result back. The catch-all fallback cannot do that, since it would call this same
/// operation on the host, where it does not exist.
at::Tensor convolution_overrideable(const at::Tensor& input,
                                    const at::Tensor& weight,
                                    const std::optional<at::Tensor>& bias,
                                    at::IntArrayRef stride,
                                    at::IntArrayRef padding,
                                    at::IntArrayRef dilation,
                                    bool transposed,
                                    at::IntArrayRef output_padding,
                                    int64_t groups) {
  const std::optional<at::Tensor> given =
      bias.has_value() && bias->defined() ? bias : std::optional<at::Tensor>();
  const std::optional<int> ordinal = cuda_ordinal_of(input.device());
  const std::optional<codegen::ConvKernel> kernel =
      ordinal.has_value()
          ? describe(input, weight, given, stride, padding, dilation, transposed, groups)
          : std::nullopt;
  const std::shared_ptr<const codegen::ConvPlan> plan =
      kernel.has_value() ? plan_for(*kernel, *ordinal) : nullptr;
  if (kernel.has_value() && plan != nullptr) {
    const at::Tensor packed_weight = weight.contiguous();
    const std::optional<at::Tensor> packed_bias =
        given.has_value() ? std::optional<at::Tensor>(given->contiguous()) : std::nullopt;
    at::Tensor out = at::empty(kernel->output_shape, input.options());
    const Result<void> ran =
        plan->launch(input.const_data_ptr(),
                     packed_weight.const_data_ptr(),
                     packed_bias.has_value() ? packed_bias->const_data_ptr() : nullptr,
                     out.data_ptr(),
                     codegen::kDefaultStream);
    TORCH_CHECK(ran.has_value(), ran.has_value() ? "" : ran.error().to_string());
    return out;
  }

  count_cpu_fallback({"aten::convolution_overrideable", ""});
  const std::optional<at::Tensor> host_bias =
      given.has_value() ? std::optional<at::Tensor>(given->cpu()) : std::nullopt;
  const at::Tensor host = at::convolution(input.cpu(),
                                          weight.cpu(),
                                          host_bias,
                                          stride,
                                          padding,
                                          dilation,
                                          transposed,
                                          output_padding,
                                          groups);
  return host.to(input.device());
}

} // namespace

Result<void> register_eager_convolution(torch::Library& library) {
  try {
    library.impl("convolution_overrideable", &convolution_overrideable);
  } catch (const c10::Error& error) {
    return invalid_argument(error.what_without_backtrace());
  }
  return ok();
}

} // namespace tessiq::torch_backend
