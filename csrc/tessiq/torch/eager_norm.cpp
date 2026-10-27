// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Normalisation called outside a compiled graph.
//
// Neither has a kernel of its own. Each is written in terms of operations that already run on the
// device, laid out so that every full-size step reads its per-row or per-channel values in place.

#include "tessiq/torch/eager_norm.h"

#include <ATen/core/Tensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/native_batch_norm_ops.h>
#include <ATen/ops/native_layer_norm_ops.h>
#include <ATen/ops/rsqrt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "tessiq/support/result.h"
#include "tessiq/torch/backend.h"
#include "tessiq/torch/fallback.h"

namespace tessiq::torch_backend {

namespace {

using Normalised = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

/// Whether `input` is a packed, non-empty float tensor on a GPU, where every step below has a
/// kernel.
bool runs_on_gpu(const at::Tensor& input) {
  return cuda_ordinal_of(input.device()).has_value() &&
         (input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble) &&
         input.is_contiguous() && input.numel() > 0;
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

/// `x * scale + shift` per channel, from the running statistics, as the CPU kernel computes it.
///
/// Channels sit between the batch and the spatial axes, a broadcast no single read covers. So the
/// per-channel values are repeated over the batch first, a copy of `N * C` values, and the input
/// read as `[N * C, H * W]` takes each one per row.
std::optional<Normalised> batch_norm_on_gpu(const at::Tensor& input,
                                            const at::Tensor& weight,
                                            const at::Tensor& bias,
                                            const at::Tensor& running_mean,
                                            const at::Tensor& running_var,
                                            bool training,
                                            double eps) {
  if (training || !runs_on_gpu(input) || input.dim() < 2 || !running_mean.defined() ||
      !running_var.defined()) {
    return std::nullopt;
  }
  const std::int64_t batch = input.size(0);
  const std::int64_t channels = input.size(1);
  for (const at::Tensor* parameter : {&weight, &bias, &running_mean, &running_var}) {
    if (!fits(*parameter, input, channels)) {
      return std::nullopt;
    }
  }

  at::Tensor scale = at::rsqrt(running_var.reshape({channels}).add(eps));
  if (weight.defined()) {
    scale = scale.mul(weight.reshape({channels}));
  }
  at::Tensor shift = running_mean.reshape({channels}).mul(scale).neg();
  if (bias.defined()) {
    shift = shift.add(bias.reshape({channels}));
  }
  const auto per_row = [batch, channels](const at::Tensor& values) {
    return values.expand({batch, channels}).contiguous().view({batch * channels, 1});
  };

  at::Tensor out = input.view({batch * channels, -1}).mul(per_row(scale));
  out.add_(per_row(shift));
  // In eval mode torch saves no statistics, and returns empty tensors in their place.
  return Normalised{
      out.view(input.sizes()), at::empty({0}, input.options()), at::empty({0}, input.options())};
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

/// The input read as rows of the normalised extent: its mean, the centred values' mean square,
/// and the result scaled by the reciprocal of that root. torch's CPU kernel returns the same mean
/// and reciprocal deviation, shaped to broadcast against the input, for the backward pass.
std::optional<Normalised> layer_norm_on_gpu(const at::Tensor& input,
                                            c10::SymIntArrayRef normalized_shape,
                                            const at::Tensor& weight,
                                            const at::Tensor& bias,
                                            double eps) {
  const std::optional<c10::IntArrayRef> shape = c10::asIntArrayRefSlowOpt(normalized_shape);
  if (!runs_on_gpu(input) || !shape.has_value() || shape->empty() ||
      shape->size() > static_cast<std::size_t>(input.dim()) ||
      input.sizes().slice(static_cast<std::size_t>(input.dim()) - shape->size()) != *shape) {
    return std::nullopt;
  }
  const std::int64_t width = c10::multiply_integers(*shape);
  if (!fits(weight, input, width) || !fits(bias, input, width)) {
    return std::nullopt;
  }

  const at::Tensor rows = input.view({-1, width});
  const at::Tensor mean = rows.mean(-1, /*keepdim=*/true);
  at::Tensor out = rows.sub(mean);
  const at::Tensor rstd = at::rsqrt(out.mul(out).mean(-1, /*keepdim=*/true).add(eps));
  out.mul_(rstd);
  if (weight.defined()) {
    out.mul_(weight.reshape({width}));
  }
  if (bias.defined()) {
    out.add_(bias.reshape({width}));
  }

  std::vector<std::int64_t> statistics(
      input.sizes().begin(), input.sizes().end() - static_cast<std::ptrdiff_t>(shape->size()));
  statistics.resize(static_cast<std::size_t>(input.dim()), 1);
  return Normalised{out.view(input.sizes()), mean.view(statistics), rstd.view(statistics)};
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
