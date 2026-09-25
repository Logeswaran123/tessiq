// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// RMSNorm: each row divided by its root mean square, then scaled by a weight.
//
//   y = x / sqrt(mean(x ** 2 over the last axis) + eps) * weight
//
// The kernel is rms_norm.cu, beside this file. It does the reduction over a row and the scaling in
// one launch, which no elementwise expression can say. Float32 only.

#include <ATen/ATen.h>
#include <cstdint>
#include <limits>

#include "tessiq/torch/custom_op.h"

namespace {

using tessiq::custom_ops::Launch;

/// Threads per block. A power of two, because the kernel's sum halves the block at every step.
constexpr std::uint32_t kThreads = 256;

/// What the kernel computes, in ATen.
at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight, double eps) {
  return x * (x.pow(2).mean(-1, /*keepdim=*/true) + eps).rsqrt() * weight;
}

/// One block per row, and one float of shared memory per thread for the row's partial sums.
Launch
launch(const at::Tensor& x, const at::Tensor& weight, double eps, const at::Tensor& /*out*/) {
  const std::int64_t columns = x.size(-1);
  const std::int64_t rows = x.numel() / columns;
  // The reference would broadcast any weight that fits, but the kernel reads one per column.
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == columns,
              "the rms_norm kernel reads one weight per column, and weight has shape ",
              weight.sizes(),
              " for ",
              columns,
              " columns");
  TORCH_CHECK(rows <= std::numeric_limits<std::int32_t>::max(),
              "the rms_norm kernel runs a block per row, and ",
              rows,
              " rows is more blocks than a grid holds");

  Launch launch;
  launch.grid = {static_cast<std::uint32_t>(rows), 1, 1};
  launch.block = {kThreads, 1, 1};
  launch.shared_bytes = kThreads * sizeof(float);
  launch.arguments = {columns, static_cast<float>(eps)};
  return launch;
}

} // namespace

TESSIQ_CUSTOM_OP(rms_norm, "(Tensor x, Tensor weight, float eps) -> Tensor")
    .reference(rms_norm)
    .launch(launch)
    .dtypes({at::kFloat});
