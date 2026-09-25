// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Layer normalisation over the last axis:
//
//   y = (x - mean) / sqrt(var + eps) * weight + bias, with mean and var over the last axis
//
// Its host code is layer_norm.nvcc.cu, which nvcc compiles when tessiq is built. It launches two
// kernels, one for each row's statistics and one to apply them, and keeps the statistics in
// workspace between the two. Float32 only.

#include <ATen/ATen.h>

#include "tessiq/torch/custom_op.h"

namespace {

/// What the host code computes, in ATen.
at::Tensor
layer_norm(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias, double eps) {
  const at::Tensor centred = x - x.mean(-1, /*keepdim=*/true);
  const at::Tensor variance = (centred * centred).mean(-1, /*keepdim=*/true);
  return centred * (variance + eps).rsqrt() * weight + bias;
}

} // namespace

TESSIQ_CUSTOM_OP(layer_norm, "(Tensor x, Tensor weight, Tensor bias, float eps) -> Tensor")
    .reference(layer_norm)
    .precompiled()
    .dtypes({at::kFloat});
