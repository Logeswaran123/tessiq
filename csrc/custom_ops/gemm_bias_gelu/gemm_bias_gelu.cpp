// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// A linear layer followed by GELU, as one GEMM:
//
//   y = gelu(x @ weight.T + bias)
//
// with weight laid out as nn.Linear keeps it, one row per output feature, and the exact GELU, the
// one written with erf. Its host code is gemm_bias_gelu.nvcc.cu, a CUTLASS GEMM whose epilogue adds
// the bias and applies GELU to each element before it is written, which saves writing the product
// out and reading it back. Float32 only, with float32 arithmetic throughout.

#include <ATen/ATen.h>

#include "tessiq/torch/custom_op.h"

namespace {

/// What the host code computes, in ATen.
at::Tensor gemm_bias_gelu(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias) {
  return at::gelu(at::addmm(bias, x, weight.t()));
}

} // namespace

TESSIQ_CUSTOM_OP(gemm_bias_gelu, "(Tensor x, Tensor weight, Tensor bias) -> Tensor")
    .reference(gemm_bias_gelu)
    .precompiled()
    .dtypes({at::kFloat})
    .pattern();
