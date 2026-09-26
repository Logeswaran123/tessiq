// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// A linear layer followed by the tanh approximation of GELU, as one cuBLASLt call:
//
//   y = gelu(x @ weight.T + bias, approximate="tanh")
//
// with weight laid out as nn.Linear keeps it. Its host code is gemm_bias_gelu_tanh.nvcc.cu, which
// has no kernel of its own: it calls cuBLASLt, whose GELU_BIAS epilogue adds the bias and applies
// the tanh GELU before writing. It is gemm_bias_gelu's counterpart for the approximation, which
// cuBLASLt computes and gemm_bias_gelu does not. Float32 only, with float32 arithmetic throughout.

#include <ATen/ATen.h>

#include "tessiq/torch/custom_op.h"

namespace {

/// What the host code computes, in ATen.
at::Tensor
gemm_bias_gelu_tanh(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias) {
  return at::gelu(at::addmm(bias, x, weight.t()), "tanh");
}

} // namespace

TESSIQ_CUSTOM_OP(gemm_bias_gelu_tanh, "(Tensor x, Tensor weight, Tensor bias) -> Tensor")
    .reference(gemm_bias_gelu_tanh)
    .precompiled()
    .dtypes({at::kFloat})
    .pattern();
