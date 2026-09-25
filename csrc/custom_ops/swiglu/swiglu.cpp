// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The gated activation of a SwiGLU feed-forward block: silu(gate) * up.
//
// Elementwise, like fast_gelu, so the compiler fuses it with the operations around it. Its two
// operands broadcast against each other the way torch's own binary operations do.

#include <ATen/ATen.h>

#include "tessiq/torch/custom_op.h"

namespace {

/// What the expression computes, in ATen.
at::Tensor swiglu(const at::Tensor& gate, const at::Tensor& up) { return at::silu(gate) * up; }

} // namespace

TESSIQ_CUSTOM_OP(swiglu, "(Tensor gate, Tensor up) -> Tensor")
    .reference(swiglu)
    .expression("$0 / (scalar_t(1) + exp(-$0)) * $1")
    .dtypes({at::kFloat, at::kDouble});
