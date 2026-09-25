// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The sigmoid approximation of GELU: x * sigmoid(1.702 * x).
//
// Elementwise, so it has no kernel file. The expression is one element of the result, and the
// compiler writes it into the loop of whatever it fuses the operation with. That is why it stays an
// expression rather than becoming a kernel: a kernel of its own would give up the fusion.

#include <ATen/ATen.h>

#include "tessiq/torch/custom_op.h"

namespace {

/// What the expression computes, in ATen.
at::Tensor fast_gelu(const at::Tensor& x) { return x * (1.702 * x).sigmoid(); }

} // namespace

TESSIQ_CUSTOM_OP(fast_gelu, "(Tensor x) -> Tensor")
    .reference(fast_gelu)
    .expression("$0 * (scalar_t(1) / (scalar_t(1) + exp(scalar_t(-1.702) * $0)))")
    .dtypes({at::kFloat, at::kDouble});
