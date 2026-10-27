// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/library.h>

#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// Registers tessiq kernels for the matmuls in ops.toml through `library`, which implements aten
/// operations for tessiq devices.
///
/// On GPU tensors of a type the table lists, each call is one cuBLAS multiply, with a transposed
/// operand read through a transpose flag. Anything else goes through the CPU fallback.
Result<void> register_eager_matmuls(torch::Library& library);

} // namespace tessiq::torch_backend
