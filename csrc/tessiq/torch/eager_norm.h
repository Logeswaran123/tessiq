// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/library.h>

#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// Registers tessiq kernels for eval-mode batch norm and for layer norm through `library`, which
/// implements aten operations for tessiq devices.
///
/// On a packed float GPU tensor each is a few launches of the eager elementwise and reduction
/// kernels. Training-mode batch norm, and anything else, goes through the CPU fallback.
Result<void> register_eager_normalization(torch::Library& library);

} // namespace tessiq::torch_backend
