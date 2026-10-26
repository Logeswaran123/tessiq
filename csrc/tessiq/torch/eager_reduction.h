// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/library.h>

#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// Registers tessiq kernels for the reductions in ops.toml through `library`, which implements aten
/// operations for tessiq devices.
///
/// On a packed GPU tensor, a reduction over one contiguous range of axes is one launch of the
/// compiler's reduction kernel. Anything else goes through the CPU fallback.
Result<void> register_eager_reductions(torch::Library& library);

} // namespace tessiq::torch_backend
