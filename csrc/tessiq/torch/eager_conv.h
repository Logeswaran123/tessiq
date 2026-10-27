// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/library.h>

#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// Registers `convolution_overrideable`, where torch sends a convolution on a tessiq tensor,
/// through `library`.
///
/// A 2-D convolution of packed NCHW tensors on a GPU, of a type ops.toml lists, runs through
/// cuDNN. Anything else runs PyTorch's CPU convolution on host copies, which is counted as a
/// fallback.
Result<void> register_eager_convolution(torch::Library& library);

} // namespace tessiq::torch_backend
