// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// Registers a tessiq kernel for every ATen spelling of every elementwise operation in ops.toml.
///
/// A call on GPU tensors runs one generated kernel there. Anything the kernel does not cover goes
/// through the CPU fallback, as before. Called once during module import, after the custom
/// operations, so an operation one of them replaces is left to it.
Result<void> register_eager_ops();

} // namespace tessiq::torch_backend
