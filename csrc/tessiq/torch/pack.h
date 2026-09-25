// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>

#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// Copies `src` into `dst` on the device, gathering through `src`'s own layout.
///
/// `dst` has to be contiguous, with the same shape and element size. `src` may be any view, such
/// as a permuted or expanded one.
///
/// Returns `unsupported`, the caller's signal to fall back, for the CPU reference device, a build
/// with no toolkit, a rank above `kMaxPackRank`, or an element size no integer type has.
///
/// Asynchronous on the legacy default stream.
[[nodiscard]] Result<void> pack_on_device(const at::Tensor& src, const at::Tensor& dst);

} // namespace tessiq::torch_backend
