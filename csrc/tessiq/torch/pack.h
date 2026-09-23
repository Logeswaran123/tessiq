// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>

#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// Copies `src` into `dst` on the device, gathering through `src`'s own layout.
///
/// `dst` has to be contiguous, the same shape, and the same element size. What `src` looks like is
/// the point: it may be a permuted or expanded view, and the kernel steps through it rather than
/// requiring it to be packed first.
///
/// Returns `unsupported` when there is no kernel for this case, which is the caller's signal to
/// fall back rather than a failure. That covers a device with no CUDA behind it, a build with no
/// toolkit, a rank deeper than the kernels go, and an element size no integer type has.
///
/// Asynchronous, like any launch. The caller orders the result against whatever reads it.
[[nodiscard]] Result<void> pack_on_device(const at::Tensor& src, const at::Tensor& dst);

} // namespace tessiq::torch_backend
