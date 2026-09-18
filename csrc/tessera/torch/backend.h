// Copyright 2026 The Tessera Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/Allocator.h>
#include <cstddef>

#include "tessera/runtime/allocator.h"

namespace tessera::torch_backend {

/// The c10 allocator backing tessera tensors.
c10::Allocator* get_allocator();

/// Installs the allocator with c10. Idempotent, and called during module import.
void register_allocator();

AllocatorStats allocator_stats();

/// Releases cached blocks. Returns the bytes handed back to the device.
std::size_t empty_cache();

} // namespace tessera::torch_backend
