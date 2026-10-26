// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <cstddef>
#include <optional>

#include "tessiq/runtime/allocator.h"

namespace tessiq::torch_backend {

/// The c10 allocator for tessiq device `index`, which always allocates there.
c10::Allocator* get_allocator(int index);

/// Registers the backend's hooks with torch. Idempotent, and called during module import.
void register_backend();

/// Gives a tessiq storage a new allocation of `bytes` on its own device, keeping the bytes the old
/// and new sizes share.
void reallocate_storage(const c10::Storage& storage, std::size_t bytes);

AllocatorStats allocator_stats(int index);

/// The CUDA ordinal behind tessiq device `device`, or nothing for the CPU reference device and any
/// device that is not tessiq's.
std::optional<int> cuda_ordinal_of(const c10::Device& device);

/// Releases cached blocks on `index`. Returns the bytes handed back to the device.
std::size_t empty_cache(int index);

} // namespace tessiq::torch_backend
