// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "tessiq/runtime/device.h"
#include "tessiq/support/result.h"

namespace tessiq {

/// Constructs a Device for every CUDA device the driver reports.
///
/// Empty when the build has no CUDA support or the machine has no usable device. Neither is an
/// error; the CPU reference is always there.
std::vector<std::unique_ptr<Device>> discover_cuda_devices();

/// Allocates page-locked host memory, which a GPU copies to and from without staging it.
///
/// Portable, so it is page-locked for every device, not only the current one. Fails with
/// kUnsupported in a build without CUDA.
Result<void*> allocate_page_locked(std::size_t bytes);

/// Frees memory from allocate_page_locked. The caller makes sure no copy still reads it.
void free_page_locked(void* ptr) noexcept;

/// Whether `ptr` is in page-locked host memory, whoever page-locked it. False without CUDA.
[[nodiscard]] bool is_page_locked(const void* ptr) noexcept;

} // namespace tessiq
