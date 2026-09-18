// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "tessiq/runtime/device.h"

namespace tessiq {

/// Constructs a Device for every CUDA device the driver reports.
///
/// Returns an empty vector when this build has no CUDA support, or when the machine has no usable
/// device. Neither is an error: the CPU reference is always there, so a GPU-less machine gets a
/// working default rather than a failed import.
std::vector<std::unique_ptr<Device>> discover_cuda_devices();

} // namespace tessiq
