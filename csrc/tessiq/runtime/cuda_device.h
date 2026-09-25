// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "tessiq/runtime/device.h"

namespace tessiq {

/// Constructs a Device for every CUDA device the driver reports.
///
/// Empty when the build has no CUDA support or the machine has no usable device. Neither is an
/// error; the CPU reference is always there.
std::vector<std::unique_ptr<Device>> discover_cuda_devices();

} // namespace tessiq
