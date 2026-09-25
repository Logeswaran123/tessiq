// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>

#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// The most axes a pack kernel is written for. A deeper tensor is packed on the host.
inline constexpr std::size_t kMaxPackRank = 8;

/// The generated packing function's name.
[[nodiscard]] std::string pack_name(std::size_t element_bytes, std::size_t rank);

/// Source for the kernel that reads a strided tensor and writes a packed one, which is what `clone`
/// and `.contiguous()` on a view do. Without it both reach the host fallback.
///
/// It copies bytes, so there is one kernel per element size rather than per type, and half
/// precision and booleans need no spelling in the emitter. Extents and strides are arguments, so
/// one kernel serves every shape and layout of a given rank, including shapes not known until the
/// call. The destination is packed, so only the source index needs arithmetic.
[[nodiscard]] Result<std::string> pack_source(std::size_t element_bytes, std::size_t rank);

} // namespace tessiq::codegen
