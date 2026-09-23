// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>

#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// The most axes a pack kernel is written for.
///
/// One kernel per rank, so this is how many exist rather than a limit anything runs into. A
/// tensor deeper than this falls back, which costs a transfer and is correct.
inline constexpr std::size_t kMaxPackRank = 8;

/// The name the generated packing function carries, which is also the cubin cache's key.
[[nodiscard]] std::string pack_name(std::size_t element_bytes, std::size_t rank);

/// Source for the kernel that reads a strided tensor and writes a packed one.
///
/// This is what a `clone` is, and what `.contiguous()` on a view is. Without it both reach the
/// host fallback, which copies the whole tensor across, packs it there and copies it back, and
/// which measured 120 microseconds against 6 for the same 32768 elements on the card.
///
/// Two things make one kernel serve a lot of cases.
///
/// It copies bytes rather than numbers, so it is written for an element *size* rather than an
/// element type. A pack does no arithmetic, so what the bits mean does not come into it, and half
/// precision and booleans are covered without the emitter needing a spelling for them.
///
/// The extents and the steps arrive as arguments rather than being written in. So one kernel
/// serves every layout of every shape at a given rank, which means at most `kMaxPackRank` of them
/// exist per element size, and a shape that is not known until the call is no obstacle.
///
/// The destination is packed by definition, so the thread's own index walks it directly and only
/// the source needs the arithmetic.
[[nodiscard]] Result<std::string> pack_source(std::size_t element_bytes, std::size_t rank);

} // namespace tessiq::codegen
