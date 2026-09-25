// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/support/result.h"

namespace tessiq::ir {

/// Where one value sits inside the arena.
struct Allocation {
  const Value* value = nullptr;
  std::size_t offset = 0;
  /// Bytes reserved, which is the value's size rounded up to the requested alignment.
  std::size_t size = 0;
};

/// One buffer holding every intermediate a graph needs, and where each one lives in it.
///
/// Parameters and returned values are not in here. The caller owns its inputs, and a returned
/// value outlives the call, so neither can share space with something the graph overwrites.
struct MemoryPlan {
  /// Bytes to allocate once, before the graph runs.
  std::size_t arena_bytes = 0;

  /// What the same intermediates would cost with no reuse, every buffer alive at once.
  std::size_t naive_bytes = 0;

  std::vector<Allocation> allocations;

  /// Values left out of the arena because a dynamic dimension gives them no size until the graph
  /// runs.
  std::size_t unsized = 0;

  /// One line per allocation, for `tessiq-opt --memory-plan` and tests.
  [[nodiscard]] std::string to_string() const;
};

/// Assigns every intermediate an offset in a single arena, reusing space between values whose
/// live ranges do not overlap.
///
/// `alignment` is the device's and must be a power of two, so a kernel can assume an aligned
/// pointer. A value whose shape is not static is counted in `unsized` and left out rather than
/// sized by a guess; the rest of the graph is still planned.
Result<MemoryPlan> plan_memory(const Graph& graph, std::size_t alignment);

} // namespace tessiq::ir
