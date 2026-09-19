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

  /// What the same intermediates would cost with no reuse, which is every buffer alive at once.
  /// The ratio against `arena_bytes` is what the planner is for.
  std::size_t naive_bytes = 0;

  std::vector<Allocation> allocations;

  /// Human-readable, one line per allocation, for `tessiq-opt --memory-plan` and for tests that
  /// would rather compare text than walk a structure.
  [[nodiscard]] std::string to_string() const;
};

/// Assigns every intermediate an offset in a single arena, reusing space between values whose
/// live ranges do not overlap.
///
/// `alignment` is the caller's, because the IR does not know what hardware it is planning for. It
/// must be a power of two. Passing the device's own alignment is what lets a kernel assume an
/// aligned pointer instead of testing for one.
///
/// Every shape must be static. A dynamic dimension means the size is unknown, and a plan built on
/// a guess would hand a kernel a buffer smaller than what it writes.
Result<MemoryPlan> plan_memory(const Graph& graph, std::size_t alignment);

} // namespace tessiq::ir
