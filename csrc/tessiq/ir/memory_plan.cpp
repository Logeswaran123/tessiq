// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/ir/memory_plan.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/liveness.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/macros.h"

namespace tessiq::ir {

namespace {

std::size_t round_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

/// A range that has been given an offset, kept so later ranges can avoid it.
struct Placed {
  std::size_t first = 0;
  std::size_t last = 0;
  std::size_t begin = 0;
  std::size_t end = 0;
};

/// Whether two live ranges need space at the same time.
///
/// Inclusive at both ends. A value last read by node 4 and a value defined by node 4 overlap,
/// because the operation reading the first is the one writing the second and it has not finished
/// reading when it starts writing.
bool overlaps(std::size_t a_first, std::size_t a_last, std::size_t b_first, std::size_t b_last) {
  return a_first <= b_last && b_first <= a_last;
}

Result<std::size_t> byte_size(const Value* value, std::size_t alignment) {
  const TensorType& type = value->type();
  if (!type.shape().is_static()) {
    return invalid_argument("cannot plan memory for %" + value->name() + " of type " +
                            type.to_string() +
                            ": a dynamic dimension has no size; run shape-infer first");
  }
  const auto elements = static_cast<std::size_t>(type.shape().element_count());
  return round_up(elements * dtype_size(type.dtype()), alignment);
}

/// The lowest offset at which `size` bytes fit without touching anything live at the same time.
std::size_t first_gap(const std::vector<Placed>& placed,
                      std::size_t first,
                      std::size_t last,
                      std::size_t size) {
  std::vector<std::pair<std::size_t, std::size_t>> blocked;
  for (const Placed& other : placed) {
    if (overlaps(first, last, other.first, other.last)) {
      blocked.emplace_back(other.begin, other.end);
    }
  }
  std::sort(blocked.begin(), blocked.end());

  std::size_t offset = 0;
  for (const auto& [begin, end] : blocked) {
    if (offset + size <= begin) {
      break;
    }
    offset = std::max(offset, end);
  }
  return offset;
}

} // namespace

std::string MemoryPlan::to_string() const {
  std::string text = "arena " + std::to_string(arena_bytes) + " bytes, naive " +
                     std::to_string(naive_bytes) + " bytes\n";
  for (const Allocation& allocation : allocations) {
    text += "  %" + allocation.value->name() + " @ " + std::to_string(allocation.offset) + " + " +
            std::to_string(allocation.size) + "\n";
  }
  return text;
}

Result<MemoryPlan> plan_memory(const Graph& graph, std::size_t alignment) {
  const bool power_of_two = alignment != 0 && (alignment & (alignment - 1)) == 0;
  TESSIQ_CHECK(power_of_two, "alignment must be a power of two, got " + std::to_string(alignment));

  TESSIQ_ASSIGN_OR_RETURN(const std::vector<LiveRange> ranges, compute_liveness(graph));

  MemoryPlan plan;
  std::vector<Placed> placed;

  // Definition order, which is the order compute_liveness returns node results in. Placing in a
  // fixed order is what makes the plan reproducible, and a reproducible plan is one a test can
  // compare against a file.
  for (const LiveRange& range : ranges) {
    if (range.is_parameter || range.escapes) {
      continue;
    }

    TESSIQ_ASSIGN_OR_RETURN(const std::size_t size, byte_size(range.value, alignment));
    const std::size_t offset = first_gap(placed, range.first, range.last, size);

    placed.push_back({range.first, range.last, offset, offset + size});
    plan.allocations.push_back({range.value, offset, size});
    plan.arena_bytes = std::max(plan.arena_bytes, offset + size);
    plan.naive_bytes += size;
  }

  return plan;
}

} // namespace tessiq::ir
