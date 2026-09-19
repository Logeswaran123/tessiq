// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/support/result.h"

namespace tessiq::ir {

/// When a value has to exist, as a range of node positions.
///
/// Positions index the graph's node list, so `first` and `last` are comparable across values.
/// The range is closed at both ends: a value defined at 2 and last read at 5 occupies 2 through 5
/// inclusive, and a value whose last reader is the node that defines its successor still overlaps
/// that successor. Treating the boundary as shared would let a buffer be overwritten while the
/// operation reading it is still running.
struct LiveRange {
  const Value* value = nullptr;

  /// Position of the node defining the value. Zero for a graph parameter, which exists before any
  /// node runs.
  std::size_t first = 0;

  /// Position of the last node reading the value. Equal to `first` when nothing reads it, and
  /// `node_count()` when the graph returns it, since a returned value outlives every node.
  std::size_t last = 0;

  /// The caller supplies this value, so nothing here may place it.
  bool is_parameter = false;

  /// The graph returns this value, so it outlives any buffer this graph owns.
  bool escapes = false;
};

/// Live ranges for every value in the graph, parameters first and then node results in order.
///
/// Precondition: the graph verifies, so every operand is defined before it is used and every use
/// list agrees with the operands that exist. Liveness reads use lists directly and would silently
/// shorten a range if one were stale.
Result<std::vector<LiveRange>> compute_liveness(const Graph& graph);

} // namespace tessiq::ir
