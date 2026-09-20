// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/support/result.h"

namespace tessiq::ir {

/// The cluster of a step the backend does not claim.
inline constexpr int kHost = -1;

/// One operation, in execution order.
struct Step {
  std::string op;

  /// Names of the values it produces. The first is the name the importer read from the FX node,
  /// which is how a caller holding the original graph finds the operation again.
  std::vector<std::string> results;

  /// Which cluster runs this step, or `kHost`.
  int cluster = kHost;
};

/// One buffer in the arena.
struct Buffer {
  std::string name;
  std::size_t offset = 0;
  std::size_t size = 0;
};

/// Everything needed to run a graph, worked out once.
///
/// A recipe holds names rather than pointers into the graph it came from, because it is built to
/// outlive that graph: it gets cached, and the next call that matches reuses it long after the
/// `Graph` object has gone. A pointer here would be a dangling read that happens to work until
/// the allocator reuses the memory.
///
/// It says nothing about what a step runs on, beyond which cluster claimed it. Turning a cluster
/// into a kernel is a separate job, and a recipe that already knew the answer would have to be
/// rebuilt when that changes.
struct Recipe {
  std::vector<Step> steps;

  /// Where each intermediate lives in the arena. Parameters and returned values are absent, since
  /// neither is the graph's to place.
  std::vector<Buffer> buffers;

  /// Bytes to allocate once, before the steps run.
  std::size_t arena_bytes = 0;

  /// What the same buffers would cost with no reuse.
  std::size_t naive_bytes = 0;

  /// One line per step and per buffer, for tests and for looking at what was produced.
  [[nodiscard]] std::string to_string() const;
};

/// Lowers a graph into a recipe.
///
/// Reads `tessiq.cluster` to decide which steps the backend claims. A graph that has not been
/// through `mark-supported,cluster` has no such attributes, so every step is `kHost`, which is a
/// correct recipe for a backend that claims nothing rather than an error.
///
/// `alignment` is the caller's, for the reason `plan_memory` takes one: the IR does not know what
/// hardware it is lowering for.
Result<Recipe> build_recipe(const Graph& graph, std::size_t alignment);

} // namespace tessiq::ir
