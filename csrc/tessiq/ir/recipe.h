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
/// Holds value names rather than pointers into the graph, because a cached recipe outlives the
/// graph it was built from. It records which cluster claimed each step; kernels are built
/// separately.
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
/// Reads `tessiq.cluster` to decide which steps the backend claims. Without it, as in a graph that
/// has not been through `mark-supported,cluster`, every step is `kHost`. `alignment` is the
/// device's, as for `plan_memory`.
Result<Recipe> build_recipe(const Graph& graph, std::size_t alignment);

} // namespace tessiq::ir
