// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <map>
#include <pybind11/pybind11.h>
#include <string>

#include "tessiq/ir/recipe.h"
#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// The outcome of compiling one FX graph, as far as the compiler currently goes.
struct PartitionResult {
  /// The partitioned graph in textual IR, for tests and for looking at what happened.
  std::string ir;

  /// Which cluster each node landed in, keyed by the FX node name. A node absent from this map
  /// has no kernel and runs on the host.
  std::map<std::string, int> clusters;
};

/// Imports an FX graph and partitions it.
///
/// Node names survive the round trip: the importer names each IR value after the FX node that
/// produced it, so the caller can map a cluster back onto the graph it started with. That is what
/// lets the executor decide, node by node, what runs where.
Result<PartitionResult> partition_fx(const pybind11::object& graph_module, std::string name);

/// Imports an FX graph, partitions it, and lowers the result into a recipe.
///
/// `alignment` is the device's, passed in because the compiler does not know which device the
/// caller means to run on.
Result<ir::Recipe>
build_recipe_fx(const pybind11::object& graph_module, std::string name, std::size_t alignment);

} // namespace tessiq::torch_backend
