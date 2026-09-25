// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/recipe.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/kernel.h"

namespace tessiq::torch_backend {

/// The passes that decide what the backend takes, in the order they run.
///
/// Exposed so that a tool reporting on a compilation names the pipeline that actually ran rather
/// than a copy of it. A copy is a second source of truth, and the way it goes wrong is a report
/// that describes a compilation nobody performed.
[[nodiscard]] std::string default_pipeline();

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
///
/// `rewrites` names passes to run ahead of the default pipeline, which is how a compile asks for a
/// custom operation's rewrite.
Result<PartitionResult> partition_fx(const pybind11::object& graph_module,
                                     std::string name,
                                     const std::vector<std::string>& rewrites = {});

/// A graph compiled as far as the compiler goes: a plan, and the kernels for it.
struct CompiledGraph {
  ir::Recipe recipe;

  /// Every kernel the graph can run on the device, of whatever kind, the calls to custom
  /// operations with kernels of their own included. Empty on a build with no CUDA toolkit, and
  /// short of what the graph could use wherever the compiler declined something, which then runs
  /// one operation at a time.
  std::vector<std::shared_ptr<Kernel>> kernels;
};

/// Imports an FX graph, partitions it, fuses it, and lowers the result.
///
/// `alignment` is the device's, passed in because the compiler does not know which device the
/// caller means to run on. `ordinal` is the CUDA device to compile kernels for, or negative to
/// skip compiling them, which is what a caller targeting the host reference wants.
/// `rewrites` names passes to run first, as for `partition_fx`.
Result<CompiledGraph> build_recipe_fx(const pybind11::object& graph_module,
                                      std::string name,
                                      std::size_t alignment,
                                      int ordinal,
                                      const std::vector<std::string>& rewrites = {});

} // namespace tessiq::torch_backend
