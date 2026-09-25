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
/// The overrides for registered custom operations, then the fixed pipeline. Exposed so that tools
/// report the pipeline that actually ran.
[[nodiscard]] std::string default_pipeline();

/// The outcome of compiling one FX graph, as far as the compiler currently goes.
struct PartitionResult {
  /// The partitioned graph in textual IR, for tests and for looking at what happened.
  std::string ir;

  /// Which cluster each node landed in, keyed by the name of its first result: the FX node's name,
  /// or the getitem's when one reads that result. A node absent from this map runs on the host.
  std::map<std::string, int> clusters;
};

/// Imports an FX graph and partitions it.
///
/// IR values are named after the FX nodes that produced them, so clusters map back onto the FX
/// graph. `rewrites` names rewrite passes to run before the default pipeline.
Result<PartitionResult> partition_fx(const pybind11::object& graph_module,
                                     std::string name,
                                     const std::vector<std::string>& rewrites = {});

/// A graph compiled as far as the compiler goes: a plan, and the kernels for it.
struct CompiledGraph {
  ir::Recipe recipe;

  /// Every kernel the graph runs on the device, custom operations included. Empty on a build
  /// with no CUDA toolkit. A claimed node with no kernel runs one operation at a time.
  std::vector<std::shared_ptr<Kernel>> kernels;
};

/// Imports an FX graph, partitions it, fuses it, and lowers the result.
///
/// `alignment` is the target device's. `ordinal` is the CUDA device to compile kernels for, or
/// negative to skip compiling, as for the host reference. `rewrites` is as for `partition_fx`.
Result<CompiledGraph> build_recipe_fx(const pybind11::object& graph_module,
                                      std::string name,
                                      std::size_t alignment,
                                      int ordinal,
                                      const std::vector<std::string>& rewrites = {});

} // namespace tessiq::torch_backend
