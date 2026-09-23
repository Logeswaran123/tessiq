// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <map>
#include <pybind11/pybind11.h>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/recipe.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/fused.h"

namespace tessiq::torch_backend {

/// The passes that decide what the backend takes, in the order they run.
///
/// Exposed so that a tool reporting on a compilation names the pipeline that actually ran rather
/// than a copy of it. A copy is a second source of truth, and the way it goes wrong is a report
/// that describes a compilation nobody performed.
[[nodiscard]] std::string_view default_pipeline();

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

/// A graph compiled as far as the compiler goes: a plan, and the kernels for it.
struct CompiledGraph {
  ir::Recipe recipe;

  /// One per fusion group the emitter could write code for. Empty on a build with no CUDA
  /// toolkit, and short of the number of groups when the emitter declined one.
  std::vector<FusedKernel> kernels;

  /// One per reduction the emitter could write code for. Kept apart from `kernels` because a
  /// reduction is one operation on its own rather than a group of fused ones, so it carries
  /// different extents and launches on a different grid.
  std::vector<ReductionKernel> reductions;

  /// One per matrix multiply cuBLAS can run. No compiled code behind these: what they carry is
  /// the shape of the library call.
  std::vector<GemmKernel> gemms;

  /// One per convolution cuDNN has a plan for.
  std::vector<ConvKernel> convs;
};

/// Imports an FX graph, partitions it, fuses it, and lowers the result.
///
/// `alignment` is the device's, passed in because the compiler does not know which device the
/// caller means to run on. `ordinal` is the CUDA device to compile kernels for, or negative to
/// skip compiling them, which is what a caller targeting the host reference wants.
Result<CompiledGraph> build_recipe_fx(const pybind11::object& graph_module,
                                      std::string name,
                                      std::size_t alignment,
                                      int ordinal);

} // namespace tessiq::torch_backend
