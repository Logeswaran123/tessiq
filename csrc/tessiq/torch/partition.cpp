// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/torch/partition.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/recipe.h"
#include "tessiq/ir/text.h"
#include "tessiq/passes/pass.h"
#include "tessiq/support/logging.h"
#include "tessiq/torch/fused.h"
#include "tessiq/torch/importer.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "partition";

/// The pipeline that decides what the backend takes.
///
/// Named here rather than passed in: what the backend can run is a property of the backend, not
/// something a caller should be choosing per graph.
constexpr std::string_view kPipeline = "mark-supported,cluster,fuse-elementwise";

/// Imports and runs the pipeline. Shared by both entry points so they cannot disagree about what
/// the backend claims.
Result<std::unique_ptr<ir::Graph>> prepare(const pybind11::object& graph_module, std::string name) {
  TESSIQ_ASSIGN_OR_RETURN(auto graph, import_fx(graph_module, std::move(name)));

  passes::PassManager manager;
  TESSIQ_RETURN_IF_ERROR(manager.add_from_pipeline(kPipeline));
  TESSIQ_RETURN_IF_ERROR(manager.run(*graph));
  return graph;
}

} // namespace

Result<PartitionResult> partition_fx(const pybind11::object& graph_module, std::string name) {
  TESSIQ_ASSIGN_OR_RETURN(auto graph, prepare(graph_module, std::move(name)));

  PartitionResult result;
  result.ir = ir::print(*graph);

  for (const ir::Node* node : graph->nodes()) {
    const ir::Attribute* attribute = node->attribute("tessiq.cluster");
    if (attribute == nullptr) {
      continue;
    }
    const auto* cluster = std::get_if<std::int64_t>(attribute);
    if (cluster == nullptr) {
      return internal_error("tessiq.cluster is not an integer on '" + node->op() + "'");
    }
    // One result per node today. When an op produces several, the name of the first still
    // identifies the node, which is what the caller keys on.
    if (node->result_count() > 0) {
      result.clusters.emplace(node->result(0)->name(), static_cast<int>(*cluster));
    }
  }

  TESSIQ_LOG(kDebug, kLogModule) << result.clusters.size() << " of " << graph->node_count()
                                 << " node(s) clustered";
  return result;
}

Result<CompiledGraph> build_recipe_fx(const pybind11::object& graph_module,
                                      std::string name,
                                      std::size_t alignment,
                                      int ordinal) {
  TESSIQ_ASSIGN_OR_RETURN(auto graph, prepare(graph_module, std::move(name)));

  CompiledGraph compiled;
  TESSIQ_ASSIGN_OR_RETURN(compiled.recipe, ir::build_recipe(*graph, alignment));
  if (ordinal >= 0) {
    TESSIQ_ASSIGN_OR_RETURN(compiled.kernels, compile_fused(*graph, ordinal));
    TESSIQ_ASSIGN_OR_RETURN(compiled.reductions, compile_reductions(*graph, ordinal));
  }

  TESSIQ_LOG(kDebug, kLogModule) << "recipe with " << compiled.recipe.steps.size()
                                 << " step(s), arena " << compiled.recipe.arena_bytes << " bytes, "
                                 << compiled.kernels.size() << " kernel(s), "
                                 << compiled.reductions.size() << " reduction(s)";
  return compiled;
}

} // namespace tessiq::torch_backend
