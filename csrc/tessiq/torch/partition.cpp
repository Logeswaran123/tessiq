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
#include <vector>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/recipe.h"
#include "tessiq/ir/text.h"
#include "tessiq/passes/pass.h"
#include "tessiq/support/logging.h"
#include "tessiq/torch/custom.h"
#include "tessiq/torch/fused.h"
#include "tessiq/torch/importer.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "partition";

/// The pipeline that decides what the backend takes.
///
/// Named here rather than passed in: what the backend can run is a property of the backend, not
/// something a caller should be choosing per graph.
// layout-assign sits after mark-supported because it only cares about what the backend claims,
// and before cluster because a materialisation it inserts is a node clustering has to see.
constexpr std::string_view kPipeline = "mark-supported,layout-assign,cluster,fuse-elementwise";

/// Imports and runs the pipeline. Shared by both entry points so they cannot disagree about what
/// the backend claims.
Result<std::unique_ptr<ir::Graph>> prepare(const pybind11::object& graph_module, std::string name) {
  TESSIQ_ASSIGN_OR_RETURN(auto graph, import_fx(graph_module, std::move(name)));

  passes::PassManager manager;
  TESSIQ_RETURN_IF_ERROR(manager.add_from_pipeline(default_pipeline()));
  TESSIQ_RETURN_IF_ERROR(manager.run(*graph));
  return graph;
}

} // namespace

std::string default_pipeline() {
  // A custom operation that replaces a torch one is swapped in first, so that deciding what the
  // backend claims, and everything after it, sees the custom operation.
  std::string pipeline;
  for (const std::string& pass : override_passes()) {
    pipeline += pass;
    pipeline += ",";
  }
  return pipeline + std::string(kPipeline);
}

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
    TESSIQ_ASSIGN_OR_RETURN(compiled.kernels, compile_kernels(*graph, ordinal));
    TESSIQ_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<Kernel>> custom,
                            custom_kernels(*graph, ordinal));
    compiled.kernels.insert(compiled.kernels.end(), custom.begin(), custom.end());
  }

  TESSIQ_LOG(kDebug, kLogModule) << "recipe with " << compiled.recipe.steps.size()
                                 << " step(s), arena " << compiled.recipe.arena_bytes << " bytes, "
                                 << compiled.kernels.size() << " kernel(s)";
  return compiled;
}

} // namespace tessiq::torch_backend
