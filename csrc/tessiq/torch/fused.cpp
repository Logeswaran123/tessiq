// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/torch/fused.h"

#include <ATen/core/Tensor.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tessiq/codegen/cubin_cache.h"
#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/kernel.h"
#include "tessiq/codegen/program.h"
#include "tessiq/codegen/reduction.h"
#include "tessiq/support/logging.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "fused";

} // namespace

Result<void> FusedKernel::launch(const std::vector<at::Tensor>& tensors,
                                 const at::Tensor& out) const {
  if (program == nullptr) {
    return internal_error("fusion group " + std::to_string(group) + " has no compiled kernel");
  }
  if (tensors.size() != inputs.size()) {
    return invalid_argument("fusion group " + std::to_string(group) + " takes " +
                            std::to_string(inputs.size()) + " inputs, got " +
                            std::to_string(tensors.size()));
  }

  std::vector<void*> buffers;
  buffers.reserve(tensors.size() + 1);
  for (const at::Tensor& tensor : tensors) {
    // The kernel walks one index across every buffer, which is only the same elements if the
    // layouts agree. A non-contiguous input would be read in the wrong order silently.
    if (!tensor.is_contiguous()) {
      return invalid_argument("fusion group " + std::to_string(group) +
                              " was given a non-contiguous input");
    }
    if (tensor.numel() != elements) {
      return invalid_argument("fusion group " + std::to_string(group) + " expects " +
                              std::to_string(elements) + " elements, got " +
                              std::to_string(tensor.numel()));
    }
    buffers.push_back(tensor.data_ptr());
  }
  if (!out.is_contiguous() || out.numel() != elements) {
    return invalid_argument("fusion group " + std::to_string(group) +
                            " was given an output that is not " + std::to_string(elements) +
                            " contiguous elements");
  }
  buffers.push_back(out.data_ptr());

  return program->launch(buffers, elements);
}

Result<std::vector<FusedKernel>> compile_fused(const ir::Graph& graph, int ordinal) {
  std::vector<FusedKernel> compiled;
  if (!codegen::launcher_available()) {
    return compiled;
  }

  const auto groups = codegen::fusion_groups(graph);
  if (groups.empty()) {
    return compiled;
  }

  TESSIQ_ASSIGN_OR_RETURN(const auto capability, codegen::device_capability(ordinal));

  for (const std::int64_t group : groups) {
    TESSIQ_ASSIGN_OR_RETURN(const codegen::FusedKernel description,
                            codegen::extract_kernel(graph, group));

    const codegen::Emitter& emitter = codegen::cuda_emitter();
    if (!emitter.supports(description)) {
      // Not a failure. The operations run one at a time, which always works, and a group the
      // emitter declines is the reason `supports` exists.
      TESSIQ_LOG(kDebug, kLogModule) << "no cuda code for group " << group << ", leaving it alone";
      continue;
    }

    TESSIQ_ASSIGN_OR_RETURN(const std::string source, emitter.emit(description));
    TESSIQ_ASSIGN_OR_RETURN(
        const codegen::Cubin cubin,
        codegen::compile_cached(source, description.name, capability.first, capability.second));
    TESSIQ_ASSIGN_OR_RETURN(const codegen::Program* program,
                            codegen::load_program(cubin, description.name, ordinal));

    std::vector<std::string> members;
    members.reserve(description.body.size());
    for (const codegen::Step& step : description.body) {
      members.push_back(step.result);
    }

    compiled.push_back({.group = group,
                        .inputs = description.inputs,
                        .output = description.output,
                        .members = std::move(members),
                        .elements = description.elements,
                        .program = program});
  }

  TESSIQ_LOG(kDebug, kLogModule) << "compiled " << compiled.size() << " of " << groups.size()
                                 << " group(s)";
  return compiled;
}

Result<void> ReductionKernel::launch(const at::Tensor& in, const at::Tensor& out) const {
  if (program == nullptr) {
    return internal_error("reduction " + output + " has no compiled kernel");
  }
  // The kernel indexes both buffers arithmetically from their first element, so a layout other
  // than contiguous would have it read and write the wrong positions without failing.
  if (!in.is_contiguous() || in.numel() != inputs()) {
    return invalid_argument("reduction " + output + " expects " + std::to_string(inputs()) +
                            " contiguous input elements, got " + std::to_string(in.numel()));
  }
  if (!out.is_contiguous() || out.numel() != outputs()) {
    return invalid_argument("reduction " + output + " expects " + std::to_string(outputs()) +
                            " contiguous output elements, got " + std::to_string(out.numel()));
  }

  const std::vector<void*> buffers{in.data_ptr(), out.data_ptr()};
  return program->launch(buffers, {outer, reduce, inner}, blocks);
}

/// Compiles one reduction, or nothing when it has no kernel shape or no code the emitter writes.
///
/// Its own function because `compile_reductions` is otherwise a loop with four ways to decline
/// and three ways to fail, which is more branching than one function should carry.
Result<std::optional<ReductionKernel>>
compile_one(const ir::Node& node, std::pair<int, int> capability, int ordinal) {
  TESSIQ_ASSIGN_OR_RETURN(const auto description, codegen::extract_reduction(node));
  if (!description.has_value()) {
    TESSIQ_LOG(kDebug, kLogModule) << "no kernel shape for " << node.op() << ", leaving it alone";
    return std::optional<ReductionKernel>{};
  }

  const codegen::Emitter& emitter = codegen::cuda_emitter();
  if (!emitter.supports(*description)) {
    TESSIQ_LOG(kDebug, kLogModule) << "no cuda code for " << description->name;
    return std::optional<ReductionKernel>{};
  }

  TESSIQ_ASSIGN_OR_RETURN(const std::string source, emitter.emit(*description));
  TESSIQ_ASSIGN_OR_RETURN(
      const codegen::Cubin cubin,
      codegen::compile_cached(source, description->name, capability.first, capability.second));
  TESSIQ_ASSIGN_OR_RETURN(const codegen::Program* program,
                          codegen::load_program(cubin, description->name, ordinal));

  return std::optional<ReductionKernel>{ReductionKernel{.output = description->output,
                                                        .input = description->input,
                                                        .outer = description->outer,
                                                        .reduce = description->reduce,
                                                        .inner = description->inner,
                                                        .blocks = description->blocks(),
                                                        .program = program}};
}

Result<std::vector<ReductionKernel>> compile_reductions(const ir::Graph& graph, int ordinal) {
  std::vector<ReductionKernel> compiled;
  if (!codegen::launcher_available()) {
    return compiled;
  }

  const std::vector<const ir::Node*> candidates = codegen::reduction_nodes(graph);
  if (candidates.empty()) {
    return compiled;
  }

  TESSIQ_ASSIGN_OR_RETURN(const auto capability, codegen::device_capability(ordinal));

  for (const ir::Node* node : candidates) {
    TESSIQ_ASSIGN_OR_RETURN(auto kernel, compile_one(*node, capability, ordinal));
    if (kernel.has_value()) {
      compiled.push_back(std::move(*kernel));
    }
  }

  TESSIQ_LOG(kDebug, kLogModule) << "compiled " << compiled.size() << " of " << candidates.size()
                                 << " reduction(s)";
  return compiled;
}

} // namespace tessiq::torch_backend
