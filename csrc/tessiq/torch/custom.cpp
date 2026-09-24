// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/torch/custom.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tessiq/ir/text.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/passes/pass.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/fused.h"
#include "tessiq/torch/kernel.h"

namespace tessiq::torch_backend {

namespace {

/// The passes that decide whether a node is claimed and give it a fusion group of its own.
///
/// The front of the compiler's own pipeline. Layout assignment is left out because every operand
/// here is packed, and nothing else can join the group because there is nothing else.
constexpr std::string_view kPipeline = "mark-supported,cluster,fuse-elementwise";

/// `graph @eager(%x0: f32[?], %x1: f32[?]) -> f32[?] { %y = op(%x0, %x1) : f32[?] ... }`.
std::string one_node_graph(std::string_view op, int arity, ir::DType dtype) {
  const std::string type = std::string(ir::dtype_name(dtype)) + "[?]";
  std::string parameters;
  std::string arguments;
  for (int i = 0; i < arity; ++i) {
    const std::string name = "%x" + std::to_string(i);
    if (i > 0) {
      parameters += ", ";
      arguments += ", ";
    }
    parameters += name;
    parameters += ": ";
    parameters += type;
    arguments += name;
  }
  std::string text = "graph @eager(";
  text += parameters;
  text += ") -> ";
  text += type;
  text += " {\n  %y = ";
  text += op;
  text += "(";
  text += arguments;
  text += ") : ";
  text += type;
  text += "\n  return %y\n}\n";
  return text;
}

} // namespace

Result<std::shared_ptr<Kernel>>
compile_elementwise(std::string_view op, int arity, ir::DType dtype, int ordinal) {
  const passes::OpInfo* info = passes::find_op(op);
  if (info == nullptr || info->category != passes::OpCategory::kElementwise) {
    return invalid_argument(std::string(op) +
                            " is not an elementwise operation the compiler knows");
  }
  if (arity != info->arity) {
    return invalid_argument(std::string(op) + " takes " + std::to_string(info->arity) +
                            " operands, not " + std::to_string(arity));
  }

  TESSIQ_ASSIGN_OR_RETURN(auto graph, ir::parse(one_node_graph(op, arity, dtype)));
  passes::PassManager manager;
  TESSIQ_RETURN_IF_ERROR(manager.add_from_pipeline(kPipeline));
  TESSIQ_RETURN_IF_ERROR(manager.run(*graph));

  TESSIQ_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<Kernel>> kernels,
                          compile_kernels(*graph, ordinal));
  if (kernels.size() != 1) {
    return unsupported("no kernel for " + std::string(op) + " on " +
                       std::string(ir::dtype_name(dtype)) + " for device " +
                       std::to_string(ordinal));
  }
  return kernels.front();
}

} // namespace tessiq::torch_backend
