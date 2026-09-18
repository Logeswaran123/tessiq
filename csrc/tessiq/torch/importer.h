// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <pybind11/pybind11.h>
#include <string>

#include "tessiq/ir/graph.h"
#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// Converts a `torch.fx.GraphModule` into tessiq IR.
///
/// The graph is expected to have been through AOTAutograd, which functionalises it and decomposes
/// it to core ATen. That matters: the passes assume no operation mutates its operands, which is
/// true of a functionalised graph and not of a raw one.
///
/// Every node is imported, including operations tessiq has no kernel for. Deciding what to keep is
/// partitioning, and it runs later over the imported graph.
Result<std::unique_ptr<ir::Graph>> import_fx(const pybind11::object& graph_module,
                                             std::string name);

} // namespace tessiq::torch_backend
