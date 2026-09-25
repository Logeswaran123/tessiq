// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/ScalarType.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>

#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"

namespace tessiq::torch_backend {

/// The IR's name for torch element type `dtype`, or an unsupported error naming a type it has none
/// for. Such a type is not a failure: a value of it stays on the host.
Result<ir::DType> to_ir_dtype(at::ScalarType dtype);

/// Converts a `torch.fx.GraphModule` into tessiq IR.
///
/// The graph is expected to have been through AOTAutograd, which functionalises it and decomposes
/// it to core ATen. The passes assume no operation mutates its operands.
///
/// Every node is imported, including operations tessiq has no kernel for; partitioning decides
/// later what runs where.
Result<std::unique_ptr<ir::Graph>> import_fx(const pybind11::object& graph_module,
                                             std::string name);

} // namespace tessiq::torch_backend
