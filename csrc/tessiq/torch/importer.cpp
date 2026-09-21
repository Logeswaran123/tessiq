// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/torch/importer.h"

#include <ATen/core/Tensor.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <torch/csrc/autograd/python_variable.h>

#include "tessiq/ir/types.h"
#include "tessiq/support/logging.h"

namespace tessiq::torch_backend {

namespace {

namespace py = pybind11;

constexpr std::string_view kLogModule = "importer";

Result<ir::DType> to_ir_dtype(at::ScalarType dtype) {
  switch (dtype) {
  case at::kHalf:
    return ir::DType::kF16;
  case at::kBFloat16:
    return ir::DType::kBF16;
  case at::kFloat:
    return ir::DType::kF32;
  case at::kDouble:
    return ir::DType::kF64;
  case at::kBool:
    return ir::DType::kI1;
  case at::kInt:
    return ir::DType::kI32;
  case at::kLong:
    return ir::DType::kI64;
  default:
    // Not a failure of the importer. The graph uses a dtype the IR cannot describe, so it has to
    // stay on the host, and saying which one is what makes that actionable.
    return unsupported(std::string("tessiq has no IR dtype for ") + toString(dtype));
  }
}

/// The type of an FX node, taken from the fake tensor AOTAutograd left in `meta["val"]`.
///
/// That is the only place shape and dtype are recorded. A node without it has not been through
/// shape propagation, and guessing would put a wrong type on a value the passes then trust.
Result<ir::TensorType> type_of(const py::object& node) {
  const py::object meta = node.attr("meta");
  if (!meta.contains("val")) {
    return invalid_argument("fx node '" + node.attr("name").cast<std::string>() +
                            "' has no meta['val']; the graph has not been through shape "
                            "propagation, so its types are unknown");
  }
  const py::object value = meta[py::str("val")];
  if (!THPVariable_Check(value.ptr())) {
    return unsupported("fx node '" + node.attr("name").cast<std::string>() +
                       "' produces something other than a tensor, which the IR cannot hold yet");
  }
  const auto tensor = value.cast<at::Tensor>();
  TESSIQ_ASSIGN_OR_RETURN(const ir::DType dtype, to_ir_dtype(tensor.scalar_type()));

  std::vector<std::int64_t> dims;
  dims.reserve(static_cast<std::size_t>(tensor.dim()));
  for (const auto dim : tensor.sizes()) {
    dims.push_back(dim);
  }
  return ir::TensorType(dtype, ir::Shape(std::move(dims)));
}

bool is_fx_node(const py::object& value) {
  static const py::object node_type = py::module_::import("torch.fx").attr("Node");
  return py::isinstance(value, node_type);
}

/// A Python scalar as an IR attribute. Anything else is refused rather than guessed at.
Result<ir::Attribute> to_attribute(const py::handle& value) {
  // None is an argument, not a missing one. A convolution with no bias passes None where the bias
  // goes, and dropping it would shift the position of every argument after it.
  if (value.is_none()) {
    return ir::Attribute{std::monostate{}};
  }
  if (py::isinstance<py::bool_>(value)) {
    return ir::Attribute{value.cast<bool>()};
  }
  if (py::isinstance<py::int_>(value)) {
    return ir::Attribute{value.cast<std::int64_t>()};
  }
  if (py::isinstance<py::float_>(value)) {
    return ir::Attribute{value.cast<double>()};
  }
  if (py::isinstance<py::str>(value)) {
    return ir::Attribute{value.cast<std::string>()};
  }
  if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
    std::vector<std::int64_t> values;
    for (const py::handle item : value) {
      if (!py::isinstance<py::int_>(item) || py::isinstance<py::bool_>(item)) {
        return unsupported("only lists of integers can become attributes");
      }
      values.push_back(item.cast<std::int64_t>());
    }
    return ir::Attribute{std::move(values)};
  }
  return unsupported("cannot represent " +
                     py::str(py::type::handle_of(value).attr("__name__")).cast<std::string>() +
                     " as an IR attribute");
}

/// Attaches one FX argument to a node.
///
/// Tensor-valued arguments become operands. Everything else becomes an attribute keyed by its
/// position, so `aten.add.Tensor(%x, 2.0)` keeps the 2.0 where it was rather than losing which
/// argument it filled.
Result<void> attach_argument(const py::handle& argument,
                             std::size_t position,
                             const std::unordered_map<std::string, ir::Value*>& scope,
                             ir::Node& node) {
  const auto value = py::reinterpret_borrow<py::object>(argument);
  if (is_fx_node(value)) {
    const auto name = value.attr("name").cast<std::string>();
    const auto it = scope.find(name);
    if (it == scope.end()) {
      return internal_error("fx node references %" + name + " before it is defined");
    }
    node.append_operand(it->second);
    return ok();
  }

  auto attribute = to_attribute(argument);
  if (!attribute) {
    return std::move(attribute).error().with_context("argument " + std::to_string(position) +
                                                     " of '" + node.op() + "'");
  }
  node.set_attribute("arg" + std::to_string(position), std::move(attribute).value());
  return ok();
}

Result<void> import_call(const py::object& fx_node,
                         std::unordered_map<std::string, ir::Value*>& scope,
                         ir::Graph& graph) {
  const auto op = py::str(fx_node.attr("target")).cast<std::string>();
  ir::Node* node = graph.add_node(op);

  // Materialised before iterating. `attr()` yields an accessor proxy, and iterating one directly
  // does not give a stable sequence.
  const py::object args = fx_node.attr("args");
  std::size_t position = 0;
  for (const py::handle argument : args) {
    TESSIQ_RETURN_IF_ERROR(attach_argument(argument, position++, scope, *node));
  }

  const auto kwargs = fx_node.attr("kwargs").cast<py::dict>();
  for (const auto item : kwargs) {
    auto attribute = to_attribute(item.second);
    if (!attribute) {
      return std::move(attribute).error().with_context(
          "keyword argument '" + py::str(item.first).cast<std::string>() + "' of '" + op + "'");
    }
    node->set_attribute(py::str(item.first).cast<std::string>(), std::move(attribute).value());
  }

  TESSIQ_ASSIGN_OR_RETURN(const ir::TensorType type, type_of(fx_node));
  const auto name = fx_node.attr("name").cast<std::string>();
  scope[name] = node->append_result(name, type);
  return ok();
}

Result<void> import_output(const py::object& fx_node,
                           const std::unordered_map<std::string, ir::Value*>& scope,
                           ir::Graph& graph) {
  // args is a one-element tuple holding the tuple of returned values, so the results are one
  // level further in than the shape of every other node suggests.
  const auto args = fx_node.attr("args").cast<py::tuple>();
  if (args.size() != 1) {
    return invalid_argument("fx output node has " + std::to_string(args.size()) +
                            " arguments, expected exactly one");
  }

  const py::object returned_values = args[0];
  std::vector<ir::Value*> results;
  for (const py::handle returned : returned_values) {
    const auto value = py::reinterpret_borrow<py::object>(returned);
    // A backward graph returns None at every position whose input takes no gradient. Nothing is
    // computed there, so there is no value to record. The FX graph stays the authority on output
    // arity; results() holds the values the graph computes, which is what the passes need.
    // Anything rebuilding a return value needs an output spec as well, for which see
    // docs/design_notes/0002-output-structure-is-calling-convention.md.
    if (value.is_none()) {
      continue;
    }
    if (!is_fx_node(value)) {
      return unsupported("a graph returning the constant '" + py::str(value).cast<std::string>() +
                         "' is not supported yet; only tensors and None can be returned");
    }
    const auto name = value.attr("name").cast<std::string>();
    const auto it = scope.find(name);
    if (it == scope.end()) {
      return internal_error("fx output references %" + name + " before it is defined");
    }
    results.push_back(it->second);
  }

  graph.set_results(std::move(results));
  return ok();
}

/// Imports one FX node according to its kind.
Result<void> import_node(const py::object& fx_node,
                         std::unordered_map<std::string, ir::Value*>& scope,
                         ir::Graph& graph) {
  const auto kind = fx_node.attr("op").cast<std::string>();

  if (kind == "placeholder") {
    TESSIQ_ASSIGN_OR_RETURN(const ir::TensorType type, type_of(fx_node));
    const auto name = fx_node.attr("name").cast<std::string>();
    scope[name] = graph.add_param(name, type);
    return ok();
  }
  if (kind == "call_function") {
    return import_call(fx_node, scope, graph);
  }
  if (kind == "output") {
    return import_output(fx_node, scope, graph);
  }

  // call_method, call_module and get_attr should all be gone after AOTAutograd. Meeting one means
  // the graph did not come from where this importer expects, and importing it anyway would produce
  // IR that does not mean what the model means.
  return unsupported("fx node kind '" + kind +
                     "' is not supported; the graph should be functionalised and decomposed to "
                     "core ATen before import");
}

} // namespace

Result<std::unique_ptr<ir::Graph>> import_fx(const py::object& graph_module, std::string name) {
  auto graph = std::make_unique<ir::Graph>(std::move(name));
  std::unordered_map<std::string, ir::Value*> scope;

  // Snapshotted into a real list. fx exposes `graph.nodes` as a view over an intrusive linked
  // list, and iterating that view from C++ does not terminate: the walk wraps past the sentinel
  // and starts again. A copy also means the walk is unaffected if the graph is ever mutated
  // while it is in progress.
  const py::list nodes(graph_module.attr("graph").attr("nodes"));
  for (const py::handle handle : nodes) {
    TESSIQ_RETURN_IF_ERROR(import_node(py::reinterpret_borrow<py::object>(handle), scope, *graph));
  }

  TESSIQ_RETURN_IF_ERROR(ir::verify(*graph));
  TESSIQ_LOG(kDebug, kLogModule) << "imported " << graph->node_count() << " node(s)";
  return graph;
}

} // namespace tessiq::torch_backend
