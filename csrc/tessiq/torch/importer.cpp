// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/torch/importer.h"

#include <ATen/core/Tensor.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorImpl.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

/// What an argument list whose entries are computed reads as in the IR.
constexpr std::string_view kComputedList = "dynamic";

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
/// A symbolic extent or step as the IR spells it: the number when torch knows it, `kDynamic`
/// otherwise. maybe_as_int() is the question that does not add a guard, which matters because a
/// guard added here would narrow the graph torch compiles on our behalf.
std::int64_t known_or_dynamic(const c10::SymInt& value) {
  const std::optional<std::int64_t> known = value.maybe_as_int();
  return known.value_or(ir::kDynamic);
}

/// The type of a tensor whose shape torch only knows symbolically.
///
/// Every unknown extent becomes the same `?`. The IR does not record which symbol an extent came
/// from, so it cannot tell two unknowns apart, and nothing here concludes that two `?` axes match.
/// The kernels that run on a shape like this check the sizes they were actually handed instead.
ir::TensorType symbolic_tensor_type(const at::Tensor& tensor, ir::DType dtype) {
  const c10::TensorImpl* impl = tensor.unsafeGetTensorImpl();

  std::vector<std::int64_t> dims;
  dims.reserve(static_cast<std::size_t>(tensor.dim()));
  for (const c10::SymInt& dim : tensor.sym_sizes()) {
    dims.push_back(known_or_dynamic(dim));
  }

  // Asked of torch rather than worked out from the steps, because `[?,32]` with steps `[32,1]`
  // is contiguous and comparing an unknown step against an unknown product cannot show it.
  // GUARD_OR_FALSE answers "not contiguous" when it cannot decide, which costs a fusion rather
  // than producing a kernel that reads the wrong elements.
  if (TORCH_GUARD_OR_FALSE(impl->sym_is_contiguous())) {
    return {dtype, ir::Shape(std::move(dims))};
  }

  ir::Strides strides;
  strides.reserve(static_cast<std::size_t>(tensor.dim()));
  for (const c10::SymInt& stride : tensor.sym_strides()) {
    strides.push_back(known_or_dynamic(stride));
  }
  return {dtype, ir::Shape(std::move(dims)), std::move(strides)};
}

Result<ir::TensorType> tensor_type(const at::Tensor& tensor) {
  TESSIQ_ASSIGN_OR_RETURN(const ir::DType dtype, to_ir_dtype(tensor.scalar_type()));

  if (tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
    return symbolic_tensor_type(tensor, dtype);
  }

  std::vector<std::int64_t> dims;
  dims.reserve(static_cast<std::size_t>(tensor.dim()));
  for (const auto dim : tensor.sizes()) {
    dims.push_back(dim);
  }

  // Taken from the fake tensor rather than derived. A permute produces a value with the same
  // shape as something contiguous and a different order of steps through it, and torch has
  // already worked out which. Deriving it here would be a second implementation of that, and the
  // two would disagree on the first operation this compiler did not model exactly.
  ir::Strides strides;
  strides.reserve(static_cast<std::size_t>(tensor.dim()));
  for (const auto stride : tensor.strides()) {
    strides.push_back(stride);
  }

  return ir::TensorType(dtype, ir::Shape(std::move(dims)), std::move(strides));
}

/// The IR type of a symbolic scalar, or nothing when `value` is not one.
///
/// A graph traced for more than one shape carries its extents as values: `s33`, `s50` and the
/// `s33*s50` a view multiplies them into. They are integers that are not known yet, so they are
/// typed as the rank-0 integers they are. Nothing claims the arithmetic on them, which is right:
/// it is a handful of host operations feeding the shapes, not work for an accelerator.
std::optional<ir::DType> symbolic_scalar_dtype(const py::object& value) {
  static const py::module_ torch = py::module_::import("torch");
  if (py::isinstance(value, torch.attr("SymInt"))) {
    return ir::DType::kI64;
  }
  if (py::isinstance(value, torch.attr("SymFloat"))) {
    return ir::DType::kF64;
  }
  if (py::isinstance(value, torch.attr("SymBool"))) {
    return ir::DType::kI1;
  }
  return std::nullopt;
}

/// The types an FX node produces, one per result.
///
/// Usually one. A few operations return several: a layer norm gives the normalised tensor plus the
/// mean and the reciprocal standard deviation it computed, and a batch norm does the same. FX
/// spells reading one of them as a `getitem` on the tuple, which `import_getitem` turns back into
/// a reference to the right result.
Result<std::vector<ir::TensorType>> types_of(const py::object& node) {
  const py::object meta = node.attr("meta");
  const auto name = node.attr("name").cast<std::string>();
  if (!meta.contains("val")) {
    return invalid_argument("fx node '" + name +
                            "' has no meta['val']; the graph has not been through shape "
                            "propagation, so its types are unknown");
  }

  const py::object value = meta[py::str("val")];
  if (THPVariable_Check(value.ptr())) {
    TESSIQ_ASSIGN_OR_RETURN(ir::TensorType type, tensor_type(value.cast<at::Tensor>()));
    return std::vector<ir::TensorType>{std::move(type)};
  }
  if (const std::optional<ir::DType> scalar = symbolic_scalar_dtype(value)) {
    return std::vector<ir::TensorType>{ir::TensorType(*scalar, ir::Shape())};
  }

  if (!py::isinstance<py::tuple>(value) && !py::isinstance<py::list>(value)) {
    return unsupported("fx node '" + name +
                       "' produces something other than a tensor or a tuple of them, which the "
                       "IR cannot hold");
  }

  std::vector<ir::TensorType> types;
  for (const py::handle item : value) {
    if (!THPVariable_Check(item.ptr())) {
      // A tuple holding something that is not a tensor, such as the symbolic integer some
      // operations return beside their output. Nothing here can give that a type.
      return unsupported("fx node '" + name +
                         "' returns a tuple holding something that is not a "
                         "tensor, which the IR cannot hold");
    }
    TESSIQ_ASSIGN_OR_RETURN(ir::TensorType type, tensor_type(item.cast<at::Tensor>()));
    types.push_back(std::move(type));
  }
  if (types.empty()) {
    return unsupported("fx node '" + name + "' returns an empty tuple");
  }
  return types;
}

bool is_fx_node(const py::object& value) {
  static const py::object node_type = py::module_::import("torch.fx").attr("Node");
  return py::isinstance(value, node_type);
}

/// Whether `value` is one of torch's enumerations, which the compiler carries without reading.
///
/// A memory format or a dtype decides what an operation does, but nothing in this compiler acts
/// on one: the operations that take them have no kernel here and run through the fallback, which
/// reads the original FX graph rather than this IR. So carrying the spelling is enough, and it is
/// better than refusing the graph the argument appears in.
bool is_opaque_torch_label(const py::handle& value) {
  static const py::module_ torch = py::module_::import("torch");
  for (const char* name : {"memory_format", "dtype", "layout", "device"}) {
    if (py::isinstance(value, torch.attr(name))) {
      return true;
    }
  }
  return false;
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
  // Labels the compiler never interprets: torch.contiguous_format, torch.float32, torch.strided,
  // and a device. They have to survive the import because a real graph carries them, and their
  // canonical spelling is all anything here needs. Only these four, so a type that does mean
  // something to a pass still refuses rather than arriving as an uninterpretable string.
  if (is_opaque_torch_label(value)) {
    return ir::Attribute{py::str(value).cast<std::string>()};
  }

  return unsupported("cannot represent " +
                     py::str(py::type::handle_of(value).attr("__name__")).cast<std::string>() +
                     " as an IR attribute");
}

/// What the importer carries from one FX node to the next.
struct ImportState {
  /// IR value for each FX name, which is what an operand lookup resolves through.
  std::unordered_map<std::string, ir::Value*> scope;

  /// IR node for each FX name, so a getitem can reach a result other than the first.
  std::unordered_map<std::string, ir::Node*> producers;

  /// The FX name of the getitem that reads result `index` of a multi-result node, keyed by the
  /// producer's FX name. Collected before anything is imported, because a getitem appears after
  /// the node it reads and the results are named when that node is imported.
  std::unordered_map<std::string, std::unordered_map<std::int64_t, std::string>> selections;
};

/// Whether a list or tuple argument holds a value the graph computes rather than a constant.
///
/// A reshape in a graph traced for more than one shape takes its sizes as `[s33*s50, 32]`, where
/// the first entry is a node. The list is an argument, not a constant.
bool holds_fx_node(const py::object& value) {
  if (!py::isinstance<py::list>(value) && !py::isinstance<py::tuple>(value)) {
    return false;
  }
  for (const py::handle item : value) {
    if (is_fx_node(py::reinterpret_borrow<py::object>(item))) {
      return true;
    }
  }
  return false;
}

/// Marks an argument list whose entries the graph computes.
///
/// The nodes in it become operands, so the IR shows what the operation reads. The attribute
/// records the spelling `kDynamic` rather than a list of numbers, because there are no numbers:
/// a pass reading it finds something it cannot interpret and declines the node, which is what an
/// operation whose sizes arrive at run time should do until the kernels can take them.
Result<void> attach_computed_list(const py::object& value,
                                  std::size_t position,
                                  const std::unordered_map<std::string, ir::Value*>& scope,
                                  ir::Node& node) {
  for (const py::handle item : value) {
    const auto entry = py::reinterpret_borrow<py::object>(item);
    if (!is_fx_node(entry)) {
      continue;
    }
    const auto name = entry.attr("name").cast<std::string>();
    const auto it = scope.find(name);
    if (it == scope.end()) {
      return internal_error("fx node references %" + name + " before it is defined");
    }
    node.append_operand(it->second);
  }
  node.set_attribute("arg" + std::to_string(position), ir::Attribute{std::string(kComputedList)});
  return ok();
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

  if (holds_fx_node(value)) {
    return attach_computed_list(value, position, scope, node);
  }

  auto attribute = to_attribute(argument);
  if (!attribute) {
    return std::move(attribute).error().with_context("argument " + std::to_string(position) +
                                                     " of '" + node.op() + "'");
  }
  node.set_attribute("arg" + std::to_string(position), std::move(attribute).value());
  return ok();
}

/// How the IR spells an FX target.
///
/// A torch overload already prints as `aten.view.default`. A plain Python function prints as
/// `<built-in function mul>`, which is not a name the textual IR can be read back from, so one is
/// built from the module and the function: `operator.mul`.
std::string op_name(const py::object& target) {
  auto printed = py::str(target).cast<std::string>();
  if (!printed.starts_with('<') || !py::hasattr(target, "__qualname__")) {
    return printed;
  }
  // `_operator` is the C module behind `operator`, and the second is what anyone reading the IR
  // would go looking for.
  const auto module = py::hasattr(target, "__module__")
                          ? py::str(target.attr("__module__")).cast<std::string>()
                          : std::string();
  const std::string prefix = module == "_operator" ? "operator" : module;
  const auto function = target.attr("__qualname__").cast<std::string>();
  return prefix.empty() ? function : prefix + "." + function;
}

Result<void> import_call(const py::object& fx_node, ImportState& state, ir::Graph& graph) {
  const auto op = op_name(fx_node.attr("target"));
  ir::Node* node = graph.add_node(op);

  // Materialised before iterating. `attr()` yields an accessor proxy, and iterating one directly
  // does not give a stable sequence.
  const py::object args = fx_node.attr("args");
  std::size_t position = 0;
  for (const py::handle argument : args) {
    TESSIQ_RETURN_IF_ERROR(attach_argument(argument, position++, state.scope, *node));
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

  TESSIQ_ASSIGN_OR_RETURN(const std::vector<ir::TensorType> types, types_of(fx_node));
  const auto name = fx_node.attr("name").cast<std::string>();
  const auto selected = state.selections.find(name);
  for (std::size_t i = 0; i < types.size(); ++i) {
    // A result that a getitem reads takes that getitem's FX name. The executor runs the FX graph
    // and looks its tensors up by FX name, so a result named anything else would resolve to the
    // whole tuple the producer returned rather than to the tensor a kernel wants.
    std::string result_name;
    if (selected != state.selections.end()) {
      const auto reader = selected->second.find(static_cast<std::int64_t>(i));
      if (reader != selected->second.end()) {
        result_name = reader->second;
      }
    }
    if (result_name.empty()) {
      // Nothing reads it. Named after its position with a dot, because FX names its own nodes
      // with an underscore and a number: `native_layer_norm_1` is the second layer norm in the
      // graph, not the second result of the first one, and an underscore collided with it.
      result_name = i == 0 ? name : graph.unique_name(name + "." + std::to_string(i));
    }
    node->append_result(result_name, types[i]);
  }
  state.scope[name] = node->result(0);
  state.producers[name] = node;
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

/// Whether `fx_node` is the `operator.getitem` that reads one result out of a tuple.
bool is_getitem(const py::object& fx_node) {
  static const py::object getitem = py::module_::import("operator").attr("getitem");
  return py::object(fx_node.attr("target")).is(getitem);
}

/// Binds a getitem's name to the result it selects, without adding a node.
///
/// FX spells "the second thing native_layer_norm returned" as a getitem on the tuple. The IR has
/// no tuples: a node has several results and a value refers to one of them directly. So a getitem
/// is not an operation here, it is a name for something that already exists.
Result<void> import_getitem(const py::object& fx_node, ImportState& state) {
  const auto args = fx_node.attr("args").cast<py::tuple>();
  const auto name = fx_node.attr("name").cast<std::string>();
  if (args.size() != 2 || !py::isinstance<py::int_>(args[1])) {
    return unsupported("fx getitem '" + name + "' does not index a node by an integer");
  }
  const auto tuple = args[0].cast<py::object>();
  if (!is_fx_node(tuple)) {
    return unsupported("fx getitem '" + name + "' does not index a node");
  }

  const auto source = tuple.attr("name").cast<std::string>();
  const auto found = state.producers.find(source);
  if (found == state.producers.end()) {
    return invalid_argument("fx getitem '" + name + "' reads '" + source + "', which has no node");
  }

  const auto index = args[1].cast<std::int64_t>();
  ir::Node* producer = found->second;
  if (index < 0 || static_cast<std::size_t>(index) >= producer->result_count()) {
    return invalid_argument("fx getitem '" + name + "' takes result " + std::to_string(index) +
                            " of '" + source + "', which has " +
                            std::to_string(producer->result_count()));
  }

  state.scope[name] = producer->result(static_cast<std::size_t>(index));
  return ok();
}

/// Records which getitem reads which result, before any of them are imported.
///
/// A getitem appears after the node it reads, and the results are named when that node is
/// imported, so the graph has to be walked once first.
Result<void> collect_selections(const py::list& nodes, ImportState& state) {
  for (const py::handle handle : nodes) {
    const auto fx_node = py::reinterpret_borrow<py::object>(handle);
    if (fx_node.attr("op").cast<std::string>() != "call_function" || !is_getitem(fx_node)) {
      continue;
    }
    const auto args = fx_node.attr("args").cast<py::tuple>();
    if (args.size() != 2 || !py::isinstance<py::int_>(args[1])) {
      continue;
    }
    const auto tuple = args[0].cast<py::object>();
    if (!is_fx_node(tuple)) {
      continue;
    }
    state.selections[tuple.attr("name").cast<std::string>()].emplace(
        args[1].cast<std::int64_t>(), fx_node.attr("name").cast<std::string>());
  }
  return ok();
}

/// Imports one FX node according to its kind.
Result<void> import_node(const py::object& fx_node, ImportState& state, ir::Graph& graph) {
  const auto kind = fx_node.attr("op").cast<std::string>();

  if (kind == "placeholder") {
    TESSIQ_ASSIGN_OR_RETURN(const std::vector<ir::TensorType> types, types_of(fx_node));
    const auto name = fx_node.attr("name").cast<std::string>();
    if (types.size() != 1) {
      return unsupported("fx placeholder '" + name + "' is not a single tensor");
    }
    state.scope[name] = graph.add_param(name, types.front());
    return ok();
  }
  if (kind == "call_function") {
    return is_getitem(fx_node) ? import_getitem(fx_node, state)
                               : import_call(fx_node, state, graph);
  }
  if (kind == "output") {
    return import_output(fx_node, state.scope, graph);
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
  // Snapshotted into a real list. fx exposes `graph.nodes` as a view over an intrusive linked
  // list, and iterating that view from C++ does not terminate: the walk wraps past the sentinel
  // and starts again. A copy also means the walk is unaffected if the graph is ever mutated
  // while it is in progress, and it is walked twice.
  const py::list nodes(graph_module.attr("graph").attr("nodes"));

  ImportState state;
  TESSIQ_RETURN_IF_ERROR(collect_selections(nodes, state));

  for (const py::handle handle : nodes) {
    TESSIQ_RETURN_IF_ERROR(import_node(py::reinterpret_borrow<py::object>(handle), state, *graph));
  }

  TESSIQ_RETURN_IF_ERROR(ir::verify(*graph));
  TESSIQ_LOG(kDebug, kLogModule) << "imported " << graph->node_count() << " node(s)";
  return graph;
}

} // namespace tessiq::torch_backend
