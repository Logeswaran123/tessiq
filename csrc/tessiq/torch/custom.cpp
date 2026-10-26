// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/torch/custom.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/stack.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <algorithm>
#include <atomic>
#include <c10/core/Device.h>
#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/library.h>

#include "tessiq/codegen/call.h"
#include "tessiq/codegen/precompiled.h"
#include "tessiq/codegen/program.h"
#include "tessiq/ir/text.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_registry.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/passes/pass.h"
#include "tessiq/passes/replace_op.h"
#include "tessiq/passes/rewrite.h"
#include "tessiq/runtime/allocator.h"
#include "tessiq/runtime/device.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/backend.h"
#include "tessiq/torch/custom_op.h"
#include "tessiq/torch/fused.h"
#include "tessiq/torch/importer.h"
#include "tessiq/torch/kernel.h"

namespace tessiq::torch_backend {

using custom_ops::CustomOp;
using custom_ops::Parameter;

/// A custom operation after its definition was checked and registered.
struct RegisteredOp {
  const CustomOp* definition = nullptr;

  /// `tessiq::rms_norm`, as torch names it.
  std::string qualified;

  /// `tessiq.rms_norm.default`, as a graph spells it.
  std::string spelling;

  /// Every argument's type, in schema order.
  std::vector<Parameter> parameters;

  /// Every argument's name, in schema order.
  std::vector<std::string> names;

  /// How many of the arguments are tensors.
  int arity = 0;

  /// The torch operation it stands in for, as torch names it (`aten::_softmax`), or empty.
  std::string replaced;

  /// The same torch operation as a graph spells it (`aten._softmax.default`).
  std::string replaced_spelling;

  /// The kernel file's text. Empty unless the operation's kernel is compiled at run time.
  std::string_view source;

  /// The operation's host code, when it declares some and this build compiled it.
  const codegen::Precompiled* precompiled = nullptr;

  /// The schema as parsed when the definition was checked, handed to torch to register.
  std::unique_ptr<c10::FunctionSchema> schema;

  /// The torch operation, found once it is registered.
  std::unique_ptr<c10::OperatorHandle> handle;

  std::atomic<std::int64_t> fallbacks{0};

  std::mutex mutex;
  /// The kernel compiled for each CUDA ordinal. Guarded by `mutex`.
  std::map<int, const codegen::Program*> programs;
  /// The expression compiled for each element type and CUDA ordinal. Guarded by `mutex`.
  std::map<std::pair<at::ScalarType, int>, std::shared_ptr<Kernel>> expressions;
  /// The CUDA ordinals the host code has been checked to load on. Guarded by `mutex`.
  std::set<int> loads_on;

  [[nodiscard]] bool has_kernel() const { return !source.empty() || precompiled != nullptr; }

  /// Whether a call can run on a GPU at all: false for host code this build did not compile.
  [[nodiscard]] bool runs_on_gpu() const {
    return has_kernel() || !definition->expression_text().empty();
  }
};

namespace {

constexpr std::string_view kLogModule = "custom_ops";

/// The namespace every custom operation is registered under.
constexpr std::string_view kNamespace = "tessiq";

// ---------------------------------------------------------------------------------------------
// Registry

std::map<std::string, std::unique_ptr<RegisteredOp>, std::less<>>& registered_ops() {
  static std::map<std::string, std::unique_ptr<RegisteredOp>, std::less<>> ops;
  return ops;
}

/// The torch libraries the operations were registered through. An operation stays registered
/// only as long as its library lives, which is as long as the process.
std::vector<std::unique_ptr<torch::Library>>& libraries() {
  static std::vector<std::unique_ptr<torch::Library>> held;
  return held;
}

RegisteredOp* find_qualified(std::string_view qualified) {
  auto& ops = registered_ops();
  const auto found = ops.find(qualified);
  return found == ops.end() ? nullptr : found->second.get();
}

/// The operations that stand in for a torch one, by the torch operation's name.
std::map<std::string, RegisteredOp*, std::less<>>& replacements() {
  static std::map<std::string, RegisteredOp*, std::less<>> ops;
  return ops;
}

/// The name of every pass that puts a custom operation in place of a torch one.
std::vector<std::string>& replacing_passes() {
  static std::vector<std::string> names;
  return names;
}

/// The name of each operation's rewrite pass, by the operation's name.
std::map<std::string, std::string, std::less<>>& rewrite_passes() {
  static std::map<std::string, std::string, std::less<>> passes;
  return passes;
}

/// `rms_norm` as a pass name spells it, `rms-norm`, like every other pass.
std::string hyphenated(std::string name) {
  std::ranges::replace(name, '_', '-');
  return name;
}

/// `(x, weight)`.
std::string listed(std::vector<std::string> names) {
  std::string text = "(";
  for (std::size_t i = 0; i < names.size(); ++i) {
    text += i == 0 ? "" : ", ";
    text += names[i];
  }
  return text + ")";
}

/// The custom operation a call through `handle` runs: the operation itself, or the one standing in
/// for the torch operation `handle` names.
RegisteredOp* find_for(const c10::OperatorHandle& handle) {
  if (RegisteredOp* op = find_qualified(handle.schema().name())) {
    return op;
  }
  const auto& ops = replacements();
  const auto found = ops.find(c10::toString(handle.operator_name()));
  return found == ops.end() ? nullptr : found->second;
}

RegisteredOp* find_spelled(std::string_view spelling) {
  for (auto& [qualified, op] : registered_ops()) {
    if (op->spelling == spelling) {
      return op.get();
    }
  }
  return nullptr;
}

// ---------------------------------------------------------------------------------------------
// Checking a definition

std::string_view parameter_name(Parameter parameter) {
  switch (parameter) {
  case Parameter::kTensor:
    return "Tensor";
  case Parameter::kFloat:
    return "float";
  case Parameter::kInt:
    return "int";
  case Parameter::kBool:
    return "bool";
  }
  return "?";
}

/// `(Tensor, Tensor, float)`.
std::string listed(const std::vector<Parameter>& parameters) {
  std::string text = "(";
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    text += i == 0 ? "" : ", ";
    text += parameter_name(parameters[i]);
  }
  return text + ")";
}

std::optional<Parameter> parameter_of(const c10::Argument& argument) {
  switch (argument.type()->kind()) {
  case c10::TypeKind::TensorType:
    return Parameter::kTensor;
  case c10::TypeKind::FloatType:
    return Parameter::kFloat;
  case c10::TypeKind::IntType:
    return Parameter::kInt;
  case c10::TypeKind::BoolType:
    return Parameter::kBool;
  default:
    return std::nullopt;
  }
}

/// The part of a definition that comes from its schema.
struct Signature {
  std::unique_ptr<c10::FunctionSchema> schema;
  std::vector<Parameter> parameters;
  std::vector<std::string> names;
  int arity = 0;
};

Result<Signature> read_signature(const std::string& qualified, const CustomOp& op) {
  Signature signature;
  try {
    signature.schema =
        std::make_unique<c10::FunctionSchema>(torch::jit::parseSchema(qualified + op.signature()));
  } catch (const c10::Error& error) {
    return invalid_argument(std::string("its schema does not parse: ") +
                            error.what_without_backtrace());
  }

  const auto& returns = signature.schema->returns();
  if (returns.size() != 1 || parameter_of(returns.front()) != Parameter::kTensor ||
      returns.front().alias_info() != nullptr) {
    return invalid_argument("it has to return one new Tensor");
  }
  for (const c10::Argument& argument : signature.schema->arguments()) {
    const std::optional<Parameter> parameter = parameter_of(argument);
    if (!parameter.has_value()) {
      return invalid_argument("argument '" + argument.name() + "' is a " + argument.type()->str() +
                              "; a custom operation takes Tensor, float, int and bool");
    }
    if (argument.alias_info() != nullptr) {
      return invalid_argument("argument '" + argument.name() +
                              "' is written to; a custom operation returns its result instead");
    }
    // A compiled graph leaves out an argument equal to its default, and the kernel would then
    // receive nothing in its place.
    if (argument.default_value().has_value()) {
      return invalid_argument("argument '" + argument.name() +
                              "' has a default, which a custom operation cannot have yet");
    }
    signature.parameters.push_back(*parameter);
    signature.names.push_back(argument.name());
    signature.arity += *parameter == Parameter::kTensor ? 1 : 0;
  }
  if (signature.arity == 0) {
    return invalid_argument("it takes no tensor, so there is nothing to run it over");
  }
  return signature;
}

/// `aten::add.Tensor` as torch keys it, the name `aten::add` and the overload `Tensor`.
c10::OperatorName operator_name_of(const std::string& text) {
  const std::size_t dot = text.find('.', text.find("::"));
  if (dot == std::string::npos) {
    return {text, ""};
  }
  return {text.substr(0, dot), text.substr(dot + 1)};
}

/// The torch operation as a graph spells it: `aten._softmax.default`, `aten.add.Tensor`.
std::string spelling_of(const c10::OperatorName& name) {
  std::string spelled = name.name;
  spelled.replace(spelled.find("::"), 2, ".");
  return spelled + "." + (name.overload_name.empty() ? "default" : name.overload_name);
}

/// Checks that the torch operation `op` replaces exists, and takes what `op` takes.
Result<void> check_replacement(const CustomOp& op, RegisteredOp& checked) {
  const std::string& text = op.replaced();
  if (text.find("::") == std::string::npos) {
    return invalid_argument("it replaces '" + text +
                            "', which is not a torch operation's name; write it as aten::_softmax "
                            "or aten::add.Tensor");
  }
  const c10::OperatorName name = operator_name_of(text);
  const std::optional<c10::OperatorHandle> handle = c10::Dispatcher::singleton().findSchema(name);
  if (!handle.has_value()) {
    return invalid_argument("it replaces " + text + ", which torch does not have");
  }
  // Such an operation is broken into others as it is called, before a backend's kernel or a graph
  // could see it, so there would be nothing to replace.
  if (handle->hasKernelForDispatchKey(c10::DispatchKey::CompositeImplicitAutograd)) {
    return invalid_argument("it replaces " + text +
                            ", which torch breaks into other operations before any backend sees "
                            "it; replace one of those instead");
  }
  if (handle->hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1)) {
    return invalid_argument("it replaces " + text + ", which tessiq already implements itself");
  }

  const c10::FunctionSchema& theirs = handle->schema();
  const std::vector<c10::Argument>& arguments = theirs.arguments();
  bool same = theirs.returns().size() == 1 && arguments.size() == checked.parameters.size();
  for (std::size_t i = 0; same && i < arguments.size(); ++i) {
    same = arguments[i].name() == checked.names[i] &&
           parameter_of(arguments[i]) == checked.parameters[i] &&
           !arguments[i].default_value().has_value() && arguments[i].alias_info() == nullptr;
  }
  if (!same) {
    return invalid_argument("it replaces " + text + ", whose schema is " + c10::toString(theirs) +
                            "; the two have to take the same arguments, with the same names and "
                            "types, and no defaults");
  }
  checked.replaced = c10::toString(name);
  checked.replaced_spelling = spelling_of(name);
  return ok();
}

/// Checks that a definition holds together, before anything is registered.
Result<std::unique_ptr<RegisteredOp>> check_definition(const CustomOp& op) {
  auto checked = std::make_unique<RegisteredOp>();
  checked->definition = &op;
  checked->qualified = std::string(kNamespace) + "::" + op.name();
  checked->spelling = std::string(kNamespace) + "." + op.name() + ".default";

  TESSIQ_ASSIGN_OR_RETURN(Signature signature, read_signature(checked->qualified, op));
  checked->schema = std::move(signature.schema);
  checked->parameters = signature.parameters;
  checked->names = signature.names;
  checked->arity = signature.arity;

  if (!op.reference_registration()) {
    return invalid_argument("it has no reference; add .reference(fn), computing it in ATen");
  }
  if (op.reference_parameters() != checked->parameters) {
    return invalid_argument("its reference takes " + listed(op.reference_parameters()) +
                            " but the schema declares " + listed(checked->parameters));
  }

  const bool has_expression = !op.expression_text().empty();
  const bool has_launch = static_cast<bool>(op.launch_function());
  const int kinds = static_cast<int>(has_expression) + static_cast<int>(has_launch) +
                    static_cast<int>(op.is_precompiled());
  if (kinds != 1) {
    return invalid_argument(
        std::string("it needs one of .launch(fn), for a kernel in a .cu file, ") +
        ".expression(text), or .precompiled(), for host code in a .nvcc.cu " + "file, and it has " +
        (kinds == 0 ? "none" : std::to_string(kinds)));
  }
  if (op.is_precompiled()) {
    checked->precompiled = codegen::find_precompiled(op.name());
  }
  if (has_expression && std::cmp_not_equal(checked->arity, checked->parameters.size())) {
    return invalid_argument("an operation defined by an expression takes only tensors");
  }
  if (has_launch) {
    if (op.launch_parameters() != checked->parameters) {
      return invalid_argument("its launch function takes " + listed(op.launch_parameters()) +
                              " before the result, but the schema declares " +
                              listed(checked->parameters));
    }
    const std::string key = op.directory() + "/" + op.kernel_file();
    const std::optional<std::string_view> source = custom_ops::find_embedded_file(key);
    if (!source.has_value() || source->empty()) {
      return invalid_argument("there is no kernel file csrc/custom_ops/" + key);
    }
    checked->source = *source;
  }
  for (const at::ScalarType dtype : op.element_types()) {
    if (!to_ir_dtype(dtype).has_value()) {
      return invalid_argument(std::string("it declares ") + c10::toString(dtype) +
                              ", which the compiler has no type for");
    }
  }
  if (!op.replaced().empty()) {
    TESSIQ_RETURN_IF_ERROR(check_replacement(op, *checked));
  }
  return checked;
}

// ---------------------------------------------------------------------------------------------
// Running an operation

/// The GPU a call can run its kernel on, or nothing when it has to run the reference.
///
/// Every tensor has to be on the same GPU and of a type the operation declares. An expression is
/// compiled for one element type, so its operands also have to share one.
std::optional<int> runnable_on(const RegisteredOp& op, const std::vector<c10::IValue>& arguments) {
  if (!op.runs_on_gpu()) {
    return std::nullopt;
  }
  const std::vector<at::ScalarType>& declared = op.definition->element_types();
  std::optional<c10::Device> device;
  std::optional<at::ScalarType> dtype;
  for (const c10::IValue& argument : arguments) {
    if (!argument.isTensor()) {
      continue;
    }
    const at::Tensor& tensor = argument.toTensor();
    if (device.has_value() && tensor.device() != *device) {
      return std::nullopt;
    }
    if (!op.has_kernel() && dtype.has_value() && tensor.scalar_type() != *dtype) {
      return std::nullopt;
    }
    if (std::ranges::find(declared, tensor.scalar_type()) == declared.end()) {
      return std::nullopt;
    }
    device = tensor.device();
    dtype = tensor.scalar_type();
  }
  return device.has_value() ? cuda_ordinal_of(*device) : std::nullopt;
}

/// The result of a call as a tensor with a shape and no data, from running the reference on
/// arguments like that.
at::Tensor result_like(const RegisteredOp& op, const std::vector<c10::IValue>& arguments) {
  torch::jit::Stack stack;
  stack.reserve(arguments.size());
  for (const c10::IValue& argument : arguments) {
    if (argument.isTensor()) {
      const at::Tensor& tensor = argument.toTensor();
      stack.emplace_back(
          at::empty_strided(tensor.sizes(), tensor.strides(), tensor.options().device(c10::kMeta)));
    } else {
      stack.push_back(argument);
    }
  }
  op.handle->callBoxed(stack);
  return stack.back().toTensor();
}

Result<const codegen::Program*> program_for(RegisteredOp& op, int ordinal) {
  const std::scoped_lock guard(op.mutex);
  if (const auto found = op.programs.find(ordinal); found != op.programs.end()) {
    return found->second;
  }
  Result<const codegen::Program*> compiled =
      codegen::compile_program(op.source, op.definition->entry(), ordinal);
  if (!compiled.has_value()) {
    return compiled.error().with_context(op.qualified + " does not compile");
  }
  op.programs.emplace(ordinal, *compiled);
  return *compiled;
}

Result<std::shared_ptr<Kernel>>
expression_for(RegisteredOp& op, at::ScalarType dtype, int ordinal) {
  const std::scoped_lock guard(op.mutex);
  const auto key = std::make_pair(dtype, ordinal);
  if (const auto found = op.expressions.find(key); found != op.expressions.end()) {
    return found->second;
  }
  TESSIQ_ASSIGN_OR_RETURN(const ir::DType type, to_ir_dtype(dtype));
  Result<std::shared_ptr<Kernel>> compiled =
      compile_elementwise(op.spelling, op.arity, type, ordinal);
  if (!compiled.has_value()) {
    return compiled.error().with_context(op.qualified + " does not compile");
  }
  op.expressions.emplace(key, *compiled);
  return *compiled;
}

/// Makes `op`'s kernel ready to run on CUDA device `ordinal`: compiles a `.cu` kernel, or checks
/// that host code was built for the card's architecture.
Result<void> ready_on(RegisteredOp& op, int ordinal) {
  if (op.precompiled == nullptr) {
    TESSIQ_RETURN_IF_ERROR(program_for(op, ordinal));
    return ok();
  }
  const std::scoped_lock guard(op.mutex);
  if (!op.loads_on.contains(ordinal)) {
    TESSIQ_RETURN_IF_ERROR(codegen::check_runs_on(*op.precompiled, ordinal));
    op.loads_on.insert(ordinal);
  }
  return ok();
}

codegen::ElementType element_type(at::ScalarType type) {
  switch (type) {
  case at::kFloat:
    return codegen::ElementType::kFloat32;
  case at::kDouble:
    return codegen::ElementType::kFloat64;
  case at::kHalf:
    return codegen::ElementType::kFloat16;
  case at::kBFloat16:
    return codegen::ElementType::kBFloat16;
  case at::kInt:
    return codegen::ElementType::kInt32;
  case at::kLong:
    return codegen::ElementType::kInt64;
  case at::kBool:
    return codegen::ElementType::kBool;
  default:
    return codegen::ElementType::kOther;
  }
}

codegen::DeviceTensor device_tensor(const at::Tensor& tensor) {
  codegen::DeviceTensor device;
  device.data = tensor.data_ptr();
  device.sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
  device.type = element_type(tensor.scalar_type());
  return device;
}

/// Gives back the workspace a call took on tessiq device `index`, once nothing still reads it.
///
/// Every launch tessiq makes goes on the legacy default stream, which is ordered against the rest
/// of the device's work, so a block given back is not handed out again before this call's work
/// finishes. Work on any other stream lacks that ordering, so the stream is waited for first. A
/// stream capturing a CUDA graph cannot be waited for, and every replay of the graph reads the
/// blocks again, so those are kept for the life of the process.
Result<void> give_back(const std::vector<void*>& blocks, int index, codegen::Stream stream) {
  if (blocks.empty()) {
    return ok();
  }
  if (stream.handle != nullptr) {
    TESSIQ_ASSIGN_OR_RETURN(const bool capturing, codegen::is_capturing(stream));
    if (capturing) {
      TESSIQ_LOG(kDebug, kLogModule)
          << "keeping " << blocks.size() << " workspace block(s) that a CUDA graph captured";
      return ok();
    }
    TESSIQ_RETURN_IF_ERROR(codegen::synchronize(stream));
  }
  CachingAllocator& allocator = allocator_at(index);
  for (void* block : blocks) {
    allocator.deallocate(block);
  }
  return ok();
}

/// Runs `op`'s host code over `arguments`, whose tensors are packed, writing `out`.
///
/// False, having queued nothing, when the host code declines the call.
Result<bool> run_host_code(RegisteredOp& op,
                           const std::vector<c10::IValue>& arguments,
                           const at::Tensor& out,
                           int ordinal,
                           codegen::Stream stream) {
  TESSIQ_RETURN_IF_ERROR(ready_on(op, ordinal));
  std::vector<codegen::DeviceTensor> inputs;
  std::vector<codegen::Scalar> scalars;
  for (const c10::IValue& argument : arguments) {
    if (argument.isTensor()) {
      inputs.push_back(device_tensor(argument.toTensor()));
    } else if (argument.isDouble()) {
      scalars.emplace_back(argument.toDouble());
    } else if (argument.isInt()) {
      scalars.emplace_back(argument.toInt());
    } else {
      scalars.emplace_back(argument.toBool());
    }
  }
  const int index = static_cast<int>(out.device().index());
  codegen::Call call(std::move(inputs),
                     std::move(scalars),
                     device_tensor(out),
                     stream.handle,
                     ordinal,
                     [index](std::size_t bytes) -> void* {
                       const Result<void*> block = allocator_at(index).allocate(bytes);
                       return block.has_value() ? *block : nullptr;
                     });
  const Result<void> ran = codegen::run_precompiled(*op.precompiled, call);
  TESSIQ_RETURN_IF_ERROR(give_back(call.workspaces(), index, stream));
  TESSIQ_RETURN_IF_ERROR(ran);
  if (call.declined()) {
    TESSIQ_LOG(kDebug, kLogModule)
        << op.qualified << " declines a call, so it runs its reference: " << call.reason();
    return false;
  }
  return true;
}

/// Launches `op`'s kernel over `arguments`, whose tensors are packed, writing `out`.
///
/// Returns false, having launched nothing, when the launch function refuses the arguments with
/// TORCH_CHECK. That is how a kernel says it cannot take a call its reference can, such as a
/// weight that broadcasts, and the caller then runs the reference instead. Any other failure is an
/// error.
Result<bool> launch_kernel(RegisteredOp& op,
                           const std::vector<c10::IValue>& arguments,
                           const at::Tensor& out,
                           int ordinal,
                           codegen::Stream stream) {
  if (out.numel() == 0) {
    // Nothing to compute, and a launch of no blocks is an error.
    return true;
  }
  if (op.precompiled != nullptr) {
    return run_host_code(op, arguments, out, ordinal, stream);
  }
  custom_ops::Launch launch;
  try {
    launch = op.definition->launch_function()(arguments, out);
  } catch (const c10::Error& error) {
    TESSIQ_LOG(kDebug, kLogModule)
        << op.qualified
        << " declines a call, so it runs its reference: " << error.what_without_backtrace();
    return false;
  }
  TESSIQ_ASSIGN_OR_RETURN(const codegen::Program* program, program_for(op, ordinal));

  std::vector<void*> buffers;
  for (const c10::IValue& argument : arguments) {
    if (argument.isTensor()) {
      buffers.push_back(argument.toTensor().data_ptr());
    }
  }
  buffers.push_back(out.data_ptr());
  codegen::LaunchShape shape;
  shape.grid = launch.grid;
  shape.block = launch.block;
  shape.shared_bytes = launch.shared_bytes;
  TESSIQ_RETURN_IF_ERROR(program->launch(buffers, launch.arguments, shape, stream));
  return true;
}

/// Runs `op`'s reference on host copies of `arguments` and writes what it gives into `out`.
///
/// For a call the kernel declined in a compiled graph, where the result already has a tensor the
/// executor allocated. A call made outside one goes through the dispatcher's own fallback instead.
Result<void>
reference_into(RegisteredOp& op, const std::vector<c10::IValue>& arguments, const at::Tensor& out) {
  op.fallbacks.fetch_add(1, std::memory_order_relaxed);
  torch::jit::Stack stack;
  stack.reserve(arguments.size());
  for (const c10::IValue& argument : arguments) {
    stack.push_back(argument.isTensor() ? c10::IValue(argument.toTensor().cpu()) : argument);
  }
  try {
    op.handle->callBoxed(stack);
    out.copy_(stack.back().toTensor());
  } catch (const c10::Error& error) {
    return invalid_argument(op.qualified + ": " + error.what_without_backtrace());
  }
  return ok();
}

void raise_if_error(const Result<void>& result) {
  TORCH_CHECK(result.has_value(), result.has_value() ? "" : result.error().to_string());
}

/// The result of launching `op`'s kernel, or nothing when the kernel declined the call.
std::optional<at::Tensor>
run_kernel(RegisteredOp& op, std::vector<c10::IValue> arguments, int ordinal) {
  for (c10::IValue& argument : arguments) {
    if (argument.isTensor()) {
      argument = argument.toTensor().contiguous();
    }
  }
  const at::Tensor like = result_like(op, arguments);
  const c10::Device device = [&arguments] {
    for (const c10::IValue& argument : arguments) {
      if (argument.isTensor()) {
        return argument.toTensor().device();
      }
    }
    return c10::Device(c10::kCPU);
  }();
  at::Tensor out = at::empty(like.sizes(), like.options().device(device));
  const Result<bool> launched = launch_kernel(op, arguments, out, ordinal, codegen::kDefaultStream);
  TORCH_CHECK(launched.has_value(), launched.has_value() ? "" : launched.error().to_string());
  if (!*launched) {
    return std::nullopt;
  }
  return out;
}

/// Runs an expression the way a compiled graph runs it on its own: every operand broadcast to the
/// result's shape and packed, so the kernel reads each at the result's own position.
at::Tensor
run_expression(RegisteredOp& op, const std::vector<c10::IValue>& arguments, int ordinal) {
  std::vector<at::Tensor> operands;
  operands.reserve(arguments.size());
  for (const c10::IValue& argument : arguments) {
    operands.push_back(argument.toTensor());
  }
  at::DimVector shape(operands.front().sizes());
  for (std::size_t i = 1; i < operands.size(); ++i) {
    shape = at::infer_size_dimvector(shape, operands[i].sizes());
  }
  at::Tensor out = at::empty(shape, operands.front().options());
  if (out.numel() == 0) {
    return out;
  }

  std::vector<at::Tensor> packed;
  packed.reserve(operands.size());
  for (const at::Tensor& operand : operands) {
    packed.push_back(operand.expand(shape).contiguous().view(-1));
  }
  const Result<std::shared_ptr<Kernel>> kernel = expression_for(op, out.scalar_type(), ordinal);
  TORCH_CHECK(kernel.has_value(), kernel.has_value() ? "" : kernel.error().to_string());
  raise_if_error((*kernel)->launch(packed, out.view(-1), codegen::kDefaultStream));
  return out;
}

/// What a custom operation does when it is called on a tessiq tensor.
///
/// On a GPU, in a type it declares, it runs its kernel or its expression there. Anywhere else, the
/// CPU reference device included, the arguments are copied to the host, the reference runs, and
/// the result comes back, which is counted.
void run_on_device(const c10::OperatorHandle& handle, torch::jit::Stack* stack) {
  RegisteredOp* op = find_for(handle);
  TORCH_CHECK(
      op != nullptr, handle.schema().name(), " is not a custom operation tessiq registered");

  const std::size_t count = handle.schema().arguments().size();
  const auto first = stack->end() - static_cast<std::ptrdiff_t>(count);
  std::vector<c10::IValue> arguments(first, stack->end());

  const std::optional<int> ordinal = runnable_on(*op, arguments);
  if (!ordinal.has_value()) {
    op->fallbacks.fetch_add(1, std::memory_order_relaxed);
    TESSIQ_LOG(kDebug, kLogModule) << op->qualified << " runs its reference";
    at::native::cpu_fallback(handle, stack);
    return;
  }

  std::optional<at::Tensor> out = op->has_kernel() ? run_kernel(*op, std::move(arguments), *ordinal)
                                                   : run_expression(*op, arguments, *ordinal);
  if (!out.has_value()) {
    op->fallbacks.fetch_add(1, std::memory_order_relaxed);
    at::native::cpu_fallback(handle, stack);
    return;
  }
  torch::jit::drop(*stack, count);
  torch::jit::push(*stack, std::move(*out));
}

// ---------------------------------------------------------------------------------------------
// Registering

Result<void> register_with_compiler(const RegisteredOp& op) {
  passes::OpDefinition definition;
  definition.name = op.spelling;
  definition.canonical = op.qualified;
  definition.arity = op.arity;
  if (op.has_kernel()) {
    definition.category = passes::OpCategory::kCustom;
    definition.parameters = op.names;
  } else {
    definition.category = passes::OpCategory::kElementwise;
    definition.expression = op.definition->expression_text();
  }
  for (const at::ScalarType dtype : op.definition->element_types()) {
    TESSIQ_ASSIGN_OR_RETURN(const ir::DType type, to_ir_dtype(dtype));
    definition.dtypes.push_back(type);
  }
  TESSIQ_RETURN_IF_ERROR(passes::register_op(definition));
  return ok();
}

Result<void> register_with_torch(RegisteredOp& op) {
  const std::string& name = op.definition->name();
  auto library = std::make_unique<torch::Library>(
      torch::Library::FRAGMENT, std::string(kNamespace), std::nullopt, __FILE__, __LINE__);
  try {
    c10::FunctionSchema schema = *op.schema;
    library->def(std::move(schema));
    op.definition->reference_registration()(*library, name.c_str());
    library->impl(name.c_str(),
                  torch::dispatch(c10::DispatchKey::PrivateUse1,
                                  torch::CppFunction::makeFromBoxedFunction<&run_on_device>()));
    op.handle = std::make_unique<c10::OperatorHandle>(
        c10::Dispatcher::singleton().findSchemaOrThrow(op.qualified.c_str(), ""));
  } catch (const c10::Error& error) {
    return invalid_argument(error.what_without_backtrace());
  }
  libraries().push_back(std::move(library));
  return ok();
}

std::vector<ir::DType> ir_dtypes(const RegisteredOp& op) {
  std::vector<ir::DType> dtypes;
  for (const at::ScalarType dtype : op.definition->element_types()) {
    const Result<ir::DType> type = to_ir_dtype(dtype);
    if (type.has_value()) {
      dtypes.push_back(*type);
    }
  }
  return dtypes;
}

/// Puts `op` in place of the torch operation it replaces: as that operation's implementation on
/// tessiq's devices, and as a pass that swaps it in a graph.
Result<void> register_replacement(RegisteredOp& op) {
  if (const auto found = replacements().find(op.replaced); found != replacements().end()) {
    return invalid_argument(found->second->qualified + " already replaces " + op.replaced);
  }
  const c10::OperatorName name = operator_name_of(op.replaced);
  const std::size_t separator = name.name.find("::");
  std::string unqualified = name.name.substr(separator + 2);
  if (!name.overload_name.empty()) {
    unqualified += "." + name.overload_name;
  }
  auto library = std::make_unique<torch::Library>(torch::Library::IMPL,
                                                  name.name.substr(0, separator),
                                                  c10::DispatchKey::PrivateUse1,
                                                  __FILE__,
                                                  __LINE__);
  try {
    library->impl(unqualified.c_str(), torch::CppFunction::makeFromBoxedFunction<&run_on_device>());
  } catch (const c10::Error& error) {
    return invalid_argument(error.what_without_backtrace());
  }
  libraries().push_back(std::move(library));

  std::string pass = "override-" + hyphenated(op.definition->name());
  passes::PassRegistry::instance().add(
      pass, [pass, from = op.replaced_spelling, to = op.spelling, dtypes = ir_dtypes(op)] {
        return std::make_unique<passes::ReplaceOp>(pass, from, to, dtypes);
      });
  replacing_passes().push_back(std::move(pass));
  replacements().emplace(op.replaced, &op);
  return ok();
}

/// Checks `op`'s pattern against its schema and registers the pass that rewrites to it.
Result<void> register_rewrite(const RegisteredOp& op) {
  const std::string key = op.definition->directory() + "/" + op.definition->pattern_file();
  const std::optional<std::string_view> text = custom_ops::find_embedded_file(key);
  if (!text.has_value()) {
    return invalid_argument("there is no pattern file csrc/custom_ops/" + key);
  }
  Result<std::unique_ptr<ir::Graph>> parsed = ir::parse(*text);
  if (!parsed.has_value()) {
    return parsed.error().with_context("its pattern csrc/custom_ops/" + key + " does not parse");
  }
  const std::shared_ptr<const ir::Graph> pattern = std::move(*parsed);
  const Result<passes::PatternShape> shape = passes::check_pattern(*pattern);
  if (!shape.has_value()) {
    return shape.error().with_context("its pattern csrc/custom_ops/" + key);
  }

  std::vector<passes::RewriteArgument> arguments;
  std::vector<std::string> tensors;
  std::vector<std::string> scalars;
  for (std::size_t i = 0; i < op.parameters.size(); ++i) {
    const bool tensor = op.parameters[i] == Parameter::kTensor;
    arguments.push_back({.name = op.names[i], .tensor = tensor});
    (tensor ? tensors : scalars).push_back(op.names[i]);
  }
  const auto same = [](std::vector<std::string> first, std::vector<std::string> second) {
    std::ranges::sort(first);
    std::ranges::sort(second);
    return first == second;
  };
  // Each argument of the call the pattern becomes has to come from somewhere in the match.
  if (!same(shape->inputs, tensors)) {
    return invalid_argument("its pattern csrc/custom_ops/" + key + " has the inputs " +
                            listed(shape->inputs) + ", and its tensor arguments are " +
                            listed(tensors) + "; they have to be the same names");
  }
  if (!same(shape->captures, scalars)) {
    return invalid_argument("its pattern csrc/custom_ops/" + key + " captures " +
                            listed(shape->captures) + ", and its scalar arguments are " +
                            listed(scalars) + "; capture each one as \"$name\"");
  }

  std::string pass = "rewrite-" + hyphenated(op.definition->name());
  passes::PassRegistry::instance().add(
      pass, [pass, pattern, to = op.spelling, arguments, dtypes = ir_dtypes(op)] {
        return std::make_unique<passes::Rewrite>(pass, pattern, to, arguments, dtypes);
      });
  rewrite_passes().emplace(op.qualified, std::move(pass));
  return ok();
}

/// How `op` runs, for the log.
std::string how_it_runs(const RegisteredOp& op) {
  if (op.precompiled != nullptr) {
    return "with host code built for " + codegen::architecture_names(op.precompiled->architectures);
  }
  if (!op.runs_on_gpu()) {
    return "with no host code in this build, so it runs its reference";
  }
  return op.has_kernel() ? "with a kernel" : "as an expression";
}

Result<void> register_one(const CustomOp& definition) {
  TESSIQ_ASSIGN_OR_RETURN(std::unique_ptr<RegisteredOp> op, check_definition(definition));
  if (find_qualified(op->qualified) != nullptr) {
    return invalid_argument("it is defined twice");
  }
  // Host code this build did not compile leaves nothing for the compiler to claim, so the
  // operation stays on the host, where its reference runs.
  if (op->runs_on_gpu()) {
    TESSIQ_RETURN_IF_ERROR(register_with_compiler(*op));
  }
  TESSIQ_RETURN_IF_ERROR(register_with_torch(*op));
  if (!op->replaced.empty()) {
    TESSIQ_RETURN_IF_ERROR(register_replacement(*op));
  }
  if (!definition.pattern_file().empty()) {
    TESSIQ_RETURN_IF_ERROR(register_rewrite(*op));
  }
  TESSIQ_LOG(kDebug, kLogModule) << "registered " << op->qualified << " " << how_it_runs(*op);
  const std::string qualified = op->qualified;
  registered_ops().emplace(qualified, std::move(op));
  return ok();
}

/// Scalar argument `argument` typed as the schema declares its parameter, or nothing when it
/// cannot be.
std::optional<c10::IValue> scalar_as(Parameter parameter, const codegen::CallArgument& argument) {
  if (const auto* integer = std::get_if<std::int64_t>(&argument)) {
    if (parameter == Parameter::kInt) {
      return c10::IValue(*integer);
    }
    if (parameter == Parameter::kFloat) {
      return c10::IValue(static_cast<double>(*integer));
    }
  } else if (const auto* real = std::get_if<double>(&argument)) {
    if (parameter == Parameter::kFloat) {
      return c10::IValue(*real);
    }
  } else if (const auto* flag = std::get_if<bool>(&argument)) {
    if (parameter == Parameter::kBool) {
      return c10::IValue(*flag);
    }
  }
  return std::nullopt;
}

/// The call's scalars typed as the schema declares them, or nothing when the call does not match
/// the schema.
std::optional<std::vector<std::optional<c10::IValue>>> scalars_of(const RegisteredOp& op,
                                                                  const codegen::CustomCall& call) {
  if (call.arguments.size() != op.parameters.size()) {
    return std::nullopt;
  }
  std::vector<std::optional<c10::IValue>> scalars;
  scalars.reserve(call.arguments.size());
  for (std::size_t i = 0; i < call.arguments.size(); ++i) {
    const bool is_tensor = std::holds_alternative<codegen::CallOperand>(call.arguments[i]);
    if (is_tensor != (op.parameters[i] == Parameter::kTensor)) {
      return std::nullopt;
    }
    if (is_tensor) {
      scalars.emplace_back(std::nullopt);
      continue;
    }
    std::optional<c10::IValue> scalar = scalar_as(op.parameters[i], call.arguments[i]);
    if (!scalar.has_value()) {
      return std::nullopt;
    }
    scalars.emplace_back(std::move(scalar));
  }
  return scalars;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Compiling an elementwise operation on its own

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

// ---------------------------------------------------------------------------------------------
// The registry's public face

Result<void> register_custom_ops() {
  for (const CustomOp* definition : custom_ops::definitions()) {
    const Result<void> registered = register_one(*definition);
    if (!registered.has_value()) {
      return registered.error().with_context("custom operation " + std::string(kNamespace) +
                                             "::" + definition->name() + " in csrc/custom_ops/" +
                                             definition->directory());
    }
  }

  // Host code nothing declares would never run, and two files for one operation would leave which
  // one runs to the order they load in.
  std::set<std::string_view> seen;
  for (const codegen::Precompiled* entry : codegen::precompiled_operations()) {
    const std::string qualified = std::string(kNamespace) + "::" + std::string(entry->name);
    if (!seen.insert(entry->name).second) {
      return invalid_argument("two .nvcc.cu files register host code for " + qualified);
    }
    const RegisteredOp* op = find_qualified(qualified);
    if (op == nullptr || !op->definition->is_precompiled()) {
      return invalid_argument("a .nvcc.cu file registers host code for " + qualified +
                              ", which no .cpp defines with .precompiled()");
    }
  }
  return ok();
}

std::vector<std::string> precompiled_op_names() {
  std::vector<std::string> names;
  for (const auto& [qualified, op] : registered_ops()) {
    if (op->precompiled != nullptr) {
      names.push_back(qualified);
    }
  }
  return names;
}

std::vector<std::string> custom_op_names() {
  std::vector<std::string> names;
  for (const auto& [qualified, op] : registered_ops()) {
    names.push_back(qualified);
  }
  return names;
}

std::vector<CustomOverride> custom_op_overrides() {
  std::vector<CustomOverride> overrides;
  for (const auto& [replaced, op] : replacements()) {
    overrides.push_back({.op = op->qualified,
                         .replaced = op->replaced_spelling,
                         .dtypes = op->definition->element_types()});
  }
  return overrides;
}

std::vector<std::pair<std::string, std::string>> custom_op_rewrites() {
  return {rewrite_passes().begin(), rewrite_passes().end()};
}

std::vector<std::string> override_passes() {
  std::vector<std::string> names = replacing_passes();
  std::ranges::sort(names);
  return names;
}

Result<std::int64_t> custom_op_fallbacks(std::string_view name) {
  const RegisteredOp* op = find_qualified(name);
  if (op == nullptr) {
    return not_found(std::string(name) + " is not a custom operation");
  }
  return op->fallbacks.load(std::memory_order_relaxed);
}

Result<void> compile_custom_op(std::string_view name, int ordinal) {
  RegisteredOp* op = find_qualified(name);
  if (op == nullptr) {
    return not_found(std::string(name) + " is not a custom operation");
  }
  if (op->has_kernel()) {
    return ready_on(*op, ordinal);
  }
  if (!op->runs_on_gpu()) {
    return ok();
  }
  for (const at::ScalarType dtype : op->definition->element_types()) {
    TESSIQ_RETURN_IF_ERROR(expression_for(*op, dtype, ordinal));
  }
  return ok();
}

CustomKernel::CustomKernel(codegen::CustomCall call,
                           RegisteredOp* op,
                           std::vector<std::optional<c10::IValue>> scalars,
                           int ordinal)
    : call_(std::move(call)), op_(op), scalars_(std::move(scalars)), ordinal_(ordinal) {}

const std::string& CustomKernel::op() const { return op_->qualified; }

std::vector<std::string> CustomKernel::members() const {
  std::vector<std::string> members = call_.absorbed;
  members.push_back(call_.output);
  return members;
}

std::vector<c10::IValue> CustomKernel::arguments(const std::vector<at::Tensor>& tensors) const {
  std::vector<c10::IValue> arguments;
  arguments.reserve(scalars_.size());
  std::size_t next = 0;
  for (const std::optional<c10::IValue>& scalar : scalars_) {
    if (scalar.has_value()) {
      arguments.push_back(*scalar);
    } else {
      arguments.emplace_back(tensors.at(next++));
    }
  }
  return arguments;
}

Result<std::vector<std::int64_t>>
CustomKernel::output_shape(const std::vector<at::Tensor>& tensors) const {
  try {
    const at::Tensor like = result_like(*op_, arguments(tensors));
    return std::vector<std::int64_t>(like.sizes().begin(), like.sizes().end());
  } catch (const c10::Error& error) {
    return invalid_argument(op_->qualified + ": " + error.what_without_backtrace());
  }
}

Result<void> CustomKernel::launch(const std::vector<at::Tensor>& tensors,
                                  const at::Tensor& out,
                                  codegen::Stream stream) const {
  const std::vector<c10::IValue> called = arguments(tensors);
  TESSIQ_ASSIGN_OR_RETURN(const bool launched, launch_kernel(*op_, called, out, ordinal_, stream));
  if (launched) {
    return ok();
  }
  return reference_into(*op_, called, out);
}

Result<std::vector<std::shared_ptr<Kernel>>> custom_kernels(const ir::Graph& graph, int ordinal) {
  std::vector<std::shared_ptr<Kernel>> kernels;
  for (codegen::CustomCall& call : codegen::custom_calls(graph)) {
    RegisteredOp* op = find_spelled(call.op);
    if (op == nullptr || !op->has_kernel()) {
      continue;
    }
    std::optional<std::vector<std::optional<c10::IValue>>> scalars = scalars_of(*op, call);
    if (!scalars.has_value()) {
      TESSIQ_LOG(kDebug, kLogModule) << call.output << " calls " << op->qualified
                                     << " with arguments its schema does not declare";
      continue;
    }
    TESSIQ_RETURN_IF_ERROR(ready_on(*op, ordinal));
    kernels.push_back(
        std::make_shared<CustomKernel>(std::move(call), op, std::move(*scalars), ordinal));
  }
  return kernels;
}

} // namespace tessiq::torch_backend
