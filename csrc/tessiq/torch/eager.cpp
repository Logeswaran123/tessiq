// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Elementwise operations called outside a compiled graph.
//
// Each call on GPU tensors runs one kernel from the compiler's own emitter, compiled once per
// operation, element type and the way each operand is read. Scalars are kernel arguments, so
// `x * 0.5` and `x * 0.25` share a kernel. Anything else goes through the CPU fallback.

#include "tessiq/torch/eager.h"

#include <ATen/ExpandUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <algorithm>
#include <array>
#include <c10/core/DispatchKey.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/library.h>

#include "tessiq/codegen/cubin_cache.h"
#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/kernel.h"
#include "tessiq/codegen/program.h"
#include "tessiq/codegen/stream.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_table.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/backend.h"
#include "tessiq/torch/fallback.h"
#include "tessiq/torch/importer.h"
#include "tessiq/torch/pack.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "eager";

/// The operations whose expression is exact on integers. Every other one goes through a
/// floating-point function such as `fabs` or `floor`, which rounds an int64 beyond 2^53, or is one
/// torch computes in floating point for an integer tensor.
constexpr std::array<std::string_view, 5> kIntegerOps = {"add", "sub", "mul", "neg", "relu"};

/// How the kernel reaches one operand.
enum class Read : char {
  /// A tensor laid out as the result is, read at `i`.
  kSame = 's',
  /// A tensor each of whose elements covers a run of the result, read at `i / d`.
  kRepeated = 'r',
  /// A tensor read over and over, at `i % d`.
  kCycled = 'c',
  /// A number, passed by value.
  kScalar = 'k',
};

/// Everything a launch needs, worked out from one call's arguments.
struct Plan {
  at::Tensor out;
  /// One per operand, in schema order.
  std::vector<Read> reads;
  /// The tensors the kernel reads, in operand order, each laid out as its read says.
  std::vector<at::Tensor> tensors;
  /// The divisor for each tensor not read at `i`, in the same order.
  std::vector<std::int64_t> divisors;
  /// The value of each scalar operand, in operand order.
  std::vector<c10::Scalar> scalars;
};

/// The operation each registered spelling is, filled once during registration and only read after.
std::unordered_map<c10::OperatorName, const passes::OpInfo*>& eager_ops() {
  static std::unordered_map<c10::OperatorName, const passes::OpInfo*> ops;
  return ops;
}

// ---------------------------------------------------------------------------------------------
// Deciding whether a call runs here

/// Whether the arguments after the operands are all their defaults, since the kernel reads none
/// of them: `alpha` on add and sub, `approximate` on gelu.
bool only_defaults(const c10::FunctionSchema& schema,
                   const std::vector<c10::IValue>& arguments,
                   int arity) {
  for (auto i = static_cast<std::size_t>(arity); i < arguments.size(); ++i) {
    const std::optional<c10::IValue>& fallback = schema.arguments()[i].default_value();
    if (!fallback.has_value() || !(arguments[i] == *fallback)) {
      return false;
    }
  }
  return true;
}

/// Whether `dtype` is one the kernel computes `info` in exactly as torch does.
bool computes_in(const passes::OpInfo& info, at::ScalarType dtype) {
  const Result<ir::DType> type = to_ir_dtype(dtype);
  if (!type.has_value() || !info.accepts(*type)) {
    return false;
  }
  switch (dtype) {
  case at::kFloat:
  case at::kDouble:
    return true;
  case at::kInt:
  case at::kLong:
    return std::ranges::find(kIntegerOps, info.canonical) != kIntegerOps.end();
  default:
    return false;
  }
}

/// One call's operands, split into the tensors on the GPU and the numbers.
struct Operands {
  /// One per operand, in schema order, empty for a number.
  std::vector<std::optional<at::Tensor>> tensors;
  /// The numbers, in schema order.
  std::vector<c10::Scalar> scalars;
  /// The element type of every tensor, and of the result.
  at::ScalarType dtype = at::kFloat;
  int ordinal = 0;
};

/// The operands as the kernel takes them, or nothing when the call has to run on the host.
///
/// Every tensor has to be on one GPU in one element type, which is also the type torch gives the
/// result. A number, or a 0-dim tensor on the host such as the 2 in `x + 2`, becomes a scalar.
std::optional<Operands> operands_of(const passes::OpInfo& info,
                                    const std::vector<c10::IValue>& arguments) {
  Operands operands;
  std::optional<c10::Device> device;
  at::native::ResultTypeState promotion;
  for (int i = 0; i < info.arity; ++i) {
    const c10::IValue& argument = arguments[static_cast<std::size_t>(i)];
    if (argument.isTensor()) {
      const at::Tensor& tensor = argument.toTensor();
      promotion = at::native::update_result_type_state(tensor, promotion);
      if (tensor.device().is_cpu() && tensor.dim() == 0) {
        operands.tensors.emplace_back(std::nullopt);
        operands.scalars.push_back(tensor.item());
        continue;
      }
      if (device.has_value() &&
          (tensor.device() != *device || tensor.scalar_type() != operands.dtype)) {
        return std::nullopt;
      }
      device = tensor.device();
      operands.dtype = tensor.scalar_type();
      operands.tensors.emplace_back(tensor);
    } else if (argument.isScalar()) {
      const c10::Scalar scalar = argument.toScalar();
      promotion = at::native::update_result_type_state(scalar, promotion);
      operands.tensors.emplace_back(std::nullopt);
      operands.scalars.push_back(scalar);
    } else {
      return std::nullopt;
    }
  }
  if (!device.has_value() || at::native::result_type(promotion) != operands.dtype ||
      !computes_in(info, operands.dtype)) {
    return std::nullopt;
  }
  const std::optional<int> ordinal = cuda_ordinal_of(*device);
  if (!ordinal.has_value()) {
    return std::nullopt;
  }
  operands.ordinal = *ordinal;
  return operands;
}

// ---------------------------------------------------------------------------------------------
// Laying out the result and its operands

/// Whether `sizes` and `strides` describe a packed tensor. Strides of size-1 dimensions do not
/// matter, as in torch.
bool packed(const std::vector<std::int64_t>& sizes, const std::vector<std::int64_t>& strides) {
  std::int64_t expected = 1;
  for (std::size_t k = sizes.size(); k-- > 0;) {
    if (sizes[k] == 1) {
      continue;
    }
    if (strides[k] != expected) {
      return false;
    }
    expected *= sizes[k];
  }
  return true;
}

/// The result torch would give, as a tensor with the right sizes and strides on the device, or
/// nothing when its layout is not one the kernel can write.
///
/// Operands that are all packed give a packed result, and operands that share one dense layout
/// give that layout. Anything else asks torch's own meta kernel, which is what torch would do.
std::optional<at::Tensor> result_for(const c10::OperatorHandle& handle,
                                     const std::vector<c10::IValue>& arguments,
                                     const Operands& operands) {
  std::vector<at::Tensor> tensors;
  for (const std::optional<at::Tensor>& tensor : operands.tensors) {
    if (tensor.has_value()) {
      tensors.push_back(*tensor);
    }
  }
  const at::TensorOptions options = tensors.front().options();

  const at::Tensor& first = tensors.front();
  const bool one_layout =
      first.is_non_overlapping_and_dense() &&
      std::ranges::all_of(tensors, [&first](const at::Tensor& tensor) {
        return tensor.sizes() == first.sizes() && tensor.strides() == first.strides();
      });
  if (one_layout) {
    return at::empty_strided(first.sizes(), first.strides(), options);
  }

  at::DimVector shape;
  try {
    for (const at::Tensor& tensor : tensors) {
      shape = shape.empty() ? at::DimVector(tensor.sizes())
                            : at::infer_size_dimvector(shape, tensor.sizes());
    }
  } catch (const c10::Error&) {
    // The fallback raises torch's own message.
    return std::nullopt;
  }
  if (std::ranges::all_of(tensors,
                          [](const at::Tensor& tensor) { return tensor.is_contiguous(); })) {
    return at::empty(shape, options);
  }

  torch::jit::Stack stack;
  stack.reserve(arguments.size());
  for (const c10::IValue& argument : arguments) {
    if (argument.isTensor() && !argument.toTensor().device().is_cpu()) {
      const at::Tensor& tensor = argument.toTensor();
      stack.emplace_back(
          at::empty_strided(tensor.sizes(), tensor.strides(), tensor.options().device(c10::kMeta)));
    } else {
      stack.push_back(argument);
    }
  }
  try {
    handle.callBoxed(stack);
  } catch (const c10::Error&) {
    return std::nullopt;
  }
  const at::Tensor like = stack.back().toTensor();
  if (!like.is_non_overlapping_and_dense()) {
    return std::nullopt;
  }
  return at::empty_strided(like.sizes(), like.strides(), options);
}

/// The order of `out`'s dimensions from outermost in memory to innermost. Walking the result in
/// that order is walking its memory.
std::vector<std::size_t> memory_order(const at::Tensor& out) {
  std::vector<std::size_t> order(static_cast<std::size_t>(out.dim()));
  std::iota(order.begin(), order.end(), std::size_t{0});
  if (!out.is_contiguous()) {
    const c10::IntArrayRef strides = out.strides();
    std::ranges::stable_sort(
        order, [&strides](std::size_t a, std::size_t b) { return strides[a] > strides[b]; });
  }
  return order;
}

/// `values` right-aligned against `order.size()` dimensions, as broadcasting aligns them, padded
/// with `pad`, and put in `order`.
std::vector<std::int64_t>
permuted(c10::IntArrayRef values, std::int64_t pad, const std::vector<std::size_t>& order) {
  std::vector<std::int64_t> aligned(order.size() - values.size(), pad);
  aligned.insert(aligned.end(), values.begin(), values.end());
  std::vector<std::int64_t> result(order.size());
  for (std::size_t k = 0; k < order.size(); ++k) {
    result[k] = aligned[order[k]];
  }
  return result;
}

/// How the kernel reads an operand of `sizes` and `strides` where it lies, for a result of `shape`
/// walked in memory order, with the divisor that read takes. Nothing when it has to be packed.
std::optional<std::pair<Read, std::int64_t>> read_in_place(const std::vector<std::int64_t>& sizes,
                                                           const std::vector<std::int64_t>& strides,
                                                           const std::vector<std::int64_t>& shape) {
  const std::int64_t elements = c10::multiply_integers(sizes);
  if (elements != 1 && !packed(sizes, strides)) {
    return std::nullopt;
  }
  if (sizes == shape) {
    return std::pair{Read::kSame, std::int64_t{1}};
  }
  std::size_t lead = 0;
  while (lead < sizes.size() && sizes[lead] == 1) {
    ++lead;
  }
  std::size_t kept = sizes.size();
  while (kept > 0 && sizes[kept - 1] == 1) {
    --kept;
  }
  const std::span<const std::int64_t> own(sizes);
  const std::span<const std::int64_t> result(shape);
  // Leading ones then the result's own extents: a bias row, read over and over.
  if (std::ranges::equal(own.subspan(lead), result.subspan(lead))) {
    return std::pair{Read::kCycled, elements};
  }
  // The result's own extents then trailing ones: a per-row value, each covering a run.
  if (std::ranges::equal(own.first(kept), result.first(kept))) {
    return std::pair{Read::kRepeated, c10::multiply_integers(shape) / elements};
  }
  return std::nullopt;
}

/// Lays out every tensor operand for a kernel walking `out` in memory order, packing any it
/// cannot read where it is. Nothing when a pack fails.
std::optional<Plan> plan_for(const Operands& operands, at::Tensor out) {
  const std::vector<std::size_t> order = memory_order(out);
  const std::vector<std::int64_t> shape = permuted(out.sizes(), 1, order);
  if (!packed(shape, permuted(out.strides(), 0, order))) {
    return std::nullopt;
  }

  Plan plan;
  plan.scalars = operands.scalars;
  for (const std::optional<at::Tensor>& operand : operands.tensors) {
    if (!operand.has_value()) {
      plan.reads.push_back(Read::kScalar);
      continue;
    }
    const auto read = read_in_place(
        permuted(operand->sizes(), 1, order), permuted(operand->strides(), 0, order), shape);
    if (read.has_value()) {
      plan.reads.push_back(read->first);
      plan.tensors.push_back(*operand);
      if (read->first != Read::kSame) {
        plan.divisors.push_back(read->second);
      }
      continue;
    }
    // A transposed operand, or a broadcast that is neither a run nor a cycle. Packing it in the
    // result's memory order costs one more pass over the data, on the device.
    const std::vector<std::int64_t> dims(order.begin(), order.end());
    at::Tensor buffer = at::empty(shape, out.options());
    const Result<void> copied = pack_on_device(operand->expand(out.sizes()).permute(dims), buffer);
    if (!copied.has_value()) {
      TESSIQ_LOG(kDebug, kLogModule) << "cannot pack an operand: " << copied.error().message();
      return std::nullopt;
    }
    plan.reads.push_back(Read::kSame);
    plan.tensors.push_back(std::move(buffer));
  }
  plan.out = std::move(out);
  return plan;
}

// ---------------------------------------------------------------------------------------------
// Compiling and launching

/// `sk` for a tensor read at `i` and a scalar: the part of a kernel's name and key that varies.
std::string signature(const std::vector<Read>& reads) {
  std::string text;
  for (const Read read : reads) {
    text += static_cast<char>(read);
  }
  return text;
}

/// One compiled kernel for each operation, element type, operand reads and CUDA device.
using KernelKey = std::tuple<std::string_view, at::ScalarType, std::string, int>;

struct KernelCache {
  std::mutex mutex;
  /// Null for a kernel that did not compile, so it is not tried again. Guarded by `mutex`.
  std::map<KernelKey, const codegen::Program*> programs;
};

KernelCache& kernel_cache() {
  static KernelCache cache;
  return cache;
}

/// The kernel for one call, as the emitter writes a fusion group of one operation.
codegen::FusedKernel
describe(const passes::OpInfo& info, ir::DType dtype, const std::vector<Read>& reads) {
  codegen::FusedKernel kernel;
  kernel.name = "tessiq_eager_" + std::string(info.canonical) + "_" +
                std::string(ir::dtype_name(dtype)) + "_" + signature(reads);
  kernel.dtype = dtype;
  kernel.elements = ir::kDynamic;
  kernel.output = "y";

  codegen::Step step;
  step.op = std::string(info.name);
  step.result = "y";
  for (std::size_t i = 0; i < reads.size(); ++i) {
    const std::string name = "x" + std::to_string(i);
    if (reads[i] == Read::kScalar) {
      kernel.scalars.push_back(name);
      step.args.push_back({.text = name, .is_value = false});
      continue;
    }
    codegen::KernelInput input;
    input.name = name;
    input.type = ir::TensorType(dtype, ir::Shape({ir::kDynamic}));
    if (reads[i] != Read::kSame) {
      input.index.clear();
      input.read.kind = reads[i] == Read::kCycled ? codegen::OperandRead::Kind::kCycled
                                                  : codegen::OperandRead::Kind::kRepeated;
      input.read.divisor = ir::kDynamic;
    }
    kernel.inputs.push_back(std::move(input));
    step.args.push_back({.text = name, .is_value = true});
  }
  kernel.body.push_back(std::move(step));
  return kernel;
}

Result<const codegen::Program*>
compile(const passes::OpInfo& info, ir::DType dtype, const std::vector<Read>& reads, int ordinal) {
  const codegen::FusedKernel kernel = describe(info, dtype, reads);
  const codegen::Emitter& emitter = codegen::cuda_emitter();
  if (!emitter.supports(kernel)) {
    return unsupported("no cuda code for " + kernel.name);
  }
  TESSIQ_ASSIGN_OR_RETURN(const std::string source, emitter.emit(kernel));
  TESSIQ_ASSIGN_OR_RETURN(const auto capability, codegen::device_capability(ordinal));
  TESSIQ_ASSIGN_OR_RETURN(
      const codegen::Cubin cubin,
      codegen::compile_cached(source, kernel.name, capability.first, capability.second));
  return codegen::load_program(cubin, kernel.name, ordinal);
}

/// The compiled kernel for a call, or null when there is none and the call runs on the host.
///
/// `computes_in` has already checked that the element type is one the compiler has.
const codegen::Program*
program_for(const passes::OpInfo& info, const Operands& operands, const Plan& plan) {
  KernelKey key{info.canonical, operands.dtype, signature(plan.reads), operands.ordinal};
  KernelCache& cache = kernel_cache();
  const std::scoped_lock guard(cache.mutex);
  if (const auto found = cache.programs.find(key); found != cache.programs.end()) {
    return found->second;
  }
  const Result<const codegen::Program*> compiled =
      compile(info, *to_ir_dtype(operands.dtype), plan.reads, operands.ordinal);
  const codegen::Program* program = compiled.has_value() ? *compiled : nullptr;
  if (!compiled.has_value()) {
    TESSIQ_LOG(kDebug, kLogModule)
        << info.canonical << " has no eager kernel: " << compiled.error().message();
  }
  cache.programs.emplace(std::move(key), program);
  return program;
}

/// `scalar` as the kernel's element type, converted as torch converts it.
codegen::KernelArgument scalar_argument(const c10::Scalar& scalar, at::ScalarType dtype) {
  switch (dtype) {
  case at::kFloat:
    return static_cast<float>(scalar.toDouble());
  case at::kInt:
    return static_cast<std::int32_t>(scalar.toLong());
  case at::kLong:
    return scalar.toLong();
  default:
    return scalar.toDouble();
  }
}

Result<void> launch(const codegen::Program& program, const Plan& plan, at::ScalarType dtype) {
  const std::int64_t count = plan.out.numel();
  std::vector<void*> buffers;
  buffers.reserve(plan.tensors.size() + 1);
  for (const at::Tensor& tensor : plan.tensors) {
    buffers.push_back(tensor.data_ptr());
  }
  buffers.push_back(plan.out.data_ptr());

  std::vector<codegen::KernelArgument> arguments{count};
  arguments.insert(arguments.end(), plan.divisors.begin(), plan.divisors.end());
  for (const c10::Scalar& scalar : plan.scalars) {
    arguments.push_back(scalar_argument(scalar, dtype));
  }

  const std::int64_t blocks =
      (count + static_cast<std::int64_t>(codegen::kBlockSize) - 1) / codegen::kBlockSize;
  codegen::LaunchShape shape;
  shape.grid[0] = static_cast<std::uint32_t>(blocks);
  shape.block[0] = codegen::kBlockSize;
  return program.launch(buffers, arguments, shape, codegen::kDefaultStream);
}

/// What an elementwise operation from ops.toml does when it is called on a tessiq tensor.
void run_elementwise(const c10::OperatorHandle& handle, torch::jit::Stack* stack) {
  const auto found = eager_ops().find(handle.operator_name());
  TORCH_CHECK(found != eager_ops().end(), handle.operator_name(), " has no tessiq eager kernel");
  const passes::OpInfo& info = *found->second;

  const std::size_t count = handle.schema().arguments().size();
  const std::vector<c10::IValue> arguments(stack->end() - static_cast<std::ptrdiff_t>(count),
                                           stack->end());

  const std::optional<Operands> operands = only_defaults(handle.schema(), arguments, info.arity)
                                               ? operands_of(info, arguments)
                                               : std::nullopt;
  std::optional<at::Tensor> out =
      operands.has_value() ? result_for(handle, arguments, *operands) : std::nullopt;
  if (out.has_value() && out->numel() == 0) {
    torch::jit::drop(*stack, count);
    torch::jit::push(*stack, std::move(*out));
    return;
  }
  // One thread per element, and a grid holds at most 2^32 - 1 blocks.
  constexpr std::int64_t kMaxElements = ((std::int64_t{1} << 32) - 1) * codegen::kBlockSize;
  const std::optional<Plan> plan = out.has_value() && out->numel() <= kMaxElements
                                       ? plan_for(*operands, std::move(*out))
                                       : std::nullopt;
  const codegen::Program* program =
      plan.has_value() ? program_for(info, *operands, *plan) : nullptr;
  if (!operands.has_value() || !plan.has_value() || program == nullptr) {
    cpu_fallback(handle, stack);
    return;
  }
  const Result<void> launched = launch(*program, *plan, operands->dtype);
  TORCH_CHECK(launched.has_value(), launched.has_value() ? "" : launched.error().to_string());
  torch::jit::drop(*stack, count);
  torch::jit::push(*stack, plan->out);
}

/// `aten.add.Tensor` as torch keys it: `aten::add` with overload `Tensor`, and no overload for
/// `.default`.
c10::OperatorName operator_name_of(std::string_view spelling) {
  const std::size_t first = spelling.find('.');
  const std::size_t last = spelling.rfind('.');
  std::string name = std::string(spelling.substr(0, first)) +
                     "::" + std::string(spelling.substr(first + 1, last - first - 1));
  std::string overload(spelling.substr(last + 1));
  return {std::move(name), overload == "default" ? "" : std::move(overload)};
}

} // namespace

Result<void> register_eager_ops() {
  static std::unique_ptr<torch::Library> library;
  if (library != nullptr) {
    return ok();
  }
  library = std::make_unique<torch::Library>(
      torch::Library::IMPL, "aten", c10::DispatchKey::PrivateUse1, __FILE__, __LINE__);

  for (const std::string& spelling : passes::op_names()) {
    const passes::OpInfo* info = passes::find_builtin_op(spelling);
    if (info->category != passes::OpCategory::kElementwise || !info->has_kernel ||
        !spelling.starts_with("aten.")) {
      continue;
    }
    c10::OperatorName name = operator_name_of(spelling);
    const std::optional<c10::OperatorHandle> handle = c10::Dispatcher::singleton().findSchema(name);
    if (!handle.has_value()) {
      return invalid_argument("ops.toml lists " + spelling + ", which torch does not have");
    }
    // A custom operation that replaces it has already registered itself, and wins.
    if (handle->hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1)) {
      TESSIQ_LOG(kDebug, kLogModule) << spelling << " is already implemented, leaving it alone";
      continue;
    }
    std::string unqualified = name.name.substr(name.name.find("::") + 2);
    if (!name.overload_name.empty()) {
      unqualified += "." + name.overload_name;
    }
    try {
      library->impl(unqualified.c_str(),
                    torch::CppFunction::makeFromBoxedFunction<&run_elementwise>());
    } catch (const c10::Error& error) {
      return invalid_argument(error.what_without_backtrace());
    }
    eager_ops().emplace(std::move(name), info);
  }
  return ok();
}

} // namespace tessiq::torch_backend
