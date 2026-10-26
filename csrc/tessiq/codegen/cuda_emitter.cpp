// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <string>
#include <string_view>

#include "tessiq/codegen/emitter.h"
#include "tessiq/codegen/program.h"
#include "tessiq/ir/types.h"
#include "tessiq/passes/op_traits.h"

namespace tessiq::codegen {

namespace {

/// The C spelling of an element type.
std::string_view c_type(ir::DType dtype) {
  switch (dtype) {
  case ir::DType::kF32:
    return "float";
  case ir::DType::kF64:
    return "double";
  case ir::DType::kI32:
    return "int";
  case ir::DType::kI64:
    return "long long";
  case ir::DType::kF16:
  case ir::DType::kBF16:
  case ir::DType::kI1:
    return {};
  }
  return {};
}

/// Whether the type has NaNs to worry about, which decides how max and min are written.
bool is_floating(ir::DType dtype) { return dtype == ir::DType::kF32 || dtype == ir::DType::kF64; }

/// What a thread's accumulator starts at, as an expression the kernel body can use.
///
/// A sum starts at zero. Max and min start at the slice's first element rather than a per-type
/// lowest value, since `-INFINITY` needs math.h, which NVRTC does not include. Every thread can
/// read that element, max and min are idempotent, and a NaN there still propagates.
std::string_view accumulator_start(ReduceOp op) {
  return op == ReduceOp::kSum || op == ReduceOp::kMean ? "(scalar_t)0" : "base[0]";
}

/// How two values combine, as an expression in `a` and `b`.
///
/// max and min test for NaN first, because torch.amax propagates it. `fmax` drops a NaN, and
/// `a > b ? a : b` keeps one only when it is `b`.
std::string combine(ReduceOp op, ir::DType dtype) {
  switch (op) {
  case ReduceOp::kSum:
  case ReduceOp::kMean:
    return "(a) + (b)";
  case ReduceOp::kMax:
    return is_floating(dtype) ? "(isnan(a) || (a) > (b)) ? (a) : (b)" : "(a) > (b) ? (a) : (b)";
  case ReduceOp::kMin:
    return is_floating(dtype) ? "(isnan(a) || (a) < (b)) ? (a) : (b)" : "(a) < (b) ? (a) : (b)";
  }
  return {};
}

/// Substitutes `$0` and `$1` in an expression with the arguments.
///
/// A value argument becomes the variable holding it, named after the IR value. A constant is
/// written at the element type.
std::string substitute(std::string_view expression, const std::vector<Argument>& args) {
  std::string out;
  for (std::size_t i = 0; i < expression.size(); ++i) {
    if (expression[i] != '$' || i + 1 >= expression.size()) {
      out += expression[i];
      continue;
    }
    const auto index = static_cast<std::size_t>(expression[i + 1] - '0');
    if (index >= args.size()) {
      // The table and the extractor disagree about arity, which the extractor already rejects.
      // Emitting the marker unchanged makes that visible rather than producing plausible code.
      out += expression[i];
      continue;
    }
    // Cast rather than written bare: `v * 0.1` on a float would be a double multiply narrowed
    // afterwards, which can differ from the float multiply torch does.
    out += args[index].is_value ? "v_" + args[index].text : "scalar_t(" + args[index].text + ")";
    ++i;
  }
  return out;
}

/// Whether this operand's index needs a number the source cannot hold.
bool needs_divisor(const FusedKernel& kernel, const KernelInput& input) {
  return kernel.has_dynamic_extent() && !input.read.is_known();
}

/// The index expression for this operand, with the divisor folded in or named as a parameter.
std::string read_index(const FusedKernel& kernel, const KernelInput& input) {
  if (!needs_divisor(kernel, input)) {
    return input.index;
  }
  switch (input.read.kind) {
  case OperandRead::Kind::kRepeated:
    return "i / d_" + input.name;
  case OperandRead::Kind::kCycled:
    return "i % d_" + input.name;
  case OperandRead::Kind::kSameShape:
  case OperandRead::Kind::kStrided:
    break;
  }
  // Unreachable: `operand_read` gives only those two an unknown divisor, and refuses a strided
  // operand outright when the extents are not known.
  return {};
}

/// Emits CUDA C++ for NVRTC.
///
/// One thread per element with a bounds check, so the grid need not divide the count exactly. The
/// kernel takes packed buffers and an element count, and reads each operand at the index its
/// `KernelInput` gives.
class CudaEmitter final : public Emitter {
 public:
  [[nodiscard]] std::string_view name() const override { return "cuda"; }

  [[nodiscard]] bool supports(const FusedKernel& kernel) const override {
    // A dynamic count is allowed: the count is always the argument `n`, and a divisor not known
    // here becomes a parameter.
    const bool sized = kernel.elements > 0 || kernel.has_dynamic_extent();
    if (c_type(kernel.dtype).empty() || !sized || kernel.body.empty()) {
      return false;
    }
    for (const KernelInput& input : kernel.inputs) {
      // A known kernel has every index folded in, so an empty one means no expression exists.
      if (!needs_divisor(kernel, input) && input.index.empty()) {
        return false;
      }
    }
    for (const Step& step : kernel.body) {
      if (passes::elementwise_expression(step.op).empty()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool supports(const ReductionKernel& kernel) const override {
    // No C spelling for the half and boolean types yet.
    if (c_type(kernel.dtype).empty()) {
      return false;
    }
    // The extents are kernel arguments, so they only have to be positive or not yet known.
    const auto usable = [](std::int64_t extent) { return extent > 0 || extent == ir::kDynamic; };
    if (!usable(kernel.reduce) || !usable(kernel.outer) || !usable(kernel.inner)) {
      return false;
    }
    // torch refuses a mean over integers, so this declines nothing a real graph contains.
    return kernel.op != ReduceOp::kMean || is_floating(kernel.dtype);
  }

  [[nodiscard]] Result<std::string> emit(const ReductionKernel& kernel) const override {
    if (!supports(kernel)) {
      return unsupported("no cuda reduction for " + kernel.name);
    }

    const std::string combiner = combine(kernel.op, kernel.dtype);
    const std::string start = std::string(accumulator_start(kernel.op));
    const std::string mean = kernel.op == ReduceOp::kMean ? " / (scalar_t)reduce" : "";

    std::string source = "// Generated by tessiq. Do not edit.\n";
    source += "typedef " + std::string(c_type(kernel.dtype)) + " scalar_t;\n\n";
    source += "__device__ inline scalar_t tessiq_combine(scalar_t a, scalar_t b) {\n";
    source += "  return " + combiner + ";\n}\n\n";
    source += "extern \"C\" __global__ void " + kernel.name + "(\n";
    source += "    const scalar_t* __restrict__ p_in,\n";
    source += "    scalar_t* __restrict__ p_out,\n";
    source += "    long long outer,\n";
    source += "    long long reduce,\n";
    source += "    long long inner) {\n";

    if (kernel.shape == ReductionShape::kThreadPerOutput) {
      // One thread per output. Adjacent threads differ in the kept axis after the reduced one, so
      // they read adjacent addresses on every step.
      source += "  long long o = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
      source += "  if (o >= outer * inner) return;\n\n";
      source += "  const scalar_t* base = p_in + (o / inner) * reduce * inner + (o % inner);\n\n";
      source += "  scalar_t acc = " + start + ";\n";
      source += "  for (long long r = 0; r < reduce; ++r) {\n";
      source += "    acc = tessiq_combine(acc, base[r * inner]);\n  }\n\n";
      source += "  p_out[o] = acc" + mean + ";\n}\n";
      return source;
    }

    // One block per output, combining through shared memory. The reduced axis is last, so one
    // block reads adjacent elements.
    source += "  long long o = blockIdx.x;\n";
    source += "  if (o >= outer * inner) return;\n\n";
    source += "  const scalar_t* base = p_in + (o / inner) * reduce * inner + (o % inner);\n\n";
    source += "  scalar_t acc = " + start + ";\n";
    source += "  for (long long r = threadIdx.x; r < reduce; r += blockDim.x) {\n";
    source += "    acc = tessiq_combine(acc, base[r * inner]);\n  }\n\n";
    source += "  __shared__ scalar_t scratch[" + std::to_string(kBlockSize) + "];\n";
    source += "  scratch[threadIdx.x] = acc;\n";
    source += "  __syncthreads();\n";
    source += "  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n";
    source += "    if (threadIdx.x < s) {\n";
    source += "      scratch[threadIdx.x] = tessiq_combine(scratch[threadIdx.x], "
              "scratch[threadIdx.x + s]);\n";
    source += "    }\n    __syncthreads();\n  }\n\n";
    source += "  if (threadIdx.x == 0) {\n    p_out[o] = scratch[0]" + mean + ";\n  }\n}\n";
    return source;
  }

  [[nodiscard]] Result<std::string> emit(const FusedKernel& kernel) const override {
    const std::string_view scalar = c_type(kernel.dtype);
    if (scalar.empty()) {
      return unsupported("no CUDA type for " + std::string(ir::dtype_name(kernel.dtype)));
    }

    std::string source = "// Generated by tessiq. Do not edit.\n";
    source += "typedef " + std::string(scalar) + " scalar_t;\n\n";
    source += "extern \"C\" __global__ void " + kernel.name + "(\n";
    for (const KernelInput& input : kernel.inputs) {
      source += "    const scalar_t* __restrict__ p_" + input.name + ",\n";
    }
    source += "    scalar_t* __restrict__ p_" + kernel.output + ",\n";
    source += "    long long n";
    // Divisors not known here follow the count as parameters, one per operand that needs one, in
    // operand order.
    for (const KernelInput& input : kernel.inputs) {
      if (needs_divisor(kernel, input)) {
        source += ",\n    long long d_" + input.name;
      }
    }
    for (const std::string& name : kernel.scalars) {
      source += ",\n    scalar_t " + name;
    }
    source += ") {\n";
    source += "  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
    source += "  if (i >= n) return;\n\n";

    for (const KernelInput& input : kernel.inputs) {
      // Usually `i`. A broadcast operand reads at the index `read_index` gives.
      source += "  scalar_t v_" + input.name + " = p_" + input.name + "[" +
                read_index(kernel, input) + "];\n";
    }
    source += "\n";

    for (const Step& step : kernel.body) {
      source += "  scalar_t v_" + step.result + " = " +
                substitute(passes::elementwise_expression(step.op), step.args) + ";\n";
    }

    source += "\n  p_" + kernel.output + "[i] = v_" + kernel.output + ";\n";
    source += "}\n";
    return source;
  }
};

} // namespace

const Emitter& cuda_emitter() {
  static const CudaEmitter emitter;
  return emitter;
}

} // namespace tessiq::codegen
