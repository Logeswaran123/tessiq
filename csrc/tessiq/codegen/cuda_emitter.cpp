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
/// A sum starts at zero, which is its identity and correct for a thread that is given no elements
/// to add. A max has no such constant: the lowest representable value is one per type, and
/// `-INFINITY` needs math.h, which NVRTC does not include the way nvcc does. Starting from the
/// slice's first element avoids the question. Every thread can read it, max and min are
/// idempotent so combining it more than once changes nothing, and it is a real value from the
/// data, so a NaN in the slice still propagates.
std::string_view accumulator_start(ReduceOp op) {
  return op == ReduceOp::kSum || op == ReduceOp::kMean ? "(scalar_t)0" : "base[0]";
}

/// How two values combine, as an expression in `a` and `b`.
///
/// max and min test for a NaN first. torch.amax of data containing a NaN is a NaN, and both
/// `fmax` and a plain `a > b ? a : b` return the other operand instead, so neither is the
/// operation torch performs.
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
/// A value argument becomes the variable holding it, which is the IR name because SSA makes it
/// unique. A constant becomes itself.
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
    // A constant is constructed at the element type rather than written bare. `v * 0.1` is a
    // double multiply narrowed afterwards, and for a constant that is not exactly representable
    // in float that is not the same number as a float multiply. The compiler folds the harmless
    // cases either way, so this costs nothing and removes the ones it cannot.
    out += args[index].is_value ? "v_" + args[index].text : "scalar_t(" + args[index].text + ")";
    ++i;
  }
  return out;
}

/// Emits CUDA C++ for NVRTC.
///
/// One thread per element, with a bounds check rather than a launch shaped to divide exactly,
/// because a grid that assumes divisibility is a grid somebody eventually launches wrongly.
///
/// The kernel takes raw pointers and an element count. It knows nothing about strides, which is
/// the contract fusion already enforces: every buffer in a group has the same shape, so walking
/// one index is walking all of them.
class CudaEmitter final : public Emitter {
 public:
  [[nodiscard]] std::string_view name() const override { return "cuda"; }

  [[nodiscard]] bool supports(const FusedKernel& kernel) const override {
    // A dynamic count is allowed: it reaches the kernel as the argument `n` either way, and the
    // source below holds no extent of its own once every index is `i`, which is the only shape a
    // fusion group gets when its extents are not known.
    const bool sized = kernel.elements > 0 || kernel.has_dynamic_extent();
    if (c_type(kernel.dtype).empty() || !sized || kernel.body.empty()) {
      return false;
    }
    for (const Step& step : kernel.body) {
      if (passes::elementwise_expression(step.op).empty()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool supports(const ReductionKernel& kernel) const override {
    // c_type covers the half and boolean types, which have no scalar_t spelling here.
    if (c_type(kernel.dtype).empty()) {
      return false;
    }
    if (kernel.reduce <= 0 || kernel.outputs() <= 0 || kernel.inner <= 0) {
      return false;
    }
    // An integer mean would be an integer division, which is not what torch computes. torch
    // refuses it outright, so this is unreachable rather than a limitation worth working around.
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
      // One thread per output. Consecutive threads differ in the kept axis that follows the
      // reduced one, so they read consecutive addresses on every step of the loop.
      source += "  long long o = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
      source += "  if (o >= outer * inner) return;\n\n";
      source += "  const scalar_t* base = p_in + (o / inner) * reduce * inner + (o % inner);\n\n";
      source += "  scalar_t acc = " + start + ";\n";
      source += "  for (long long r = 0; r < reduce; ++r) {\n";
      source += "    acc = tessiq_combine(acc, base[r * inner]);\n  }\n\n";
      source += "  p_out[o] = acc" + mean + ";\n}\n";
      return source;
    }

    // One block per output, its threads walking the reduced axis together and combining what
    // they found through shared memory. The reduced axis is last here, so the elements one block
    // reads sit next to each other.
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
    source += "    long long n) {\n";
    source += "  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
    source += "  if (i >= n) return;\n\n";

    for (const KernelInput& input : kernel.inputs) {
      // Almost always `i`. An operand with fewer elements than the result, such as the per-row
      // mean a layer norm subtracts, reads a position worked out from the shapes when the kernel
      // was written rather than from strides passed in.
      source += "  scalar_t v_" + input.name + " = p_" + input.name + "[" + input.index + "];\n";
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
