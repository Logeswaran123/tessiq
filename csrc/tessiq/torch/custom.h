// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tessiq/codegen/call.h"
#include "tessiq/codegen/stream.h"
#include "tessiq/ir/graph.h"
#include "tessiq/ir/types.h"
#include "tessiq/support/result.h"
#include "tessiq/torch/kernel.h"

namespace tessiq::torch_backend {

/// A kernel running registered elementwise operation `op` on its own, for the eager path.
///
/// Built by running a one-node graph of `arity` operands of `dtype`, every extent unknown, through
/// the same passes and emitter as a compiled graph. So an eager call runs the code a compiled
/// graph would fuse, and one kernel serves every shape. Every operand has to be the result's
/// shape, which the caller arranges by broadcasting first.
///
/// Refused when `op` is not an elementwise operation the compiler knows, and when there is no
/// kernel for `dtype`, which includes a build with no CUDA toolkit.
[[nodiscard]] Result<std::shared_ptr<Kernel>>
compile_elementwise(std::string_view op, int arity, ir::DType dtype, int ordinal);

/// Checks every operation defined in csrc/custom_ops/ and registers it with torch and the compiler.
///
/// Called once, when the extension is imported. The first definition that does not hold together
/// stops it with an error naming the operation and what is wrong with it, and the import fails.
[[nodiscard]] Result<void> register_custom_ops();

/// Every custom operation registered, as `tessiq::<name>`, sorted.
[[nodiscard]] std::vector<std::string> custom_op_names();

/// A custom operation that stands in for a torch one.
struct CustomOverride {
  /// The custom operation, as `tessiq::<name>`.
  std::string op;

  /// The torch operation it replaces, as a graph spells it: `aten._softmax.default`.
  std::string replaced;

  /// The element types it replaces the torch operation for. Anything else runs the torch one.
  std::vector<at::ScalarType> dtypes;
};

/// Every custom operation that replaces a torch one, by the torch operation's name.
[[nodiscard]] std::vector<CustomOverride> custom_op_overrides();

/// The passes that put those operations in place of the torch ones, `override-<name>`, sorted.
///
/// They run ahead of the rest of the pipeline, so every later pass sees the custom operation.
[[nodiscard]] std::vector<std::string> override_passes();

/// Every custom operation with a pattern, and the name of its rewrite pass, sorted.
///
/// A rewrite pass is registered like any other and runs only in a pipeline that names it, which a
/// compile does when asked to in its options.
[[nodiscard]] std::vector<std::pair<std::string, std::string>> custom_op_rewrites();

/// How many calls of custom operation `name` on a tessiq tensor ran the reference.
///
/// A call runs the reference on the CPU reference device, and on a GPU for an element type the
/// operation was not written for. The result is the same either way, so this count is the only
/// sign that a model is on the slow path.
[[nodiscard]] Result<std::int64_t> custom_op_fallbacks(std::string_view name);

/// Every custom operation whose host code this build compiled, as `tessiq::<name>`, sorted.
[[nodiscard]] std::vector<std::string> precompiled_op_names();

/// Compiles custom operation `name` for CUDA device `ordinal` now, rather than on its first call.
///
/// A kernel is compiled; an expression is compiled once for each type it declares. The error of one
/// that does not compile names the operation and carries the compiler's log. Host code is checked
/// against the card instead, and the error of host code not built for it names both
/// architectures. An operation whose host code this build did not compile has nothing to do.
[[nodiscard]] Result<void> compile_custom_op(std::string_view name, int ordinal);

/// What a custom operation is once it has been checked and registered. Private to custom.cpp.
struct RegisteredOp;

/// A call to a custom operation with a kernel of its own, in a compiled graph.
///
/// The compiler supplies which values the call reads and where its scalars go among them. The
/// kernel and its launch come from the operation's definition.
class CustomKernel final : public Kernel {
 public:
  CustomKernel(codegen::CustomCall call,
               RegisteredOp* op,
               std::vector<std::optional<c10::IValue>> scalars,
               int ordinal);

  [[nodiscard]] std::string_view kind() const override { return "custom"; }

  [[nodiscard]] std::vector<std::string> inputs() const override { return call_.inputs(); }

  [[nodiscard]] const std::string& output() const override { return call_.output; }

  [[nodiscard]] bool packs_inputs() const override { return true; }

  [[nodiscard]] bool has_dynamic_extent() const override { return call_.has_dynamic_extent; }

  /// Its output, and whatever a rewrite folded into the call.
  [[nodiscard]] std::vector<std::string> members() const override;

  /// Found by running the operation's reference on tensors that have these shapes and no data.
  [[nodiscard]] Result<std::vector<std::int64_t>>
  output_shape(const std::vector<at::Tensor>& tensors) const override;

  [[nodiscard]] Result<void> launch(const std::vector<at::Tensor>& tensors,
                                    const at::Tensor& out,
                                    codegen::Stream stream) const override;

  /// The operation it calls, as `tessiq::<name>`.
  [[nodiscard]] const std::string& op() const;

 private:
  /// The call's arguments in schema order, with `tensors` where its tensors go.
  [[nodiscard]] std::vector<c10::IValue> arguments(const std::vector<at::Tensor>& tensors) const;

  codegen::CustomCall call_;
  /// Not const, because compiling its kernel on first use fills in the operation's cache.
  RegisteredOp* op_;
  /// One entry per argument: the scalar, typed as the schema declares it, or nothing for a tensor.
  std::vector<std::optional<c10::IValue>> scalars_;
  int ordinal_;
};

/// A kernel for every call in `graph` to a custom operation with a kernel of its own.
///
/// Each operation's kernel is compiled for `ordinal` here, so one that does not compile fails the
/// graph's compile rather than its first run. A call whose arguments cannot be put back in schema
/// order gets no kernel, and runs through the operation's device implementation instead.
[[nodiscard]] Result<std::vector<std::shared_ptr<Kernel>>> custom_kernels(const ir::Graph& graph,
                                                                          int ordinal);

} // namespace tessiq::torch_backend
