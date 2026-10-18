// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// What an operation in csrc/custom_ops/ includes to define itself.
//
// An operation is defined once, in its own directory, by a .cpp that says what it computes and how
// its kernel runs:
//
//   TESSIQ_CUSTOM_OP(rms_norm, "(Tensor x, Tensor weight, float eps) -> Tensor")
//       .reference(rms_norm)
//       .launch(launch)
//       .dtypes({at::kFloat});
//
// Its kernel is one of three things: a `.cu` file compiled at run time and launched as `.launch()`
// describes, an `.expression()` the compiler fuses, or host code in a `.nvcc.cu` file compiled
// when tessiq is built, declared with `.precompiled()`.
//
// Once tessiq is imported it is `torch.ops.tessiq.rms_norm`, and it runs on the GPU both called
// directly and inside torch.compile. csrc/custom_ops/README.md walks through adding one.

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
#include <array>
#include <c10/core/DispatchKey.h>
#include <c10/core/ScalarType.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <torch/library.h>

#include "tessiq/codegen/program.h"

namespace tessiq::custom_ops {

/// How one launch of an operation's kernel is laid over the device.
struct Launch {
  /// Blocks along x, y and z.
  std::array<std::uint32_t, 3> grid{1, 1, 1};

  /// Threads per block along x, y and z. At most 1024 in all.
  std::array<std::uint32_t, 3> block{1, 1, 1};

  /// Dynamic shared memory per block, in bytes. Past 48 KB it is asked for from the driver, which
  /// refuses more than the card has.
  std::size_t shared_bytes = 0;

  /// The kernel's parameters after its tensors, in the order it declares them.
  ///
  /// Each one's C++ type is its parameter's C type: `std::int64_t` for `long long`, `std::int32_t`
  /// for `int`, `float` and `double` for themselves. The kernel reads the bytes as its signature
  /// says, so a `double` passed for a `float` parameter is read wrong rather than refused.
  std::vector<codegen::KernelArgument> arguments;
};

/// One parameter of an operation, as its schema names the type.
enum class Parameter {
  kTensor,
  kFloat,
  kInt,
  kBool,
};

/// The schema type a C++ parameter stands for, which is how torch maps them too.
template <typename T>
constexpr Parameter parameter_of() {
  using Plain = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<Plain, at::Tensor>) {
    return Parameter::kTensor;
  } else if constexpr (std::is_same_v<Plain, double>) {
    return Parameter::kFloat;
  } else if constexpr (std::is_same_v<Plain, std::int64_t>) {
    return Parameter::kInt;
  } else {
    static_assert(std::is_same_v<Plain, bool>,
                  "a custom operation's parameters are const at::Tensor&, double, std::int64_t or "
                  "bool, which are the schema's Tensor, float, int and bool");
    return Parameter::kBool;
  }
}

/// Lays a launch out for one call, given its arguments in schema order and the result tensor.
using LaunchFunction =
    std::function<Launch(const std::vector<c10::IValue>& arguments, const at::Tensor& out)>;

/// Registers the reference with torch, under the operation's name, once tessiq is imported.
using ReferenceRegistration = std::function<void(torch::Library& library, const char* name)>;

namespace detail {

template <typename... Args, std::size_t... I>
Launch call_launch(Launch (*fn)(Args...),
                   const std::vector<c10::IValue>& arguments,
                   const at::Tensor& out,
                   std::index_sequence<I...> /*positions*/) {
  using Types = std::tuple<std::remove_cv_t<std::remove_reference_t<Args>>...>;
  return fn(arguments[I].to<std::tuple_element_t<I, Types>>()..., out);
}

} // namespace detail

/// One custom operation: its schema, its reference, and either a kernel or an expression.
///
/// Made by TESSIQ_CUSTOM_OP and filled in by the calls chained onto it. Nothing is checked or
/// registered until tessiq is imported, and a definition that does not hold together fails the
/// import with a message naming the operation, rather than terminating the process during load.
class CustomOp {
 public:
  CustomOp(std::string name, std::string signature, std::string_view file);

  /// What the operation computes, written in ATen. Required.
  ///
  /// It takes the operation's arguments as the schema declares them, `const at::Tensor&` for a
  /// Tensor and `double`, `std::int64_t` and `bool` for float, int and bool. It runs wherever the
  /// kernel cannot: on the host, on tessiq's CPU reference device, and for an element type the
  /// kernel was not written for. It is also how the result's shape is found, by running it on
  /// tensors that have a shape and no data.
  template <typename... Args>
  CustomOp& reference(at::Tensor (*fn)(Args...)) {
    reference_parameters_ = {parameter_of<Args>()...};
    reference_ = [fn](torch::Library& library, const char* name) {
      library.impl(name, torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd, fn));
    };
    return *this;
  }

  /// How to launch the kernel in `<name>.cu`, next to the .cpp this is written in.
  ///
  /// `fn` takes the operation's arguments like the reference does, followed by the result tensor,
  /// and returns the launch for them. The kernel receives a pointer to every tensor argument in
  /// schema order, then a pointer to the result, then `Launch::arguments`. Every tensor is packed
  /// first, so the kernel can index each one from its first element.
  template <typename... Args>
  CustomOp& launch(Launch (*fn)(Args...)) {
    static_assert(sizeof...(Args) >= 1, "a launch function takes the result tensor last");
    using Last = std::tuple_element_t<sizeof...(Args) - 1, std::tuple<Args...>>;
    static_assert(parameter_of<Last>() == Parameter::kTensor,
                  "a launch function takes the result tensor last, as const at::Tensor&");
    launch_parameters_ = {parameter_of<Args>()...};
    launch_parameters_.pop_back();
    launch_ = [fn](const std::vector<c10::IValue>& arguments, const at::Tensor& out) {
      return detail::call_launch(
          fn, arguments, out, std::make_index_sequence<sizeof...(Args) - 1>());
    };
    return *this;
  }

  /// Host code in `<name>.nvcc.cu`, next to the .cpp this is written in, which nvcc compiles when
  /// tessiq is built. That file registers it with TESSIQ_PRECOMPILED, from
  /// tessiq/codegen/precompiled.h, and launches what it needs on the stream it is handed.
  ///
  /// A build without nvcc has no host code for the operation, which then runs its reference.
  CustomOp& precompiled();

  /// One element of the result, for an operation that has no kernel of its own.
  ///
  /// The compiler writes the expression into the loop of whatever it fuses the operation with, the
  /// way it does for `add` and `exp`. `$0` is the first operand's element, `$1` the second's.
  /// Constants are written `scalar_t(1.702)`, never a bare `1.702`, which C++ reads as a double and
  /// which would turn a float32 kernel into double arithmetic. Every argument has to be a tensor.
  CustomOp& expression(std::string expression);

  /// The element types the kernel or expression was written for. Float32 when not given.
  CustomOp& dtypes(std::initializer_list<at::ScalarType> dtypes);

  /// A kernel in another file of the directory, or under another name than the operation's.
  CustomOp& kernel(std::string file, std::string entry);

  /// Gives this operation a rewrite: a pass, `rewrite-<name>`, that finds the computation in
  /// `file` written out in a graph and replaces it with one call to this operation.
  ///
  /// `file` is in the operation's directory, `<name>.pattern.tsr` when not given, and holds a graph
  /// in tessiq's textual IR. Its inputs are named after the operation's tensor arguments, and each
  /// scalar argument is captured by writing `"$name"` where the pattern has it, such as
  /// `{arg1 = "$eps"}`. A compile runs the rewrite only when it is asked to, by naming it in
  /// `options={"rewrites": [...]}`.
  CustomOp& pattern(std::string file = {});

  /// Makes this operation stand in for torch operation `torch_op`, such as `aten::_softmax` or
  /// `aten::add.Tensor`, wherever it declares the types.
  ///
  /// A model calling the torch operation then runs this one instead: called on a tessiq tensor,
  /// and in a compiled graph, where a pass named `override-<name>` swaps the node before anything
  /// else sees it. The torch operation keeps its own backward. The two schemas have to take the
  /// same arguments, with the same names and types, and the reference has to compute what the
  /// torch operation does, which it usually does by calling it.
  CustomOp& replaces(std::string torch_op);

  [[nodiscard]] const std::string& name() const { return name_; }

  [[nodiscard]] const std::string& signature() const { return signature_; }

  [[nodiscard]] const std::string& directory() const { return directory_; }

  [[nodiscard]] const std::vector<Parameter>& reference_parameters() const {
    return reference_parameters_;
  }

  [[nodiscard]] const ReferenceRegistration& reference_registration() const { return reference_; }

  [[nodiscard]] const std::vector<Parameter>& launch_parameters() const {
    return launch_parameters_;
  }

  [[nodiscard]] const LaunchFunction& launch_function() const { return launch_; }

  [[nodiscard]] const std::string& expression_text() const { return expression_; }

  [[nodiscard]] bool is_precompiled() const { return precompiled_; }

  [[nodiscard]] const std::vector<at::ScalarType>& element_types() const { return dtypes_; }

  [[nodiscard]] const std::string& kernel_file() const { return kernel_file_; }

  [[nodiscard]] const std::string& entry() const { return entry_; }

  [[nodiscard]] const std::string& replaced() const { return replaced_; }

  [[nodiscard]] const std::string& pattern_file() const { return pattern_file_; }

 private:
  std::string name_;
  std::string signature_;
  std::string directory_;
  std::vector<Parameter> reference_parameters_;
  ReferenceRegistration reference_;
  std::vector<Parameter> launch_parameters_;
  LaunchFunction launch_;
  std::string expression_;
  bool precompiled_ = false;
  std::vector<at::ScalarType> dtypes_{at::kFloat};
  std::string kernel_file_;
  std::string entry_;
  std::string replaced_;
  std::string pattern_file_;
};

/// Starts the definition of operation `name`. Used through TESSIQ_CUSTOM_OP.
CustomOp& define(std::string name, std::string signature, std::string_view file);

/// Every operation defined in the extension, in the order their files were loaded.
const std::vector<CustomOp*>& definitions();

/// The text of a file in an operation's directory, carried in the extension. Made by the build, one
/// per .cu and .tsr file.
///
/// A kernel is compiled for the device it runs on, the first time it runs there, so the extension
/// carries its source rather than a binary for one architecture. A pattern is parsed when tessiq is
/// imported. `key` is `<directory>/<file>`.
class EmbeddedFile {
 public:
  EmbeddedFile(std::string_view key, std::string_view text);
};

/// The text of file `key`, or nothing when the build carries no such file.
std::optional<std::string_view> find_embedded_file(std::string_view key);

} // namespace tessiq::custom_ops

/// Defines custom operation `tessiq::name`, whose schema is `signature` without the name.
///
/// `name` is the operation's name and must be an identifier. The directory it is written in holds
/// its kernel, `<name>.cu`, unless `.kernel()` says otherwise.
///
/// The definition is held by pointer: bound to a reference, GCC 13 reports it as possibly dangling,
/// because the chained calls take temporaries.
// NOLINTBEGIN(bugprone-throwing-static-initialization): recording a definition allocates, and a
// failure there means the process is out of memory while loading, which is not worth recovering.
#define TESSIQ_CUSTOM_OP(name, signature)                                                          \
  [[maybe_unused]] static const ::tessiq::custom_ops::CustomOp* const tessiq_custom_op_##name =    \
      &::tessiq::custom_ops::define(#name, signature, __FILE__)
// NOLINTEND(bugprone-throwing-static-initialization)
