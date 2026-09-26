// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// What a precompiled custom operation's `<name>.nvcc.cu` includes.
//
// nvcc compiles that file when tessiq is built, so it can hold host code: several launches, a
// library call, a kernel that needs a header NVRTC does not have. It sees no torch. Its host
// function is handed a Call, which holds the operation's tensors, already packed and on the call's
// device, its scalars, the result, the stream to queue work on and a way to take workspace from
// tessiq's allocator:
//
//   void run(tessiq::codegen::Call& call) {
//     const tessiq::codegen::DeviceTensor& x = call.input(0);
//     ...
//     kernel<<<grid, block, 0, static_cast<cudaStream_t>(call.stream())>>>(...);
//   }
//
//   TESSIQ_PRECOMPILED(layer_norm, run)
//
// The operation's .cpp declares `.precompiled()`. csrc/custom_ops/README.md walks through one.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// The element type of a tensor a precompiled operation is handed.
enum class ElementType {
  kFloat32,
  kFloat64,
  kFloat16,
  kBFloat16,
  kInt32,
  kInt64,
  kBool,
  kOther,
};

/// A tensor as precompiled code sees it: packed, on the call's device.
struct DeviceTensor {
  void* data = nullptr;
  std::vector<std::int64_t> sizes;
  ElementType type = ElementType::kOther;

  [[nodiscard]] std::int64_t numel() const;

  /// The extent of `axis`, counting from the end when it is negative.
  [[nodiscard]] std::int64_t size(std::int64_t axis) const;

  template <typename T>
  [[nodiscard]] T* as() const {
    return static_cast<T*>(data);
  }
};

/// A scalar argument: a float, an int or a bool in the schema.
using Scalar = std::variant<double, std::int64_t, bool>;

/// One call of a precompiled operation.
class Call {
 public:
  /// Takes `bytes` of memory on the call's device, or returns null when there is none to give.
  using Allocate = std::function<void*(std::size_t bytes)>;

  Call(std::vector<DeviceTensor> inputs,
       std::vector<Scalar> scalars,
       DeviceTensor output,
       void* stream,
       int device,
       Allocate allocate);

  /// Tensor argument `i`, counting only the tensors, in schema order.
  [[nodiscard]] const DeviceTensor& input(std::size_t i) const;

  [[nodiscard]] std::size_t input_count() const { return inputs_.size(); }

  /// The result, already allocated with the shape the operation's reference gives.
  [[nodiscard]] const DeviceTensor& output() const { return output_; }

  /// Scalar argument `i`, counting only the scalars, in schema order.
  [[nodiscard]] double real(std::size_t i) const;
  [[nodiscard]] std::int64_t integer(std::size_t i) const;
  [[nodiscard]] bool flag(std::size_t i) const;

  /// The `cudaStream_t` to queue every launch and library call on.
  [[nodiscard]] void* stream() const { return stream_; }

  /// The CUDA device the tensors are on, which is current when the host function runs.
  [[nodiscard]] int device() const { return device_; }

  /// `bytes` of memory on the call's device from tessiq's allocator, valid until the work this
  /// call queues has finished.
  ///
  /// Null when the device has none to give, and the call has then failed: return without
  /// launching anything.
  void* workspace(std::size_t bytes);

  /// Hands the call back to the operation's reference, for arguments the kernel cannot take.
  /// Return without launching anything after calling it.
  void decline(std::string reason);

  /// Fails the call with `reason`.
  void fail(std::string reason);

  [[nodiscard]] bool declined() const { return declined_; }

  [[nodiscard]] bool failed() const { return failed_; }

  /// Why the call was declined or failed.
  [[nodiscard]] const std::string& reason() const { return reason_; }

  /// The workspace blocks the call took, for the caller to give back.
  [[nodiscard]] const std::vector<void*>& workspaces() const { return workspaces_; }

 private:
  std::vector<DeviceTensor> inputs_;
  std::vector<Scalar> scalars_;
  DeviceTensor output_;
  void* stream_;
  int device_;
  Allocate allocate_;
  std::vector<void*> workspaces_;
  bool declined_ = false;
  bool failed_ = false;
  std::string reason_;
};

/// The host function in a `.nvcc.cu`.
using PrecompiledFunction = void (*)(Call& call);

/// What TESSIQ_PRECOMPILED records about an operation's host code.
struct Precompiled {
  std::string_view name;

  /// The host function, wrapped so that a failed launch fails the call.
  PrecompiledFunction run = nullptr;

  /// Whether the file's device code has an image CUDA device `ordinal` can load.
  bool (*loads)(int ordinal) = nullptr;

  /// The architectures nvcc built the file for, as __CUDA_ARCH_LIST__ spells them: "890,1200".
  std::string_view architectures;
};

/// Records `entry`. Called by TESSIQ_PRECOMPILED at static initialisation.
bool register_precompiled(const Precompiled& entry);

/// Every operation with host code in this build, in the order the files loaded. A name registered
/// twice appears twice.
std::vector<const Precompiled*> precompiled_operations();

/// The host code registered for operation `name`, or null when this build has none.
const Precompiled* find_precompiled(std::string_view name);

/// "sm_89, sm_120" for "890,1200".
std::string architecture_names(std::string_view architectures);

/// Checks that `entry`'s device code loads on CUDA device `ordinal`. The error, kExternal, names
/// the architectures it was built for and the card's.
Result<void> check_runs_on(const Precompiled& entry, int ordinal);

/// Runs `entry`'s host function for `call`, with `call.device()` current.
///
/// An error when the call failed. A declined call is not an error: the caller asks `declined()`.
Result<void> run_precompiled(const Precompiled& entry, Call& call);

} // namespace tessiq::codegen

// Two steps, so that the argument is expanded before it is made a string.
#define TESSIQ_PRECOMPILED_QUOTE(...) #__VA_ARGS__
#define TESSIQ_PRECOMPILED_STRING(...) TESSIQ_PRECOMPILED_QUOTE(__VA_ARGS__)

#ifdef __CUDA_ARCH_LIST__
#define TESSIQ_PRECOMPILED_ARCHITECTURES TESSIQ_PRECOMPILED_STRING(__CUDA_ARCH_LIST__)
#else
#define TESSIQ_PRECOMPILED_ARCHITECTURES ""
#endif

#ifdef __CUDACC__

/// Registers `fn` as the host code of custom operation `name`, whose .cpp declares
/// `.precompiled()`.
///
/// It also compiles an empty kernel into the file. Whether the CUDA runtime can find an image of
/// that kernel for a device is whether the file was built for that device's architecture, which is
/// checked before the operation first runs there.
#define TESSIQ_PRECOMPILED(name, fn)                                                               \
  namespace {                                                                                      \
  __global__ void tessiq_precompiled_probe_##name() {}                                             \
  bool tessiq_precompiled_loads_##name(int ordinal) {                                              \
    int previous = 0;                                                                              \
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(ordinal) != cudaSuccess) {        \
      static_cast<void>(cudaGetLastError());                                                       \
      return false;                                                                                \
    }                                                                                              \
    cudaFuncAttributes attributes{};                                                               \
    const cudaError_t status =                                                                     \
        cudaFuncGetAttributes(&attributes, tessiq_precompiled_probe_##name);                       \
    static_cast<void>(cudaGetLastError());                                                         \
    static_cast<void>(cudaSetDevice(previous));                                                    \
    return status == cudaSuccess;                                                                  \
  }                                                                                                \
  void tessiq_precompiled_run_##name(::tessiq::codegen::Call& call) {                              \
    fn(call);                                                                                      \
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {                    \
      call.fail(std::string("a launch failed: ") + cudaGetErrorString(status));                    \
    }                                                                                              \
  }                                                                                                \
  [[maybe_unused]] const bool tessiq_precompiled_registered_##name =                               \
      ::tessiq::codegen::register_precompiled({#name,                                              \
                                               &tessiq_precompiled_run_##name,                     \
                                               &tessiq_precompiled_loads_##name,                   \
                                               TESSIQ_PRECOMPILED_ARCHITECTURES});                 \
  }

#endif // __CUDACC__
