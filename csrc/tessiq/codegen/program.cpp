// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/program.h"

#ifdef TESSIQ_WITH_CUDA

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq::codegen {

namespace {

constexpr std::string_view kLogModule = "program";

/// Threads per block.
///
/// 256 is the usual starting point: enough to fill a scheduler, small enough that a block's
/// registers do not limit how many are resident. Choosing it per kernel needs something to
/// measure against, so it is one number until there is.
constexpr unsigned int kBlockSize = 256;

Error driver_error(CUresult status, std::string_view what) {
  const char* name = nullptr;
  const char* description = nullptr;
  static_cast<void>(cuGetErrorName(status, &name));
  static_cast<void>(cuGetErrorString(status, &description));
  return external_error(std::string(what) + ": " + (name == nullptr ? "?" : name) + " (" +
                        (description == nullptr ? "?" : description) + ")");
}

/// Makes the device's primary context current on this thread.
///
/// The driver API acts on whatever context is current, and the runtime API creates and keeps the
/// primary one. Torch has already used the runtime by the time anything here runs, so the context
/// exists; setting the device is what binds it to this thread. Without it a launch from a thread
/// torch has not touched fails with an invalid context, which is a confusing way to learn this.
Result<void> activate(int ordinal) {
  if (const cudaError_t status = cudaSetDevice(ordinal); status != cudaSuccess) {
    return external_error(std::string("cudaSetDevice: ") + cudaGetErrorName(status));
  }
  CUcontext context = nullptr;
  if (const CUresult status = cuCtxGetCurrent(&context); status != CUDA_SUCCESS) {
    return driver_error(status, "cuCtxGetCurrent");
  }
  if (context == nullptr) {
    return external_error("no CUDA context is current on this thread after cudaSetDevice(" +
                          std::to_string(ordinal) + ")");
  }
  return ok();
}

/// A loaded module is never unloaded. A program lives for the process, and tearing one down
/// during static destruction races the driver's own teardown, which reports an error nobody can
/// act on. The module handle is therefore not kept: holding state nothing reads invites somebody
/// to wonder what it is for.
class CudaProgram final : public Program {
 public:
  CudaProgram(CUfunction function, int ordinal) : function_(function), ordinal_(ordinal) {}

  ~CudaProgram() override = default;

  TESSIQ_DISALLOW_COPY_AND_MOVE(CudaProgram);

  [[nodiscard]] Result<void> launch(const std::vector<void*>& buffers,
                                    std::int64_t elements) const override {
    TESSIQ_RETURN_IF_ERROR(activate(ordinal_));

    // The kernel takes each buffer and then the count. cuLaunchKernel wants pointers to the
    // arguments rather than the arguments, so this is an array of addresses of the entries above.
    std::vector<void*> values = buffers;
    std::int64_t count = elements;
    std::vector<void*> arguments;
    arguments.reserve(values.size() + 1);
    for (void*& value : values) {
      arguments.push_back(static_cast<void*>(&value));
    }
    arguments.push_back(static_cast<void*>(&count));

    const auto grid =
        static_cast<unsigned int>((elements + kBlockSize - 1) / static_cast<int>(kBlockSize));
    if (const CUresult status = cuLaunchKernel(
            function_, grid, 1, 1, kBlockSize, 1, 1, 0, nullptr, arguments.data(), nullptr);
        status != CUDA_SUCCESS) {
      return driver_error(status, "cuLaunchKernel");
    }
    return ok();
  }

 private:
  CUfunction function_;
  int ordinal_;
};

std::mutex& programs_mutex() {
  static std::mutex mutex;
  return mutex;
}

/// FNV-1a over a cubin's bytes, as sixteen hex digits.
///
/// The loaded-module map is keyed on this rather than on the cubin's length. Every fused kernel is
/// named tessiq_fused_<group>, so two unrelated graphs routinely produce a kernel with the same
/// name for the same device, and two such cubins are very often the same number of bytes: the same
/// loop with one arithmetic instruction changed compiles to the same size. Keying on length made
/// the second one load the first one's module and launch the wrong kernel, with no error anywhere.
std::string content_digest(const std::vector<char>& data) {
  constexpr std::uint64_t kOffsetBasis = 1469598103934665603ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;

  std::uint64_t hash = kOffsetBasis;
  for (const char byte : data) {
    hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(byte));
    hash *= kPrime;
  }

  constexpr std::string_view kDigits = "0123456789abcdef";
  std::string text(16, '0');
  for (std::size_t position = 16; position > 0; --position) {
    text[position - 1] = kDigits[hash & 0xFU];
    hash >>= 4U;
  }
  return text;
}

/// Loaded modules, keyed by the function, its cubin's contents, and the device it was loaded onto.
std::unordered_map<std::string, std::unique_ptr<CudaProgram>>& programs() {
  static std::unordered_map<std::string, std::unique_ptr<CudaProgram>> map;
  return map;
}

} // namespace

bool launcher_available() { return true; }

Result<std::pair<int, int>> device_capability(int ordinal) {
  int major = 0;
  int minor = 0;
  if (const cudaError_t status =
          cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, ordinal);
      status != cudaSuccess) {
    return external_error(std::string("cudaDeviceGetAttribute(major): ") +
                          cudaGetErrorName(status));
  }
  if (const cudaError_t status =
          cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, ordinal);
      status != cudaSuccess) {
    return external_error(std::string("cudaDeviceGetAttribute(minor): ") +
                          cudaGetErrorName(status));
  }
  return std::pair<int, int>{major, minor};
}

std::size_t loaded_program_count() {
  const std::scoped_lock guard(programs_mutex());
  return programs().size();
}

Result<const Program*> load_program(const Cubin& cubin, std::string_view function, int ordinal) {
  const std::string key = std::string(function) + "@" + std::to_string(ordinal) + "#" +
                          content_digest(cubin.data) + ":" + cubin.architecture;

  const std::scoped_lock guard(programs_mutex());
  if (const auto it = programs().find(key); it != programs().end()) {
    return it->second.get();
  }

  TESSIQ_RETURN_IF_ERROR(activate(ordinal));

  CUmodule module = nullptr;
  if (const CUresult status = cuModuleLoadData(&module, cubin.data.data());
      status != CUDA_SUCCESS) {
    return driver_error(status, "cuModuleLoadData for " + std::string(function));
  }

  CUfunction handle = nullptr;
  if (const CUresult status = cuModuleGetFunction(&handle, module, std::string(function).c_str());
      status != CUDA_SUCCESS) {
    static_cast<void>(cuModuleUnload(module));
    return driver_error(status, "cuModuleGetFunction for " + std::string(function));
  }

  TESSIQ_LOG(kDebug, kLogModule) << "loaded " << function << " on device " << ordinal;
  const auto [it, inserted] =
      programs().emplace(key, std::make_unique<CudaProgram>(handle, ordinal));
  return it->second.get();
}

} // namespace tessiq::codegen

#else // TESSIQ_WITH_CUDA

#include <cstddef>
#include <string>
#include <utility>

namespace tessiq::codegen {

bool launcher_available() { return false; }

std::size_t loaded_program_count() { return 0; }

Result<std::pair<int, int>> device_capability(int /*ordinal*/) {
  return unsupported("this build has no CUDA toolkit");
}

Result<const Program*>
load_program(const Cubin& /*cubin*/, std::string_view function, int /*ordinal*/) {
  return unsupported("cannot load '" + std::string(function) + "': this build has no CUDA toolkit");
}

} // namespace tessiq::codegen

#endif // TESSIQ_WITH_CUDA
