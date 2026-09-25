// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/program.h"

#include <cstdint>
#include <string>
#include <vector>

#include "tessiq/codegen/cubin_cache.h"
#include "tessiq/codegen/nvrtc.h"
#include "tessiq/codegen/stream.h"
#include "tessiq/support/result.h"

#ifdef TESSIQ_WITH_CUDA

#include <cstddef>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <variant>

#include <cuda.h>
#include <cuda_runtime.h>

#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq::codegen {

namespace {

constexpr std::string_view kLogModule = "program";

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
/// The driver API acts on the current context, and cudaSetDevice binds the runtime's primary
/// context to this thread. Without it, a launch from a thread torch has not used fails with an
/// invalid context.
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

/// Dynamic shared memory a block gets without the kernel asking for more.
constexpr std::size_t kSharedWithoutAsking = std::size_t{48} * 1024;

/// The most threads any CUDA device runs in one block.
constexpr std::uint64_t kMostThreadsPerBlock = 1024;

Result<void> check_shape(const LaunchShape& shape) {
  std::uint64_t threads = 1;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (shape.grid.at(axis) == 0 || shape.block.at(axis) == 0) {
      return invalid_argument("a launch with a zero in its grid or its block runs nothing");
    }
    threads *= shape.block.at(axis);
  }
  if (threads > kMostThreadsPerBlock) {
    return invalid_argument("a block of " + std::to_string(threads) +
                            " threads is more than the 1024 a device runs");
  }
  return ok();
}

/// Modules are never unloaded, so the handle is not kept. Unloading during static destruction
/// races the driver's own teardown.
class CudaProgram final : public Program {
 public:
  CudaProgram(CUfunction function, int ordinal) : function_(function), ordinal_(ordinal) {}

  ~CudaProgram() override = default;

  TESSIQ_DISALLOW_COPY_AND_MOVE(CudaProgram);

  [[nodiscard]] Result<void> launch(const std::vector<void*>& buffers,
                                    const std::vector<KernelArgument>& arguments,
                                    const LaunchShape& shape,
                                    Stream stream) const override {
    TESSIQ_RETURN_IF_ERROR(check_shape(shape));
    TESSIQ_RETURN_IF_ERROR(activate(ordinal_));
    if (shape.shared_bytes > kSharedWithoutAsking) {
      // More than 48 KB has to be opted into. The driver refuses an amount the card does not
      // have, which is a clearer error than a failed launch.
      if (const CUresult status =
              cuFuncSetAttribute(function_,
                                 CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                 static_cast<int>(shape.shared_bytes));
          status != CUDA_SUCCESS) {
        return driver_error(
            status, "asking for " + std::to_string(shape.shared_bytes) + " bytes of shared memory");
      }
    }

    // Buffers, then arguments. cuLaunchKernel takes the address of each, so both are copied into
    // vectors that outlive the call.
    std::vector<void*> values = buffers;
    std::vector<KernelArgument> scalars = arguments;
    std::vector<void*> addresses;
    addresses.reserve(values.size() + scalars.size());
    for (void*& value : values) {
      addresses.push_back(static_cast<void*>(&value));
    }
    for (KernelArgument& scalar : scalars) {
      addresses.push_back(std::visit([](auto& held) { return static_cast<void*>(&held); }, scalar));
    }

    if (const CUresult status = cuLaunchKernel(function_,
                                               shape.grid[0],
                                               shape.grid[1],
                                               shape.grid[2],
                                               shape.block[0],
                                               shape.block[1],
                                               shape.block[2],
                                               static_cast<unsigned int>(shape.shared_bytes),
                                               static_cast<CUstream>(stream.handle),
                                               addresses.data(),
                                               nullptr);
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
/// Part of the loaded-module key. Fused kernels are named tessiq_fused_<group>, so two graphs
/// often produce kernels with the same name, and often the same size, that do different things.
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

Result<const Program*>
compile_program(std::string_view source, std::string_view entry, int ordinal) {
  TESSIQ_ASSIGN_OR_RETURN(const auto capability, device_capability(ordinal));
  TESSIQ_ASSIGN_OR_RETURN(const Cubin cubin,
                          compile_cached(source, entry, capability.first, capability.second));
  return load_program(cubin, entry, ordinal);
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

Result<const Program*>
compile_program(std::string_view /*source*/, std::string_view entry, int /*ordinal*/) {
  return unsupported("cannot compile '" + std::string(entry) + "': this build has no CUDA toolkit");
}

} // namespace tessiq::codegen

#endif // TESSIQ_WITH_CUDA

namespace tessiq::codegen {

Result<void> Program::launch(const std::vector<void*>& buffers,
                             const std::vector<std::int64_t>& scalars,
                             std::int64_t blocks,
                             Stream stream) const {
  if (blocks <= 0 || blocks > static_cast<std::int64_t>(UINT32_MAX)) {
    return invalid_argument("a launch of " + std::to_string(blocks) + " blocks");
  }
  LaunchShape shape;
  shape.grid[0] = static_cast<std::uint32_t>(blocks);
  shape.block[0] = kBlockSize;
  return launch(
      buffers, std::vector<KernelArgument>(scalars.begin(), scalars.end()), shape, stream);
}

} // namespace tessiq::codegen
