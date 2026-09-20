// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/nvrtc.h"

#ifdef TESSIQ_WITH_CUDA

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <nvrtc.h>

#include "tessiq/support/logging.h"

namespace tessiq::codegen {

namespace {

constexpr std::string_view kLogModule = "nvrtc";

Error nvrtc_error(nvrtcResult status, std::string_view what) {
  return external_error(std::string(what) + ": " + nvrtcGetErrorString(status));
}

/// The compiler's log, which is where a failure in generated source becomes locatable.
std::string compile_log(nvrtcProgram program) {
  std::size_t size = 0;
  if (nvrtcGetProgramLogSize(program, &size) != NVRTC_SUCCESS || size <= 1) {
    return {};
  }
  std::string log(size, '\0');
  if (nvrtcGetProgramLog(program, log.data()) != NVRTC_SUCCESS) {
    return {};
  }
  log.resize(size - 1);
  return log;
}

/// Destroys the program however the function exits, including the error paths, of which there are
/// several.
class ProgramGuard {
 public:
  explicit ProgramGuard(nvrtcProgram program) : program_(program) {}

  ~ProgramGuard() { static_cast<void>(nvrtcDestroyProgram(&program_)); }

  ProgramGuard(const ProgramGuard&) = delete;
  ProgramGuard& operator=(const ProgramGuard&) = delete;
  ProgramGuard(ProgramGuard&&) = delete;
  ProgramGuard& operator=(ProgramGuard&&) = delete;

 private:
  nvrtcProgram program_;
};

} // namespace

bool nvrtc_available() { return true; }

std::string nvrtc_version() {
  int major = 0;
  int minor = 0;
  if (nvrtcVersion(&major, &minor) != NVRTC_SUCCESS) {
    return {};
  }
  return std::to_string(major) + "." + std::to_string(minor);
}

Result<Cubin> compile_cuda(std::string_view source, std::string_view name, int major, int minor) {
  const std::string owned(source);
  const std::string filename = std::string(name) + ".cu";

  nvrtcProgram program = nullptr;
  if (const nvrtcResult status =
          nvrtcCreateProgram(&program, owned.c_str(), filename.c_str(), 0, nullptr, nullptr);
      status != NVRTC_SUCCESS) {
    return nvrtc_error(status, "nvrtcCreateProgram");
  }
  const ProgramGuard guard(program);

  // Compiling straight to a cubin rather than to PTX skips the driver's own compile on every
  // load, which is the cost a disk cache exists to avoid. The price is that the result only loads
  // on this architecture, which is why the architecture is part of the cubin's identity.
  const std::string architecture = "sm_" + std::to_string(major) + std::to_string(minor);
  const std::string arch_flag = "--gpu-architecture=" + architecture;
  const std::array<const char*, 2> options{{arch_flag.c_str(), "--std=c++17"}};

  if (const nvrtcResult status =
          nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
      status != NVRTC_SUCCESS) {
    const std::string log = compile_log(program);
    return external_error("nvrtc could not compile '" + std::string(name) +
                          "': " + nvrtcGetErrorString(status) + (log.empty() ? "" : "\n" + log));
  }

  if (const std::string log = compile_log(program); !log.empty()) {
    TESSIQ_LOG(kDebug, kLogModule) << name << ": " << log;
  }

  std::size_t size = 0;
  if (const nvrtcResult status = nvrtcGetCUBINSize(program, &size); status != NVRTC_SUCCESS) {
    return nvrtc_error(status, "nvrtcGetCUBINSize");
  }

  Cubin cubin;
  cubin.architecture = architecture;
  cubin.data.resize(size);
  if (const nvrtcResult status = nvrtcGetCUBIN(program, cubin.data.data());
      status != NVRTC_SUCCESS) {
    return nvrtc_error(status, "nvrtcGetCUBIN");
  }

  TESSIQ_LOG(kDebug, kLogModule) << "compiled " << name << " for " << architecture << ", " << size
                                 << " bytes";
  return cubin;
}

} // namespace tessiq::codegen

#else // TESSIQ_WITH_CUDA

namespace tessiq::codegen {

bool nvrtc_available() { return false; }

std::string nvrtc_version() { return {}; }

Result<Cubin>
compile_cuda(std::string_view /*source*/, std::string_view name, int /*major*/, int /*minor*/) {
  return unsupported("cannot compile '" + std::string(name) + "': this build has no CUDA toolkit");
}

} // namespace tessiq::codegen

#endif // TESSIQ_WITH_CUDA
