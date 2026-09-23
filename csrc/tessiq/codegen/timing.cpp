// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/timing.h"

#include <string>

#include "tessiq/support/result.h"

#ifdef TESSIQ_WITH_CUDA

#include <cuda_runtime.h>

namespace tessiq::codegen {

namespace {

/// The message an external failure carries, built once so both return types can use it.
std::string cuda_message(cudaError_t status, const char* what) {
  return std::string(what) + ": " + cudaGetErrorName(status);
}

cudaEvent_t as_event(void* handle) { return static_cast<cudaEvent_t>(handle); }

} // namespace

bool device_timing_available() { return true; }

DeviceTimer::DeviceTimer(int ordinal) : ordinal_(ordinal) {
  if (cudaSetDevice(ordinal) != cudaSuccess) {
    return;
  }
  // Default flags, which time with the device's own clock. cudaEventDisableTiming would make
  // these cheaper and useless.
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (cudaEventCreate(&start) != cudaSuccess) {
    return;
  }
  if (cudaEventCreate(&stop) != cudaSuccess) {
    static_cast<void>(cudaEventDestroy(start));
    return;
  }
  start_ = start;
  stop_ = stop;
}

DeviceTimer::~DeviceTimer() {
  if (start_ != nullptr) {
    static_cast<void>(cudaEventDestroy(as_event(start_)));
  }
  if (stop_ != nullptr) {
    static_cast<void>(cudaEventDestroy(as_event(stop_)));
  }
}

Result<void> DeviceTimer::start() {
  if (!ready()) {
    return unsupported("this timer has no events");
  }
  // Switched only when it has to be. An event can only be recorded on the device it was created
  // on, so the check has to happen, but the switch itself is most of what marking costs and the
  // device is almost always already this one: the launcher set it for the kernel about to run.
  int current = -1;
  if (const cudaError_t status = cudaGetDevice(&current); status != cudaSuccess) {
    return external_error(cuda_message(status, "cudaGetDevice"));
  }
  if (current != ordinal_) {
    if (const cudaError_t status = cudaSetDevice(ordinal_); status != cudaSuccess) {
      return external_error(cuda_message(status, "cudaSetDevice"));
    }
  }
  // Stream 0, which is where the launcher puts every kernel.
  if (const cudaError_t status = cudaEventRecord(as_event(start_), nullptr);
      status != cudaSuccess) {
    return external_error(cuda_message(status, "cudaEventRecord"));
  }
  return ok();
}

Result<void> DeviceTimer::stop() {
  if (!ready()) {
    return unsupported("this timer has no events");
  }
  if (const cudaError_t status = cudaEventRecord(as_event(stop_), nullptr); status != cudaSuccess) {
    return external_error(cuda_message(status, "cudaEventRecord"));
  }
  recorded_ = true;
  return ok();
}

Result<double> DeviceTimer::microseconds() const {
  if (!ready() || !recorded_) {
    return unsupported("this timer recorded nothing");
  }
  if (const cudaError_t status = cudaEventSynchronize(as_event(stop_)); status != cudaSuccess) {
    return external_error(cuda_message(status, "cudaEventSynchronize"));
  }
  float milliseconds = 0.0F;
  if (const cudaError_t status =
          cudaEventElapsedTime(&milliseconds, as_event(start_), as_event(stop_));
      status != cudaSuccess) {
    return external_error(cuda_message(status, "cudaEventElapsedTime"));
  }
  return static_cast<double>(milliseconds) * 1000.0;
}

} // namespace tessiq::codegen

#else // TESSIQ_WITH_CUDA

namespace tessiq::codegen {

bool device_timing_available() { return false; }

DeviceTimer::DeviceTimer(int ordinal) : ordinal_(ordinal) {}

DeviceTimer::~DeviceTimer() = default;

Result<void> DeviceTimer::start() {
  return unsupported("this build has no CUDA toolkit, so there is no device clock to read");
}

Result<void> DeviceTimer::stop() {
  return unsupported("this build has no CUDA toolkit, so there is no device clock to read");
}

Result<double> DeviceTimer::microseconds() const {
  return unsupported("this build has no CUDA toolkit, so there is no device clock to read");
}

} // namespace tessiq::codegen

#endif // TESSIQ_WITH_CUDA
