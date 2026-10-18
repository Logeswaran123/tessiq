// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/timing.h"

#include <string>

#include "tessiq/support/result.h"

#ifdef TESSIQ_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace tessiq::codegen {

namespace {

#ifdef TESSIQ_WITH_CUDA

/// The message for a failed CUDA runtime call.
std::string cuda_message(cudaError_t status, const char* what) {
  return std::string(what) + ": " + cudaGetErrorName(status);
}

cudaEvent_t as_event(void* handle) { return static_cast<cudaEvent_t>(handle); }

/// Creates both events on device `ordinal`, or neither.
void create_events(int ordinal, void*& start, void*& stop) {
  if (cudaSetDevice(ordinal) != cudaSuccess) {
    return;
  }
  // Default flags, so the events record timestamps.
  cudaEvent_t first = nullptr;
  cudaEvent_t second = nullptr;
  if (cudaEventCreate(&first) != cudaSuccess) {
    return;
  }
  if (cudaEventCreate(&second) != cudaSuccess) {
    static_cast<void>(cudaEventDestroy(first));
    return;
  }
  start = first;
  stop = second;
}

void destroy_event(void* event) {
  if (event != nullptr) {
    static_cast<void>(cudaEventDestroy(as_event(event)));
  }
}

/// Records `event` on the legacy default stream, where every launch goes.
Result<void> record(void* event) {
  if (const cudaError_t status = cudaEventRecord(as_event(event), nullptr); status != cudaSuccess) {
    return external_error(cuda_message(status, "cudaEventRecord"));
  }
  return ok();
}

/// Records `event` after making `ordinal` current, which an event needs.
///
/// The launcher has usually set this device already, and switching is most of what marking costs,
/// so it switches only when needed.
Result<void> record_on(void* event, int ordinal) {
  int current = -1;
  if (const cudaError_t status = cudaGetDevice(&current); status != cudaSuccess) {
    return external_error(cuda_message(status, "cudaGetDevice"));
  }
  if (current != ordinal) {
    if (const cudaError_t status = cudaSetDevice(ordinal); status != cudaSuccess) {
      return external_error(cuda_message(status, "cudaSetDevice"));
    }
  }
  return record(event);
}

Result<double> elapsed_microseconds(void* start, void* stop) {
  if (const cudaError_t status = cudaEventSynchronize(as_event(stop)); status != cudaSuccess) {
    return external_error(cuda_message(status, "cudaEventSynchronize"));
  }
  float milliseconds = 0.0F;
  if (const cudaError_t status =
          cudaEventElapsedTime(&milliseconds, as_event(start), as_event(stop));
      status != cudaSuccess) {
    return external_error(cuda_message(status, "cudaEventElapsedTime"));
  }
  return static_cast<double>(milliseconds) * 1000.0;
}

#else // TESSIQ_WITH_CUDA

// Without the toolkit no events are created, so a timer is never ready and the members below
// return before reaching anything else here.

void create_events(int /*ordinal*/, void*& /*start*/, void*& /*stop*/) {}

void destroy_event(void* /*event*/) {}

Result<void> record(void* /*event*/) { return unsupported("this build has no CUDA toolkit"); }

Result<void> record_on(void* /*event*/, int /*ordinal*/) {
  return unsupported("this build has no CUDA toolkit");
}

Result<double> elapsed_microseconds(void* /*start*/, void* /*stop*/) {
  return unsupported("this build has no CUDA toolkit");
}

#endif // TESSIQ_WITH_CUDA

} // namespace

bool device_timing_available() {
#ifdef TESSIQ_WITH_CUDA
  return true;
#else
  return false;
#endif
}

DeviceTimer::DeviceTimer(int ordinal) : ordinal_(ordinal) { create_events(ordinal, start_, stop_); }

DeviceTimer::~DeviceTimer() {
  destroy_event(start_);
  destroy_event(stop_);
}

Result<void> DeviceTimer::start() {
  if (!ready()) {
    return unsupported("this timer has no events");
  }
  return record_on(start_, ordinal_);
}

Result<void> DeviceTimer::stop() {
  if (!ready()) {
    return unsupported("this timer has no events");
  }
  TESSIQ_RETURN_IF_ERROR(record(stop_));
  recorded_ = true;
  return ok();
}

Result<double> DeviceTimer::microseconds() const {
  if (!ready() || !recorded_) {
    return unsupported("this timer recorded nothing");
  }
  return elapsed_microseconds(start_, stop_);
}

} // namespace tessiq::codegen
