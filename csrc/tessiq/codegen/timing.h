// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// When the device started and finished a piece of work.
///
/// A launch returns before the work finishes, so a host clock measures only the launch, and
/// synchronising after each kernel serialises the pipeline. Events are recorded into the stream
/// instead and read once at the end of a run, so timing does not change how the program runs.
///
/// Not thread-safe. One timer belongs to one region of one thread.
class DeviceTimer {
 public:
  /// A timer for CUDA device `ordinal`. Check `ready()` before using it.
  explicit DeviceTimer(int ordinal);

  ~DeviceTimer();

  DeviceTimer(const DeviceTimer&) = delete;
  DeviceTimer& operator=(const DeviceTimer&) = delete;
  DeviceTimer(DeviceTimer&&) = delete;
  DeviceTimer& operator=(DeviceTimer&&) = delete;

  /// Whether the events exist, which they do not in a build with no CUDA toolkit.
  [[nodiscard]] bool ready() const { return start_ != nullptr && stop_ != nullptr; }

  /// Marks the stream before the work.
  Result<void> start();

  /// Marks it after.
  Result<void> stop();

  /// Microseconds the device spent between the two marks.
  ///
  /// Waits for the second mark. Fails if the marks were never recorded, so a region with no device
  /// work is not reported as taking no time.
  [[nodiscard]] Result<double> microseconds() const;

 private:
  int ordinal_ = 0;
  /// `cudaEvent_t`, kept as void so that this header compiles without the toolkit.
  void* start_ = nullptr;
  void* stop_ = nullptr;
  bool recorded_ = false;
};

/// Whether this build can time device work at all.
[[nodiscard]] bool device_timing_available();

} // namespace tessiq::codegen
