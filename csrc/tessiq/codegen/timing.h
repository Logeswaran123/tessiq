// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// When the device started and finished a piece of work.
///
/// A launch returns long before the work does, so timing one with a host clock measures the cost
/// of asking. The obvious fix is to wait for the device before stopping the clock, and that is
/// what this project did until now. It measures the right thing and it changes the program while
/// measuring it: waiting after every kernel turns a pipeline the device would overlap into a
/// serial one, and on a transformer block it more than doubled the call.
///
/// Marking the stream costs the host almost nothing, because the mark is a command in the queue
/// like any other. The reading waits, and it happens once at the end of a run when the work has
/// finished anyway. So the program keeps its own shape while being measured, which is the only
/// way the proportions mean anything.
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
  /// Waits for the second mark, so a caller reads after the run rather than during it. Fails if
  /// the marks were never recorded, which is what distinguishes a region that ran no device work
  /// from one that ran some and took no time.
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
