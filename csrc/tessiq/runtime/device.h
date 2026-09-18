// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string_view>

#include "tessiq/support/result.h"

namespace tessiq {

/// Minimum alignment of every pointer a Device hands out.
///
/// Chosen to satisfy the widest vector load on the targets in PLAN.md section 3. Both backends
/// meet it for free: cudaMalloc aligns to at least 256 bytes, and the CPU backend uses
/// aligned_alloc. Stating it as a contract means a kernel can assume alignment without checking,
/// and a new backend knows what it owes.
inline constexpr std::size_t kDeviceAlignment = 512;

/// Which backend implements a device.
enum class DeviceKind {
  /// Reference implementation on host memory. Correctness oracle, and the only backend that is
  /// guaranteed to exist.
  kCpu,
  kCuda,
};

std::string_view device_kind_name(DeviceKind kind);

/// Raw memory and transfers for one backend.
///
/// This is the whole hardware interface: everything above it -- the caching allocator, the
/// compiler, the kernels -- is written against this and does not know what it is talking to.
/// Adding a backend means implementing Device and a kernel emitter, and nothing else changes.
///
/// Sizes are in bytes. Pointers returned by raw_allocate are device addresses and must not be
/// dereferenced by host code unless kind() is kCpu.
///
/// Thread-compatible: concurrent calls on one instance need external synchronisation. The
/// CachingAllocator above it does that locking.
class Device {
 public:
  virtual ~Device() = default;

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;

  [[nodiscard]] virtual DeviceKind kind() const = 0;

  /// Number of usable devices of this kind. Zero means the backend is unavailable on this machine.
  [[nodiscard]] virtual int device_count() const = 0;

  /// Allocates `bytes` of raw memory. Returns kResourceExhausted if the backend cannot satisfy it.
  ///
  /// The returned pointer must be aligned to kDeviceAlignment. Nothing above this layer aligns
  /// on the backend's behalf, so a backend that returns unaligned memory produces misaligned
  /// tensors and, on CUDA, wrong results rather than a crash.
  ///
  /// Callers go through CachingAllocator rather than calling this directly; every call here is a
  /// real allocation from the driver.
  virtual Result<void*> raw_allocate(std::size_t bytes) = 0;

  /// Releases memory from raw_allocate. Must not fail: it runs from destructors.
  virtual void raw_deallocate(void* ptr, std::size_t bytes) noexcept = 0;

  virtual Result<void> copy_host_to_device(void* dst, const void* src, std::size_t bytes) = 0;

  virtual Result<void> copy_device_to_host(void* dst, const void* src, std::size_t bytes) = 0;

  virtual Result<void> copy_device_to_device(void* dst, const void* src, std::size_t bytes) = 0;

  /// Blocks until all work submitted to this device has completed.
  virtual Result<void> synchronize() = 0;

 protected:
  Device() = default;
};

/// Returns the device for `kind`, or kUnsupported if this build has no such backend and
/// kNotFound if the backend exists but the machine has no such hardware.
///
/// Devices are process-wide singletons with static storage duration; the reference stays valid
/// for the life of the process.
Result<Device*> get_device(DeviceKind kind);

/// Returns the backend the runtime is currently using. Defaults to kCpu.
Device& current_device();

/// Switches the active backend. Fails if `kind` is unavailable, leaving the current one in place.
Result<void> set_current_device(DeviceKind kind);

} // namespace tessiq
