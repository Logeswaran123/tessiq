// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

#include "tessiq/support/result.h"

namespace tessiq {

/// Minimum alignment of every pointer a Device hands out.
///
/// Chosen to satisfy the widest vector load on the supported targets. Both backends
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

  /// Which physical device of this kind this instance drives. Always 0 for the CPU reference.
  [[nodiscard]] virtual int ordinal() const = 0;

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

/// Constructs the CPU reference backend. Used by the registry; there is exactly one.
std::unique_ptr<Device> make_cpu_device();

/// Number of tessiq devices: the CPU reference, plus one per usable CUDA device.
///
/// Index 0 is always the CPU reference, so an unqualified `device="tessiq"` lands on the backend
/// that is always present and always correct. Indices 1..N are CUDA devices in driver order.
int device_count();

/// The device at `index`, or kInvalidArgument if there is none.
///
/// Devices have static storage duration; the pointer stays valid for the life of the process.
Result<Device*> device_at(int index);

/// The index this thread allocates on, set by PyTorch's device guard around every op.
///
/// Thread-local, because that is what the guard is: two threads can be working on two devices,
/// and c10's allocator interface takes no device argument, so the ambient index is how an
/// allocation knows where to land. Defaults to 0.
int current_device_index();

/// Sets the index for this thread. Fails if `index` names no device, leaving the current one.
Result<void> set_current_device_index(int index);

namespace detail {

/// Sets the index without validating it. For restoring an index that was already valid, which is
/// the only case where failure is impossible and throwing would be unacceptable.
void set_current_device_index_unchecked(int index) noexcept;

} // namespace detail

/// The device this thread is allocating on. Shorthand for device_at(current_device_index()).
Device& current_device();

/// Sets the current index for a scope and restores it on the way out.
///
/// Factory ops carry their device as an argument rather than arriving under a guard, so they set
/// the index themselves for the duration of the allocation. Without this an explicit
/// `device=tessiq:1` silently allocates wherever the thread happened to be.
class DeviceIndexGuard {
 public:
  explicit DeviceIndexGuard(int index) : previous_(current_device_index()) {
    static_cast<void>(set_current_device_index(index));
  }

  // noexcept by construction: previous_ was valid when this guard was built.
  ~DeviceIndexGuard() { detail::set_current_device_index_unchecked(previous_); }

  DeviceIndexGuard(const DeviceIndexGuard&) = delete;
  DeviceIndexGuard& operator=(const DeviceIndexGuard&) = delete;
  DeviceIndexGuard(DeviceIndexGuard&&) = delete;
  DeviceIndexGuard& operator=(DeviceIndexGuard&&) = delete;

 private:
  int previous_;
};

} // namespace tessiq
