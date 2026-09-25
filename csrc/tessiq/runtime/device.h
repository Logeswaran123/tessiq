// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

#include "tessiq/support/result.h"

namespace tessiq {

/// Minimum alignment of every pointer a Device hands out, so a kernel can assume it without
/// checking. The CPU backend uses aligned_alloc; cudaMalloc returns page-aligned memory.
inline constexpr std::size_t kDeviceAlignment = 512;

/// Which backend implements a device.
enum class DeviceKind {
  /// Reference implementation on host memory. What CUDA results are checked against, and always
  /// present.
  kCpu,
  kCuda,
};

std::string_view device_kind_name(DeviceKind kind);

/// Raw memory and transfers for one backend.
///
/// The allocator and the transfer paths are written against this interface rather than a
/// particular backend. Sizes are in bytes. Pointers returned by raw_allocate are device addresses
/// and must not be dereferenced by host code unless kind() is kCpu.
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
  /// The pointer must be aligned to kDeviceAlignment; nothing above this layer realigns it.
  /// Callers go through CachingAllocator, since every call here is a real allocation.
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
/// Index 0 is always the CPU reference, so an unqualified `device="tessiq"` lands there. Indices
/// 1..N are CUDA devices in driver order.
int device_count();

/// The device at `index`, or kInvalidArgument if there is none.
///
/// Devices have static storage duration; the pointer stays valid for the life of the process.
Result<Device*> device_at(int index);

/// The index this thread allocates on, set by PyTorch's device guard around every op.
///
/// Thread-local like the guard: c10's allocator takes no device argument, so this is how an
/// allocation knows where to land. Defaults to 0.
int current_device_index();

/// Sets the index for this thread. Fails if `index` names no device, leaving the current one.
Result<void> set_current_device_index(int index);

namespace detail {

/// Sets the index without validating it, for restoring one that was already valid.
void set_current_device_index_unchecked(int index) noexcept;

} // namespace detail

/// The device this thread is allocating on. Shorthand for device_at(current_device_index()).
Device& current_device();

/// Sets the current index for a scope and restores it on the way out.
///
/// Factory ops carry their device as an argument rather than arriving under a guard, so they set
/// the index for the allocation themselves.
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
