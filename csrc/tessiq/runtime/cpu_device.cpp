// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// The CPU reference backend, and the device registry.

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include "tessiq/runtime/allocator.h"
#include "tessiq/runtime/device.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq {

namespace {

/// Device memory is host memory here, so copies are memcpy and synchronize does nothing.
///
/// This backend is never optimised. It is what CUDA results are checked against, so its value is
/// being obviously correct; anything clever in here weakens the oracle.
class CpuDevice final : public Device {
 public:
  [[nodiscard]] DeviceKind kind() const override { return DeviceKind::kCpu; }

  [[nodiscard]] int device_count() const override { return 1; }

  // NOLINTBEGIN(cppcoreguidelines-owning-memory, cppcoreguidelines-no-malloc): this class is
  // the raw allocation layer. Handing out owning pointers is its job, and RAII lives above it in
  // CachingAllocator.
  Result<void*> raw_allocate(std::size_t bytes) override {
    // aligned_alloc requires a size that is a multiple of the alignment, which round_size
    // already guarantees.
    void* ptr = std::aligned_alloc(kDeviceAlignment, CachingAllocator::round_size(bytes));
    if (ptr == nullptr) {
      return resource_exhausted("cpu backend could not allocate " + std::to_string(bytes) +
                                " bytes");
    }
    return ptr;
  }

  void raw_deallocate(void* ptr, std::size_t /*bytes*/) noexcept override { std::free(ptr); }

  // NOLINTEND(cppcoreguidelines-owning-memory, cppcoreguidelines-no-malloc)

  Result<void> copy_host_to_device(void* dst, const void* src, std::size_t bytes) override {
    return copy(dst, src, bytes);
  }

  Result<void> copy_device_to_host(void* dst, const void* src, std::size_t bytes) override {
    return copy(dst, src, bytes);
  }

  Result<void> copy_device_to_device(void* dst, const void* src, std::size_t bytes) override {
    return copy(dst, src, bytes);
  }

  Result<void> synchronize() override { return ok(); }

 private:
  static Result<void> copy(void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0) {
      return ok();
    }
    if (dst == nullptr || src == nullptr) {
      return invalid_argument("copy with a null pointer: dst and src must both be valid");
    }
    std::memcpy(dst, src, bytes);
    return ok();
  }
};

CpuDevice& cpu_device() {
  static CpuDevice device;
  return device;
}

DeviceKind& active_kind() {
  static DeviceKind kind = DeviceKind::kCpu;
  return kind;
}

} // namespace

std::string_view device_kind_name(DeviceKind kind) {
  switch (kind) {
  case DeviceKind::kCpu:
    return "cpu";
  case DeviceKind::kCuda:
    return "cuda";
  }
  TESSIQ_UNREACHABLE("unhandled DeviceKind in device_kind_name");
}

Result<Device*> get_device(DeviceKind kind) {
  switch (kind) {
  case DeviceKind::kCpu:
    return &cpu_device();
  case DeviceKind::kCuda:
    // The CUDA backend lands with the kernels it exists to run. Until then, say so plainly
    // rather than reporting a machine without a GPU.
    return unsupported("the cuda backend is not built into this version of tessiq");
  }
  TESSIQ_UNREACHABLE("unhandled DeviceKind in get_device");
}

Device& current_device() {
  auto device = get_device(active_kind());
  TESSIQ_CHECK(device.has_value(),
               "the active device kind is unavailable: " +
                   std::string(device_kind_name(active_kind())));
  return **device;
}

Result<void> set_current_device(DeviceKind kind) {
  TESSIQ_RETURN_IF_ERROR(get_device(kind));
  active_kind() = kind;
  TESSIQ_LOG(kInfo, "runtime") << "active backend is now " << device_kind_name(kind);
  return ok();
}

} // namespace tessiq
