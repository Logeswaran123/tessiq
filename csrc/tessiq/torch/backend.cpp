// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Everything PyTorch needs in order to treat tessiq as a device.
//
// PyTorch reserves one dispatch key for out-of-tree backends. Its internal identifier is fixed in
// PyTorch's own enum and is confined to this file (PLAN.md section 4.4); the name users see is set
// by rename_privateuse1_backend("tessiq") on the Python side.
//
// This is the most version-fragile file in the project. If a torch upgrade breaks the build, it
// breaks here and nowhere else.

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <cstddef>

#include <torch/library.h>

#include "tessiq/runtime/allocator.h"
#include "tessiq/runtime/device.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "backend";

constexpr c10::DeviceType kDeviceType = c10::DeviceType::PrivateUse1;

/// The process-wide allocator for the active backend.
CachingAllocator& allocator() {
  static CachingAllocator instance(&current_device());
  return instance;
}

/// Adapts CachingAllocator to the interface c10 expects.
///
/// c10 reports allocation failure by throwing, so this is one of the few places where an Error
/// becomes an exception outside the bindings layer: the caller is torch, not us.
struct TessiqAllocator final : c10::Allocator {
  c10::DataPtr allocate(std::size_t bytes) override {
    auto ptr = tessiq::torch_backend::allocator().allocate(bytes);
    if (!ptr) {
      TORCH_CHECK(false, ptr.error().to_string());
    }
    return {*ptr, *ptr, &release, c10::Device(kDeviceType, 0)};
  }

  [[nodiscard]] c10::DeleterFnPtr raw_deleter() const override { return &release; }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    const auto result = current_device().copy_device_to_device(dest, src, count);
    TORCH_CHECK(result.has_value(), result.has_value() ? "" : result.error().to_string());
  }

  static void release(void* ptr) { tessiq::torch_backend::allocator().deallocate(ptr); }
};

// REGISTER_ALLOCATOR takes a pointer to a mutable allocator, and c10 owns the lifetime from there.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TessiqAllocator g_allocator;

/// Device and stream bookkeeping for the dispatcher.
///
/// Streams are a stub: the CPU backend runs everything inline, so there is exactly one stream and
/// synchronising it is a no-op. The CUDA backend replaces this with real stream handling.
///
/// Every signature below is fixed by c10, down to the spelling of `static_type`, which the guard
/// registry looks up by name. The suppressions cover the house style this interface cannot follow.
// NOLINTBEGIN(readability-identifier-naming, readability-named-parameter,
//             modernize-use-nodiscard, clang-diagnostic-unused-const-variable)
struct TessiqGuardImpl final : c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = kDeviceType;

  c10::DeviceType type() const override { return kDeviceType; }

  c10::Device exchangeDevice(c10::Device device) const override {
    const c10::Device previous = getDevice();
    setDevice(device);
    return previous;
  }

  c10::Device getDevice() const override { return {kDeviceType, 0}; }

  void setDevice(c10::Device device) const override {
    TORCH_CHECK(device.type() == kDeviceType, "expected a tessiq device, got ", device);
    TORCH_CHECK(device.index() == 0 || device.index() == -1,
                "tessiq currently exposes a single device, got index ",
                device.index());
  }

  void uncheckedSetDevice(c10::Device) const noexcept override {}

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(kDeviceType, 0));
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return static_cast<c10::DeviceIndex>(current_device().device_count());
  }
};

// NOLINTEND(readability-identifier-naming, readability-named-parameter,
//           modernize-use-nodiscard, clang-diagnostic-unused-const-variable)

C10_REGISTER_GUARD_IMPL(PrivateUse1, TessiqGuardImpl);

} // namespace

// REGISTER_ALLOCATOR opens its own anonymous namespace, so it has to sit at namespace scope.
REGISTER_ALLOCATOR(kDeviceType, &g_allocator)

c10::Allocator* get_allocator() { return &g_allocator; }

AllocatorStats allocator_stats() { return allocator().stats(); }

std::size_t empty_cache() { return allocator().empty_cache(); }

void register_allocator() {
  // Registration happens at static initialisation via REGISTER_ALLOCATOR above. This exists so
  // that the Python module can force this translation unit to be linked in and report what it
  // found, rather than the backend silently not existing.
  TESSIQ_LOG(kInfo, kLogModule) << "tessiq backend ready, " << current_device().device_count()
                                << " device(s) of kind "
                                << device_kind_name(current_device().kind());
}

} // namespace tessiq::torch_backend
