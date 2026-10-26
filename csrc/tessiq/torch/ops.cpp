// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Eager kernels, registered against the dispatch key from backend.cpp.
//
// They make tessiq tensors usable outside a compiled graph (allocation, transfers, views) and
// route everything else through the host fallback.

#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/clone_ops.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/view_native.h>
#include <algorithm>
#include <c10/core/ScalarType.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <torch/library.h>

#include "tessiq/runtime/device.h"
#include "tessiq/support/logging.h"
#include "tessiq/torch/backend.h"
#include "tessiq/torch/fallback.h"
#include "tessiq/torch/pack.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "ops";

constexpr c10::DeviceType kDeviceType = c10::DeviceType::PrivateUse1;
constexpr c10::DispatchKey kDispatchKey = c10::DispatchKey::PrivateUse1;

/// `index`, if a device exists there. Raises ValueError otherwise.
int existing(int index) {
  const auto device = device_at(index);
  TORCH_CHECK_VALUE(device.has_value(), device.has_value() ? "" : device.error().message());
  return index;
}

/// The device index a factory op should allocate on.
///
/// An absent or negative index, which PyTorch passes when the caller named no index, means the
/// current device.
int index_for(const std::optional<at::Device>& device) {
  if (!device.has_value() || device->index() < 0) {
    return current_device_index();
  }
  TORCH_CHECK(device->type() == kDeviceType, "expected a tessiq device, got ", *device);
  return existing(static_cast<int>(device->index()));
}

at::Tensor empty_memory_format(at::IntArrayRef size,
                               std::optional<at::ScalarType> dtype,
                               std::optional<at::Layout> /*layout*/,
                               std::optional<at::Device> device,
                               std::optional<bool> /*pin_memory*/,
                               std::optional<at::MemoryFormat> memory_format) {
  const DeviceIndexGuard guard(index_for(device));
  TESSIQ_LOG(kTrace, kLogModule) << "empty, " << size.size() << " dims on device "
                                 << current_device_index();
  return at::detail::empty_generic(size,
                                   get_allocator(current_device_index()),
                                   c10::DispatchKeySet(kDispatchKey),
                                   c10::dtype_or_default(dtype),
                                   memory_format);
}

at::Tensor empty_strided(at::IntArrayRef size,
                         at::IntArrayRef stride,
                         std::optional<at::ScalarType> dtype,
                         std::optional<at::Layout> /*layout*/,
                         std::optional<at::Device> device,
                         std::optional<bool> /*pin_memory*/) {
  const DeviceIndexGuard guard(index_for(device));
  return at::detail::empty_strided_generic(size,
                                           stride,
                                           get_allocator(current_device_index()),
                                           c10::DispatchKeySet(kDispatchKey),
                                           c10::dtype_or_default(dtype));
}

/// True when one raw copy of `a`'s bytes into `b` copies the tensor: same dtype, sizes and strides,
/// and each tensor's elements fill one contiguous span starting at its data pointer.
///
/// The common case: `.to(device)` and `.cpu()` allocate the destination with the source's strides,
/// so a transposed tensor or a slice is copied directly rather than staged.
bool same_dense_layout(const at::Tensor& a, const at::Tensor& b) {
  return a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() &&
         a.strides() == b.strides() && a.is_non_overlapping_and_dense() &&
         b.is_non_overlapping_and_dense();
}

void check(const Result<void>& result) {
  TORCH_CHECK(result.has_value(), result.has_value() ? "" : result.error().to_string());
}

/// The backend behind a tessiq tensor.
Device& backing_device(const at::Tensor& tensor) {
  auto device = device_at(static_cast<int>(tensor.device().index()));
  TORCH_CHECK(device.has_value(), device.has_value() ? "" : device.error().to_string());
  return **device;
}

/// Moves one storage's worth of bytes between a tessiq tensor and the host.
void copy_storage_to_host(Device& device, const at::Tensor& src, void* dst) {
  check(device.copy_device_to_host(dst, src.storage().data(), src.storage().nbytes()));
}

void copy_storage_from_host(Device& device, const void* src, const at::Tensor& dst) {
  check(device.copy_host_to_device(dst.storage().mutable_data(), src, dst.storage().nbytes()));
}

/// A host tensor holding a whole storage's worth of elements, uninitialised.
///
/// Sized from the storage rather than the tensor, so a view's storage offset and strides still
/// address the right elements.
at::Tensor host_buffer_for(const at::Tensor& self) {
  const std::size_t bytes = self.storage().nbytes();
  const auto width = static_cast<std::size_t>(self.element_size());
  TORCH_CHECK(width > 0 && bytes % width == 0,
              "tessiq cannot stage a storage of ",
              bytes,
              " bytes as ",
              width,
              "-byte elements");
  return at::empty({static_cast<std::int64_t>(bytes / width)}, self.options().device(at::kCPU));
}

/// Brings a tessiq tensor to the host, keeping its layout and its position in its storage.
at::Tensor to_host(const at::Tensor& self) {
  at::Tensor buffer = host_buffer_for(self);
  copy_storage_to_host(backing_device(self), self, buffer.storage().mutable_data());
  return buffer.as_strided(self.sizes(), self.strides(), self.storage_offset());
}

/// Writes a host tensor into a tessiq tensor, converting layout and dtype on the host first.
void from_host(const at::Tensor& host_src, const at::Tensor& dst) {
  at::Tensor buffer = host_buffer_for(dst);

  // Read, modify, write: when the destination is a view, the rest of its storage belongs to
  // another tensor and must come back unchanged.
  if (dst.storage_offset() != 0 || !dst.is_contiguous() || buffer.numel() != dst.numel()) {
    copy_storage_to_host(backing_device(dst), dst, buffer.storage().mutable_data());
  }

  at::Tensor host_dst = buffer.as_strided(dst.sizes(), dst.strides(), dst.storage_offset());
  host_dst.copy_(host_src);
  copy_storage_from_host(backing_device(dst), buffer.storage().data(), dst);
}

/// Backs `.to()` in both directions, and copies between tessiq devices.
///
/// Matching dense layouts copy their bytes directly. Anything else is converted on the host with
/// PyTorch's own kernels rather than a second copy of its striding and type promotion rules.
at::Tensor copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(self.sizes() == dst.sizes(),
              "tessiq copy requires matching shapes, got ",
              self.sizes(),
              " and ",
              dst.sizes());

  const bool src_on_device = self.device().type() == kDeviceType;
  const bool dst_on_device = dst.device().type() == kDeviceType;
  TORCH_CHECK(src_on_device || dst_on_device,
              "tessiq _copy_from called with neither tensor on a tessiq device");

  const bool across_backends =
      src_on_device && dst_on_device && self.device().index() != dst.device().index();

  if (across_backends) {
    // Two tessiq devices with no shared address space (the CPU reference is not addressable from
    // a GPU at all), so the bytes go through the host.
    from_host(to_host(self), dst);
  } else if (same_dense_layout(self, dst)) {
    Device& device = src_on_device ? backing_device(self) : backing_device(dst);
    const std::size_t bytes = self.nbytes();
    void* to = dst.mutable_data_ptr();
    const void* from = self.const_data_ptr();

    if (src_on_device && dst_on_device) {
      check(device.copy_device_to_device(to, from, bytes));
    } else if (src_on_device) {
      check(device.copy_device_to_host(to, from, bytes));
    } else {
      check(device.copy_host_to_device(to, from, bytes));
    }
  } else {
    const at::Tensor host_src = src_on_device ? to_host(self) : self;
    if (dst_on_device) {
      from_host(host_src, dst);
    } else {
      // Tensor is a handle, so copy_ is const: nothing is being cast away here.
      dst.copy_(host_src);
    }
  }

  if (!non_blocking) {
    if (src_on_device) {
      check(backing_device(self).synchronize());
    }
    if (dst_on_device) {
      check(backing_device(dst).synchronize());
    }
  }
  return dst;
}

/// Grows a tensor's storage to hold `new_bytes`, preserving what is already there.
///
/// Shrinking is deliberately not handled, matching PyTorch's own resize_: keeping oversized
/// storage avoids a free and realloc on every shrink.
void grow_storage(at::TensorImpl* impl, std::size_t new_bytes) {
  const c10::Storage& storage = impl->unsafe_storage();
  TORCH_CHECK(storage, "tessiq tensor has no storage to resize");

  if (new_bytes > storage.nbytes()) {
    reallocate_storage(storage, new_bytes);
  }
}

/// Native resize_, which the CPU fallback depends on. Through the fallback, resizing a tessiq
/// tensor would copy back by calling resize_ again, and recurse.
///
/// The schema returns Tensor(a!), the same tensor aliased, so this returns a const reference to
/// the parameter.
const at::Tensor& resize_tensor(const at::Tensor& self,
                                at::IntArrayRef size,
                                std::optional<at::MemoryFormat> memory_format) {
  at::TensorImpl* impl = self.unsafeGetTensorImpl();
  impl->set_sizes_contiguous(size);

  const auto elements = static_cast<std::size_t>(impl->storage_offset() + impl->numel());
  grow_storage(impl, elements * impl->dtype().itemsize());

  if (memory_format.has_value()) {
    TORCH_CHECK(*memory_format != at::MemoryFormat::Preserve,
                "tessiq resize_ does not accept MemoryFormat::Preserve");
    impl->empty_tensor_restride(*memory_format);
  }
  // The dispatcher owns the tensor, so the reference outlives the call.
  // NOLINTNEXTLINE(bugprone-return-const-ref-from-parameter)
  return self;
}

/// Points `self` at existing storage, which is how torch rebuilds each tensor it loads. Grows the
/// storage when the view needs more than it holds, as set_ does on CPU and CUDA.
at::Tensor& set_storage(at::Tensor& self,
                        at::Storage source,
                        int64_t storage_offset,
                        at::IntArrayRef size,
                        at::IntArrayRef stride) {
  at::native::checkSetStorage(self, std::move(source), storage_offset, size, stride);
  at::TensorImpl* impl = self.unsafeGetTensorImpl();
  impl->set_storage_offset(storage_offset);
  // No stride, as distinct from an empty one, means contiguous.
  if (stride.data() == nullptr) {
    impl->set_sizes_contiguous(size);
  } else {
    impl->set_sizes_and_strides(size, stride);
  }
  const std::size_t needed = at::detail::computeStorageNbytes(
      size, impl->strides(), impl->dtype().itemsize(), static_cast<std::size_t>(storage_offset));
  if (needed > impl->unsafe_storage().nbytes()) {
    reallocate_storage(impl->unsafe_storage(), needed);
  }
  return self;
}

/// Gives `self` an empty storage of its own, on its device.
at::Tensor& set_empty(at::Tensor& self) {
  c10::Storage storage(c10::Storage::use_byte_size_t(),
                       0,
                       get_allocator(static_cast<int>(self.device().index())),
                       /*resizable=*/true);
  return set_storage(self, std::move(storage), 0, {0}, {});
}

/// Copies into `dst`, resizing it to match `self` first.
///
/// The CPU fallback calls this to bring results back.
at::Tensor copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  if (dst.sizes() != self.sizes()) {
    resize_tensor(dst, self.sizes(), std::nullopt);
  }
  return copy_from(self, dst, /*non_blocking=*/false);
}

/// `clone`, and with it `.contiguous()`, which torch routes through the same operation.
///
/// A clone does no arithmetic, so the device packs the view itself instead of sending it through
/// the fallback. Only a packed result is handled here. `preserve_format` on a non-contiguous view
/// may ask for another layout, and that goes to the fallback.
at::Tensor clone(const at::Tensor& self, std::optional<at::MemoryFormat> memory_format) {
  const at::MemoryFormat format = memory_format.value_or(at::MemoryFormat::Preserve);
  const bool packed_result = format == at::MemoryFormat::Contiguous ||
                             (format == at::MemoryFormat::Preserve && self.is_contiguous());
  if (packed_result) {
    at::Tensor out = at::empty(self.sizes(), self.options());
    const auto packed = pack_on_device(self, out);
    if (packed.has_value()) {
      return out;
    }
    TESSIQ_LOG(kDebug, kLogModule)
        << "no pack kernel for this clone, falling back: " << packed.error().message();
  }
  return at::native::call_fallback_fn<&cpu_fallback, at::_ops::clone>::call(self, memory_format);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &empty_memory_format);
  m.impl("empty_strided", &empty_strided);
  m.impl("_copy_from", &copy_from);
  m.impl("_copy_from_and_resize", &copy_from_and_resize);
  m.impl("resize_", &resize_tensor);
  // set_ points a tensor at a storage, and the fallback would point a host copy at it instead.
  // The two overloads ATen implements generically redispatch to set_storage.
  m.impl("set_.source_Storage", &at::native::set_);
  m.impl("set_.source_Storage_storage_offset", &set_storage);
  m.impl("set_.source_Tensor", &at::native::set_tensor_);
  m.impl("set_", &set_empty);
  m.impl("clone", &clone);

  // A view has to share its input's storage, and the fallback would return a copy, so writes
  // through the view would be lost. Other view operations (permute, transpose, squeeze, ...) are
  // built on as_strided, so these three are enough. They need entries only because the catch-all
  // fallback above shadows them. ATen's implementations touch no data, so they work as they are.
  m.impl("as_strided", at::native::as_strided_tensorimpl);
  m.impl("view", at::native::view);
  m.impl("_reshape_alias", at::native::_reshape_alias);
}

/// Calls through `cpu_fallback`, by operator name.
struct FallbackCounts {
  std::mutex mutex;
  /// Guarded by `mutex`.
  std::map<std::string, std::int64_t> calls;
};

FallbackCounts& fallback_counts() {
  static FallbackCounts counts;
  return counts;
}

} // namespace

/// No backend implements every ATen op, so there is always a tail that lands here. Each call costs
/// a transfer each way, so a hot op stands out in a profile. In a compiled graph, partitioning
/// leaves such ops on the host instead.
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  count_cpu_fallback(op.operator_name());
  at::native::cpu_fallback(op, stack);
}

void count_cpu_fallback(const c10::OperatorName& op) {
  std::string name = c10::toString(op);
  TESSIQ_LOG(kDebug, kLogModule) << "cpu fallback for " << name;
  FallbackCounts& counts = fallback_counts();
  const std::scoped_lock guard(counts.mutex);
  ++counts.calls[std::move(name)];
}

std::map<std::string, std::int64_t> cpu_fallback_counts() {
  FallbackCounts& counts = fallback_counts();
  const std::scoped_lock guard(counts.mutex);
  return counts.calls;
}

} // namespace tessiq::torch_backend
