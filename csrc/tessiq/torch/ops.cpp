// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Eager kernels, registered against the dispatch key from backend.cpp.
//
// These exist for two reasons: they are the fallback for anything the compiler declines to take,
// and they are what makes tensors usable at all -- allocation, transfers and enough arithmetic to
// check the plumbing. Breadth arrives with the op generator (PHASES.md, phase 7).

#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <algorithm>
#include <c10/core/ScalarType.h>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <torch/library.h>

#include "tessiq/runtime/device.h"
#include "tessiq/support/logging.h"
#include "tessiq/torch/backend.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "ops";

constexpr c10::DeviceType kDeviceType = c10::DeviceType::PrivateUse1;
constexpr c10::DispatchKey kDispatchKey = c10::DispatchKey::PrivateUse1;

/// The device index a factory op should allocate on.
///
/// An absent or negative index means "wherever this thread already is", which is what PyTorch
/// passes when the caller did not name a device.
int index_for(const std::optional<at::Device>& device) {
  if (!device.has_value() || device->index() < 0) {
    return current_device_index();
  }
  TORCH_CHECK(device->type() == kDeviceType, "expected a tessiq device, got ", *device);
  return static_cast<int>(device->index());
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
                                   get_allocator(),
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
                                           get_allocator(),
                                           c10::DispatchKeySet(kDispatchKey),
                                           c10::dtype_or_default(dtype));
}

/// True when two tensors have byte-identical layouts, so one raw copy of the storage moves the
/// data correctly whether or not either is contiguous.
///
/// This is the case that matters: `.to(device)` and `.cpu()` allocate the destination with the
/// source's own strides, so transferring a transposed or sliced tensor arrives here rather than
/// in the staging path.
bool same_layout(const at::Tensor& a, const at::Tensor& b) {
  return a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() &&
         a.strides() == b.strides() && a.storage_offset() == b.storage_offset() &&
         a.storage().nbytes() == b.storage().nbytes();
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
/// Sized from the storage rather than from the tensor, because a view starts partway into its
/// storage and the elements before that offset belong to whatever it was taken from. Copying only
/// from the offset would put the data at the wrong place in the buffer, and every stride would
/// then read the wrong element.
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

  // Read, modify, write. The destination may be a view, in which case the rest of its storage
  // belongs to another tensor and has to come back unchanged; writing only the view's own
  // elements is not something one storage-sized transfer can do.
  if (dst.storage_offset() != 0 || !dst.is_contiguous() || buffer.numel() != dst.numel()) {
    copy_storage_to_host(backing_device(dst), dst, buffer.storage().mutable_data());
  }

  at::Tensor host_dst = buffer.as_strided(dst.sizes(), dst.strides(), dst.storage_offset());
  host_dst.copy_(host_src);
  copy_storage_from_host(backing_device(dst), buffer.storage().data(), dst);
}

/// Backs `.to()` in both directions, and copies between tessiq devices.
///
/// Layout and dtype changes are done on the host with PyTorch's own kernels. Reimplementing
/// striding and type promotion here would mean maintaining a second copy of those rules, and the
/// second copy is always the wrong one. A device-side cast and a strided copy kernel take over
/// once the kernel library exists.
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
    // Two tessiq devices with no shared address space. cudaMemcpyDeviceToDevice cannot reach
    // across them and the CPU reference is not addressable from a GPU at all, so the bytes go
    // through the host. A peer-to-peer path would belong with the kernels, not here.
    from_host(to_host(self), dst);
  } else if (same_layout(self, dst)) {
    Device& device = src_on_device ? backing_device(self) : backing_device(dst);
    const std::size_t bytes = self.storage().nbytes();
    void* to = dst.storage().mutable_data();
    const void* from = self.storage().data();

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

  const std::size_t current_bytes = storage.nbytes();
  if (new_bytes <= current_bytes) {
    return;
  }

  at::DataPtr data = get_allocator()->allocate(new_bytes);
  if (current_bytes > 0 && storage.data() != nullptr) {
    const auto result = current_device().copy_device_to_device(
        data.get(), storage.data(), std::min(new_bytes, current_bytes));
    TORCH_CHECK(result.has_value(), result.has_value() ? "" : result.error().to_string());
  }
  storage.set_data_ptr_noswap(std::move(data));
  storage.set_nbytes(new_bytes);
}

/// Native resize_, which the CPU fallback depends on.
///
/// Left to the fallback, resizing a tessiq tensor moves it to the host, resizes it there, and
/// copies back by calling resize_ again. That recursion crashed rather than erroring, which is
/// why this one is native.
/// The schema returns Tensor(a!) -- the same tensor, aliased -- so the return type has to be a
/// const reference to the parameter.
const at::Tensor& resize(const at::Tensor& self,
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
  // Returning the parameter by const reference can dangle when the caller passed a temporary.
  // Here the dispatcher owns the tensor and outlives the call, and the schema requires it.
  // NOLINTNEXTLINE(bugprone-return-const-ref-from-parameter)
  return self;
}

/// Copies into `dst`, resizing it to match `self` first.
///
/// The CPU fallback calls this to bring results back, so without it every fallback op fails at
/// the point of returning rather than at the point of dispatch.
at::Tensor copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  if (dst.sizes() != self.sizes()) {
    resize(dst, self.sizes(), std::nullopt);
  }
  return copy_from(self, dst, /*non_blocking=*/false);
}

/// Runs an unimplemented op by moving its tensors to the host, calling PyTorch's own kernel, and
/// moving results back.
///
/// This is what makes the backend usable before the kernel library exists, and it stays useful
/// afterwards: no backend implements all two thousand ATen ops, so there is always a tail that has
/// to land somewhere. It is slow by construction -- two transfers per op -- which is the point.
/// Anything hot shows up immediately in a profile as fallback traffic.
///
/// The compiler path treats the same question as partitioning: ops without a
/// kernel are cut out of the graph and left on the host.
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TESSIQ_LOG(kDebug, kLogModule) << "cpu fallback for " << op.schema().operator_name().name;
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &empty_memory_format);
  m.impl("empty_strided", &empty_strided);
  m.impl("_copy_from", &copy_from);
  m.impl("_copy_from_and_resize", &copy_from_and_resize);
  m.impl("resize_", &resize);
}

} // namespace

} // namespace tessiq::torch_backend
