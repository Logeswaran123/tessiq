// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/torch/pack.h"

#include <ATen/core/Tensor.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "tessiq/codegen/cubin_cache.h"
#include "tessiq/codegen/pack.h"
#include "tessiq/codegen/program.h"
#include "tessiq/runtime/device.h"
#include "tessiq/support/logging.h"

namespace tessiq::torch_backend {

namespace {

constexpr std::string_view kLogModule = "pack";

/// A compiled pack kernel is identified by what it was written for and where it was loaded.
using PackKey = std::tuple<std::size_t, std::size_t, int>;

/// The pack kernels compiled so far.
///
/// Kept rather than rebuilt because there are at most a handful: one per element width and rank
/// per device, and a model uses two or three. Compiling is milliseconds and this runs per call.
std::map<PackKey, const codegen::Program*>& compiled() {
  static std::map<PackKey, const codegen::Program*> programs;
  return programs;
}

std::mutex& compiled_mutex() {
  static std::mutex mutex;
  return mutex;
}

Result<const codegen::Program*>
program_for(std::size_t element_bytes, std::size_t rank, int ordinal) {
  const PackKey key{element_bytes, rank, ordinal};
  const std::scoped_lock guard(compiled_mutex());
  if (const auto found = compiled().find(key); found != compiled().end()) {
    return found->second;
  }

  TESSIQ_ASSIGN_OR_RETURN(const std::string source, codegen::pack_source(element_bytes, rank));
  const std::string name = codegen::pack_name(element_bytes, rank);
  TESSIQ_ASSIGN_OR_RETURN(const auto capability, codegen::device_capability(ordinal));
  TESSIQ_ASSIGN_OR_RETURN(
      const codegen::Cubin cubin,
      codegen::compile_cached(source, name, capability.first, capability.second));
  TESSIQ_ASSIGN_OR_RETURN(const codegen::Program* program,
                          codegen::load_program(cubin, name, ordinal));

  compiled().emplace(key, program);
  TESSIQ_LOG(kDebug, kLogModule) << "compiled " << name << " for device " << ordinal;
  return program;
}

/// Which CUDA device holds this tensor, or nothing when it is not on one.
///
/// The CPU reference is a tessiq device too, and its memory is host memory, so a pack there is
/// something torch can already do without a kernel.
Result<int> cuda_ordinal_of(const at::Tensor& tensor) {
  const auto index = static_cast<int>(tensor.device().index());
  TESSIQ_ASSIGN_OR_RETURN(Device * device, device_at(index));
  if (device->kind() != DeviceKind::kCuda) {
    return unsupported("device " + std::to_string(index) + " is not a cuda device");
  }
  return device->ordinal();
}

} // namespace

Result<void> pack_on_device(const at::Tensor& src, const at::Tensor& dst) {
  if (!codegen::launcher_available()) {
    return unsupported("this build has no CUDA toolkit, so there is no kernel to launch");
  }
  if (src.sizes() != dst.sizes()) {
    return invalid_argument("a pack writes the shape it reads");
  }
  if (src.element_size() != dst.element_size()) {
    return invalid_argument("a pack moves elements of one width");
  }
  if (!dst.is_contiguous()) {
    return invalid_argument("a pack writes a contiguous tensor");
  }
  if (src.numel() == 0) {
    // Nothing to move, and a launch of no threads is not something to ask the driver for.
    return ok();
  }

  TESSIQ_ASSIGN_OR_RETURN(const int ordinal, cuda_ordinal_of(src));
  const auto rank = static_cast<std::size_t>(src.dim());
  const auto element_bytes = static_cast<std::size_t>(src.element_size());
  TESSIQ_ASSIGN_OR_RETURN(const codegen::Program* program,
                          program_for(element_bytes, rank, ordinal));

  // The extents, then the steps, in the order the generated parameters take them. Steps are in
  // elements, which is what the kernel indexes by, and which is what torch reports.
  std::vector<std::int64_t> scalars;
  scalars.reserve((2 * rank) + 1);
  scalars.push_back(src.numel());
  for (const auto extent : src.sizes()) {
    scalars.push_back(extent);
  }
  for (const auto step : src.strides()) {
    scalars.push_back(step);
  }

  // data_ptr rather than const_data_ptr: the launcher takes the buffers as one list of writable
  // pointers, and a tensor's constness is shallow, so asking the const reference for a mutable
  // pointer is what every other launch here already does. data_ptr carries the storage offset, so
  // a tensor that starts partway into its storage needs nothing extra.
  const std::vector<void*> buffers{src.data_ptr(), dst.data_ptr()};
  const auto blocks = (src.numel() + static_cast<std::int64_t>(codegen::kBlockSize) - 1) /
                      static_cast<std::int64_t>(codegen::kBlockSize);
  return program->launch(buffers, scalars, blocks);
}

} // namespace tessiq::torch_backend
