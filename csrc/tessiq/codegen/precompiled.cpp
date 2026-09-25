// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/precompiled.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tessiq/codegen/program.h"

#ifdef TESSIQ_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace tessiq::codegen {

namespace {

std::vector<Precompiled>& registry() {
  static std::vector<Precompiled> entries;
  return entries;
}

std::mutex& registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

/// "sm_89" for "890". Anything that is not a number is kept as it is.
std::string architecture_name(std::string_view value) {
  int number = 0;
  for (const char c : value) {
    if (c < '0' || c > '9') {
      return std::string(value);
    }
    number = (number * 10) + (c - '0');
  }
  return "sm_" + std::to_string(number / 10);
}

} // namespace

std::int64_t DeviceTensor::numel() const {
  std::int64_t count = 1;
  for (const std::int64_t extent : sizes) {
    count *= extent;
  }
  return count;
}

std::int64_t DeviceTensor::size(std::int64_t axis) const {
  const auto rank = static_cast<std::int64_t>(sizes.size());
  const std::int64_t resolved = axis < 0 ? axis + rank : axis;
  return resolved >= 0 && resolved < rank ? sizes[static_cast<std::size_t>(resolved)] : 0;
}

Call::Call(std::vector<DeviceTensor> inputs,
           std::vector<Scalar> scalars,
           DeviceTensor output,
           void* stream,
           int device,
           Allocate allocate)
    : inputs_(std::move(inputs)), scalars_(std::move(scalars)), output_(std::move(output)),
      stream_(stream), device_(device), allocate_(std::move(allocate)) {}

const DeviceTensor& Call::input(std::size_t i) const { return inputs_.at(i); }

double Call::real(std::size_t i) const { return std::get<double>(scalars_.at(i)); }

std::int64_t Call::integer(std::size_t i) const { return std::get<std::int64_t>(scalars_.at(i)); }

bool Call::flag(std::size_t i) const { return std::get<bool>(scalars_.at(i)); }

void* Call::workspace(std::size_t bytes) {
  void* block = allocate_ ? allocate_(bytes) : nullptr;
  if (block == nullptr) {
    fail("no " + std::to_string(bytes) + " bytes of workspace on CUDA device " +
         std::to_string(device_));
    return nullptr;
  }
  workspaces_.push_back(block);
  return block;
}

void Call::decline(std::string reason) {
  declined_ = true;
  reason_ = std::move(reason);
}

void Call::fail(std::string reason) {
  failed_ = true;
  reason_ = std::move(reason);
}

bool register_precompiled(const Precompiled& entry) {
  const std::scoped_lock guard(registry_mutex());
  registry().push_back(entry);
  return true;
}

std::vector<const Precompiled*> precompiled_operations() {
  const std::scoped_lock guard(registry_mutex());
  std::vector<const Precompiled*> entries;
  entries.reserve(registry().size());
  for (const Precompiled& entry : registry()) {
    entries.push_back(&entry);
  }
  return entries;
}

const Precompiled* find_precompiled(std::string_view name) {
  const std::scoped_lock guard(registry_mutex());
  for (const Precompiled& entry : registry()) {
    if (entry.name == name) {
      return &entry;
    }
  }
  return nullptr;
}

std::string architecture_names(std::string_view architectures) {
  std::string names;
  while (!architectures.empty()) {
    const std::size_t comma = architectures.find(',');
    names += names.empty() ? "" : ", ";
    names += architecture_name(architectures.substr(0, comma));
    if (comma == std::string_view::npos) {
      break;
    }
    architectures.remove_prefix(comma + 1);
  }
  return names.empty() ? "no architecture" : names;
}

Result<void> check_runs_on(const Precompiled& entry, int ordinal) {
  if (entry.loads != nullptr && entry.loads(ordinal)) {
    return ok();
  }
  const Result<std::pair<int, int>> capability = device_capability(ordinal);
  const std::string card = capability.has_value() ? "sm_" + std::to_string(capability->first) +
                                                        std::to_string(capability->second)
                                                  : "an architecture it could not report";
  std::string message = "tessiq::" + std::string(entry.name) + " was built for " +
                        architecture_names(entry.architectures) + ", and CUDA device " +
                        std::to_string(ordinal) + " is " + card;
  if (capability.has_value()) {
    message += "; rebuild tessiq with CMAKE_CUDA_ARCHITECTURES including " +
               std::to_string(capability->first) + std::to_string(capability->second);
  }
  // External, the CUDA runtime having found no image for the card, rather than unsupported, which
  // would send a compiled graph quietly back to the host instead of saying the build is wrong.
  return external_error(message);
}

#ifdef TESSIQ_WITH_CUDA

Result<void> run_precompiled(const Precompiled& entry, Call& call) {
  if (const cudaError_t status = cudaSetDevice(call.device()); status != cudaSuccess) {
    return external_error(std::string("cudaSetDevice: ") + cudaGetErrorName(status));
  }
  entry.run(call);
  if (call.failed()) {
    return external_error("tessiq::" + std::string(entry.name) + ": " + call.reason());
  }
  return ok();
}

#else // TESSIQ_WITH_CUDA

Result<void> run_precompiled(const Precompiled& entry, Call& /*call*/) {
  return unsupported("tessiq::" + std::string(entry.name) + " needs a build with CUDA");
}

#endif // TESSIQ_WITH_CUDA

} // namespace tessiq::codegen
