// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Host code for the C++ tests, registered as test_scale: it copies its input into workspace and
// writes it back scaled, in two launches, and declines a negative factor.

#include <cstddef>

#include "tessiq/codegen/precompiled.h"

namespace {

constexpr unsigned int kThreads = 256;

__global__ void stage(const float* in, float* staged, long long count) {
  const long long i = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
  if (i < count) {
    staged[i] = in[i];
  }
}

__global__ void scale(const float* staged, float* out, long long count, float factor) {
  const long long i = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
  if (i < count) {
    out[i] = staged[i] * factor;
  }
}

void run(tessiq::codegen::Call& call) {
  const double factor = call.real(0);
  if (factor < 0.0) {
    call.decline("a negative factor");
    return;
  }
  const long long count = call.input(0).numel();
  auto* staged =
      static_cast<float*>(call.workspace(static_cast<std::size_t>(count) * sizeof(float)));
  if (staged == nullptr) {
    return;
  }
  const auto blocks = static_cast<unsigned int>((count + kThreads - 1) / kThreads);
  const auto stream = static_cast<cudaStream_t>(call.stream());
  stage<<<blocks, kThreads, 0, stream>>>(call.input(0).as<const float>(), staged, count);
  scale<<<blocks, kThreads, 0, stream>>>(
      staged, call.output().as<float>(), count, static_cast<float>(factor));
}

} // namespace

TESSIQ_PRECOMPILED(test_scale, run)
