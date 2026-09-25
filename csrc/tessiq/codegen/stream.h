// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tessiq::codegen {

/// The queue a launch goes to, as the driver's own handle.
///
/// Opaque so that code built without the CUDA toolkit can name one, and a struct so that a buffer
/// cannot be passed where a stream was meant. Every launch takes one explicitly, since work on
/// different streams is not ordered.
struct Stream {
  /// The `CUstream` or `cudaStream_t`, which are the same handle. Null is the legacy default
  /// stream.
  void* handle = nullptr;
};

/// The legacy default stream, which is where every launch in the backend goes.
///
/// It synchronises with every blocking stream on the device, and the runtime's copies use one, so
/// kernels and copies are ordered without explicit synchronisation.
inline constexpr Stream kDefaultStream{};

} // namespace tessiq::codegen
