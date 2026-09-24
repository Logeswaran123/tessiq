// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tessiq::codegen {

/// The queue a launch goes to, as the driver's own handle.
///
/// Opaque here so that code built without the CUDA toolkit can still name one, and a struct rather
/// than a bare pointer so that a buffer cannot be passed where a stream was meant. Every launch
/// takes one explicitly. Work on one stream runs in the order it was queued, and nothing orders two
/// streams against each other unless someone does it on purpose, so a kernel that picked its own
/// stream could run before the kernel that writes its input.
struct Stream {
  /// The `CUstream` or `cudaStream_t`, which are the same handle. Null is the legacy default
  /// stream.
  void* handle = nullptr;
};

/// The legacy default stream, which is where every launch in the backend goes.
///
/// It orders itself against every blocking stream on the device, and the runtime copies a tensor
/// on one of those. So a kernel queued here after the copy that fills its input waits for it, and a
/// copy back to the host after the kernel waits for the kernel, with nobody synchronising.
inline constexpr Stream kDefaultStream{};

} // namespace tessiq::codegen
