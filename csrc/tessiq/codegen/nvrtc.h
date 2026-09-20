// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// Device code, compiled and ready for the driver to load.
struct Cubin {
  std::vector<char> data;

  /// The architecture it was compiled for, such as `sm_89`. A cubin is only loadable on the
  /// architecture it names, which is why it is part of the identity rather than a detail.
  std::string architecture;
};

/// Whether this build can compile CUDA at run time.
///
/// False when the project was built without the CUDA toolkit. Callers check rather than handle a
/// failure, because a machine with no toolkit is an ordinary machine and not an error.
[[nodiscard]] bool nvrtc_available();

/// The runtime compiler's version, as `major.minor`, or empty when unavailable.
///
/// Part of a cubin's cache key. A toolkit upgrade can change what the same source compiles to, so
/// a cubin produced by an older compiler must not be reused by a newer one.
[[nodiscard]] std::string nvrtc_version();

/// Compiles CUDA source for one architecture.
///
/// `name` appears in compiler diagnostics and should identify the kernel.
///
/// A compile error returns the compiler's own log rather than a summary of it. NVRTC reports the
/// line and column of the problem in generated source nobody wrote by hand, so the log is the only
/// thing that makes the failure locatable.
Result<Cubin> compile_cuda(std::string_view source, std::string_view name, int major, int minor);

} // namespace tessiq::codegen
