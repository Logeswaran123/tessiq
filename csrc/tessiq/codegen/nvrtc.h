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

  /// The architecture it was compiled for, such as `sm_89`.
  std::string architecture;
};

/// Whether this build can compile CUDA at run time.
///
/// False when the project was built without the CUDA toolkit, which is not an error.
[[nodiscard]] bool nvrtc_available();

/// The runtime compiler's version, as `major.minor`, or empty when unavailable.
///
/// Part of the cubin cache key.
[[nodiscard]] std::string nvrtc_version();

/// Compiles CUDA source for one architecture.
///
/// `name` appears in compiler diagnostics and should identify the kernel.
///
/// A compile error returns NVRTC's full log, which gives the line and column in the generated
/// source.
Result<Cubin> compile_cuda(std::string_view source, std::string_view name, int major, int minor);

} // namespace tessiq::codegen
