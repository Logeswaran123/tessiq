// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

#include "tessiq/codegen/nvrtc.h"
#include "tessiq/support/result.h"

namespace tessiq::codegen {

/// Counters for the on-disk cubin cache.
struct CubinCacheStats {
  /// Compiles answered from a file.
  std::size_t hits = 0;
  /// Compiles that had to run the compiler.
  std::size_t misses = 0;
  /// Results that could not be written, which is not a failure: the cubin is still returned.
  std::size_t write_failures = 0;
};

/// Where compiled kernels are kept between runs.
///
/// `$TESSIQ_CACHE_DIR` if set, otherwise `$XDG_CACHE_HOME/tessiq/cubin`, otherwise
/// `$HOME/.cache/tessiq/cubin`.
[[nodiscard]] std::filesystem::path cubin_cache_dir();

/// Compiles `source`, reusing a file from a previous run when one matches.
///
/// The key is a digest of the source, the architecture and the NVRTC version, since a toolkit
/// upgrade can change what the same source compiles to.
///
/// A cache that cannot be read or written is not an error; the kernel is compiled and returned
/// either way.
Result<Cubin> compile_cached(std::string_view source, std::string_view name, int major, int minor);

[[nodiscard]] CubinCacheStats cubin_cache_stats();

/// Forgets the counters. The files stay, since something else may be using them.
void reset_cubin_cache_stats();

} // namespace tessiq::codegen
