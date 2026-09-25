// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include "tessiq/codegen/cubin_cache.h"

#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "tessiq/support/logging.h"

namespace tessiq::codegen {

namespace {

constexpr std::string_view kLogModule = "cubin";

std::mutex& stats_mutex() {
  static std::mutex mutex;
  return mutex;
}

CubinCacheStats& stats() {
  static CubinCacheStats counters;
  return counters;
}

/// FNV-1a over the inputs that decide what the compiler produces.
///
/// Not cryptographic, which is fine: the inputs are kernels tessiq compiles, not untrusted data,
/// and 64 bits makes an accidental collision negligible.
std::string
digest(std::string_view source, std::string_view architecture, std::string_view version) {
  constexpr std::uint64_t kOffset = 1469598103934665603ULL;
  constexpr std::uint64_t kPrime = 1099511628211ULL;

  std::uint64_t hash = kOffset;
  for (const std::string_view part : {source, architecture, version}) {
    for (const char c : part) {
      hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
      hash *= kPrime;
    }
    // Separated so that concatenating two fields differently cannot collide with itself.
    hash ^= 0xffULL;
    hash *= kPrime;
  }

  constexpr std::string_view kHex = "0123456789abcdef";
  std::string out(16, '0');
  for (std::size_t i = 0; i < 16; ++i) {
    out.at(15 - i) = kHex.at((hash >> (i * 4)) & 0xf);
  }
  return out;
}

const char* environment(const char* name) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe): tessiq never writes the environment.
  return std::getenv(name);
}

/// Reads the whole file by size rather than through a stream iterator.
///
/// GCC 13 at -O2 reports a false null dereference inside libstdc++ for the iterator form.
Result<Cubin> read_file(const std::filesystem::path& path, std::string architecture) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return not_found("no cached cubin at " + path.string());
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    return not_found("cached cubin at " + path.string() + " is empty");
  }
  file.seekg(0, std::ios::beg);

  Cubin cubin;
  cubin.architecture = std::move(architecture);
  cubin.data.resize(static_cast<std::size_t>(size));
  if (!file.read(cubin.data.data(), size)) {
    return not_found("could not read the cached cubin at " + path.string());
  }
  return cubin;
}

/// Writes through a temporary and renames, so a reader never sees a half-written file.
///
/// Two processes may compile the same kernel at once. Rename is atomic within a directory, so a
/// reader sees either no file or a whole one.
bool write_file(const std::filesystem::path& path, const std::vector<char>& data) {
  std::error_code code;
  std::filesystem::create_directories(path.parent_path(), code);
  if (code) {
    return false;
  }

  const std::filesystem::path temporary = path.string() + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream file(temporary, std::ios::binary);
    if (!file) {
      return false;
    }
    file.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!file) {
      return false;
    }
  }

  std::filesystem::rename(temporary, path, code);
  if (code) {
    std::filesystem::remove(temporary, code);
    return false;
  }
  return true;
}

} // namespace

std::filesystem::path cubin_cache_dir() {
  if (const char* explicit_dir = environment("TESSIQ_CACHE_DIR"); explicit_dir != nullptr) {
    return std::filesystem::path(explicit_dir) / "cubin";
  }
  if (const char* xdg = environment("XDG_CACHE_HOME"); xdg != nullptr) {
    return std::filesystem::path(xdg) / "tessiq" / "cubin";
  }
  if (const char* home = environment("HOME"); home != nullptr) {
    return std::filesystem::path(home) / ".cache" / "tessiq" / "cubin";
  }
  return std::filesystem::temp_directory_path() / "tessiq" / "cubin";
}

Result<Cubin> compile_cached(std::string_view source, std::string_view name, int major, int minor) {
  const std::string architecture = "sm_" + std::to_string(major) + std::to_string(minor);
  const std::filesystem::path path =
      cubin_cache_dir() / (digest(source, architecture, nvrtc_version()) + ".cubin");

  if (auto cached = read_file(path, architecture); cached) {
    const std::scoped_lock guard(stats_mutex());
    stats().hits++;
    TESSIQ_LOG(kDebug, kLogModule) << name << ": reused " << path.filename().string();
    return cached;
  }

  TESSIQ_ASSIGN_OR_RETURN(Cubin cubin, compile_cuda(source, name, major, minor));

  const bool written = write_file(path, cubin.data);
  {
    const std::scoped_lock guard(stats_mutex());
    stats().misses++;
    if (!written) {
      stats().write_failures++;
    }
  }
  if (!written) {
    // Otherwise every run would recompile without saying why.
    TESSIQ_LOG(kWarn, kLogModule) << "could not write " << path.string()
                                  << "; kernels will be recompiled every run";
  }
  return cubin;
}

CubinCacheStats cubin_cache_stats() {
  const std::scoped_lock guard(stats_mutex());
  return stats();
}

void reset_cubin_cache_stats() {
  const std::scoped_lock guard(stats_mutex());
  stats() = CubinCacheStats{};
}

} // namespace tessiq::codegen
