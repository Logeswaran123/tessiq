// Copyright 2026 The Tessera Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>
#include <sstream>
#include <string_view>

#include "tessera/support/macros.h"

namespace tessera {

/// Severity levels, ordered. A level is emitted when it is at or below the configured level.
enum class LogLevel {
  kError = 0,
  kWarn = 1,
  kInfo = 2,
  kDebug = 3,
  kTrace = 4,
};

/// Returns the lowercase name of `level`, stable across releases.
std::string_view log_level_name(LogLevel level);

/// Parses a level name, case-insensitively. Returns false and leaves `out` untouched on failure.
bool parse_log_level(std::string_view name, LogLevel* out);

/// Returns true if `level` from `module` would be emitted. Thread-safe: configuration is parsed
/// once from TESSERA_LOG_LEVEL and TESSERA_LOG_MODULES, then never mutated.
bool log_enabled(LogLevel level, std::string_view module);

/// Thread-safe, but a message straddling the change may be filtered by either level.
void set_log_level(LogLevel level);

LogLevel get_log_level();

namespace detail {

/// Accumulates one log message and emits it on destruction, so that concurrent messages do not
/// interleave. Each instance is a function-local temporary; the final write is serialised.
class LogMessage {
 public:
  LogMessage(LogLevel level, std::string_view module, const char* file, int line);

  ~LogMessage();

  TESSERA_DISALLOW_COPY_AND_MOVE(LogMessage);

  std::ostream& stream() { return stream_; }

 private:
  LogLevel level_;
  std::string_view module_;
  const char* file_;
  int line_;
  std::ostringstream stream_;
};

} // namespace detail

} // namespace tessera

/// Logs at `level` from `module`, if enabled.
///
///   TESSERA_LOG(kDebug, "fusion") << "merged " << count << " nodes";
///
/// The level test precedes argument evaluation, so a disabled statement costs nothing. Do not
/// restructure this into a function taking a formatted string -- that is the whole point.
#define TESSERA_LOG(level, module)                                                                 \
  if (!::tessera::log_enabled(::tessera::LogLevel::level, (module))) {                             \
  } else                                                                                           \
    ::tessera::detail::LogMessage(::tessera::LogLevel::level, (module), __FILE__, __LINE__).stream()
