// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>
#include <sstream>
#include <string_view>

#include "tessiq/support/macros.h"

namespace tessiq {

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
/// once from TESSIQ_LOG_LEVEL and TESSIQ_LOG_MODULES, then never mutated.
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

  TESSIQ_DISALLOW_COPY_AND_MOVE(LogMessage);

  std::ostream& stream() { return stream_; }

 private:
  LogLevel level_;
  std::string_view module_;
  const char* file_;
  int line_;
  std::ostringstream stream_;
};

} // namespace detail

} // namespace tessiq

/// Logs at `level` from `module`, if enabled.
///
///   TESSIQ_LOG(kDebug, "fusion") << "merged " << count << " nodes";
///
/// The level is tested before the arguments are evaluated, so a disabled statement costs nothing.
/// That is why this is a macro and not a function taking a formatted string.
#define TESSIQ_LOG(level, module)                                                                  \
  if (!::tessiq::log_enabled(::tessiq::LogLevel::level, (module))) {                               \
  } else                                                                                           \
    ::tessiq::detail::LogMessage(::tessiq::LogLevel::level, (module), __FILE__, __LINE__).stream()
