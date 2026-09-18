// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>

#include "tessiq/support/error.h"
#include "tessiq/support/logging.h"
#include "tessiq/support/macros.h"

namespace tessiq {

namespace {

/// Returns the basename of a path. __FILE__ is absolute in CMake builds and would otherwise
/// dominate every log line.
std::string_view short_file_name(const char* path) {
  std::string_view view(path);
  const auto pos = view.find_last_of('/');
  return pos == std::string_view::npos ? view : view.substr(pos + 1);
}

std::string to_lower(std::string_view text) {
  std::string lowered(text);
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return lowered;
}

/// Process-wide logging configuration.
///
/// Held in a function-local static so initialisation order is defined: logging may be used during
/// the static initialisation of other translation units.
class LogConfig {
 public:
  static LogConfig& instance() {
    static LogConfig config;
    return config;
  }

  LogLevel level() const { return level_.load(std::memory_order_relaxed); }

  void set_level(LogLevel level) { level_.store(level, std::memory_order_relaxed); }

  /// `modules_` is populated once in the constructor and never mutated, so concurrent reads
  /// need no lock.
  bool module_enabled(std::string_view module) const {
    if (modules_.empty()) {
      return true;
    }
    return modules_.find(std::string(module)) != modules_.end();
  }

  std::mutex& output_mutex() { return output_mutex_; }

 private:
  LogConfig() {
    if (const char* level_env = std::getenv("TESSIQ_LOG_LEVEL")) {
      LogLevel parsed = LogLevel::kWarn;
      if (parse_log_level(level_env, &parsed)) {
        level_.store(parsed, std::memory_order_relaxed);
      } else {
        // Not fatal: a typo in a debugging variable should not stop a run, but must not pass
        // silently either.
        std::cerr << "[tessiq] ignoring invalid TESSIQ_LOG_LEVEL='" << level_env
                  << "'; expected one of: error, warn, info, debug, trace\n";
      }
    }

    if (const char* modules_env = std::getenv("TESSIQ_LOG_MODULES")) {
      std::string_view spec(modules_env);
      size_t start = 0;
      while (start <= spec.size()) {
        const size_t comma = spec.find(',', start);
        const size_t end = comma == std::string_view::npos ? spec.size() : comma;
        std::string_view token = spec.substr(start, end - start);
        if (!token.empty()) {
          modules_.emplace(token);
        }
        if (comma == std::string_view::npos) {
          break;
        }
        start = comma + 1;
      }
    }
  }

  std::atomic<LogLevel> level_{LogLevel::kWarn};
  std::unordered_set<std::string> modules_;
  std::mutex output_mutex_;
};

} // namespace

// --- Assertions -------------------------------------------------------------------------------

namespace detail {

void check_failed(const char* file, int line, const char* condition, const std::string& message) {
  std::cerr << "\n[tessiq] CHECK FAILED at " << short_file_name(file) << ":" << line << "\n"
            << "  condition: " << condition << "\n"
            << "  message:   " << message << "\n"
            << "\nThis is an internal invariant violation, not a usage error. Please report it.\n"
            << std::flush;
  std::abort();
}

} // namespace detail

// --- Errors -----------------------------------------------------------------------------------

std::string_view error_code_name(ErrorCode code) {
  switch (code) {
  case ErrorCode::kInvalidArgument:
    return "invalid_argument";
  case ErrorCode::kUnsupported:
    return "unsupported";
  case ErrorCode::kNotFound:
    return "not_found";
  case ErrorCode::kResourceExhausted:
    return "resource_exhausted";
  case ErrorCode::kExternal:
    return "external";
  case ErrorCode::kInternal:
    return "internal";
  }
  // No default label: adding an ErrorCode without updating this switch is then a compile error.
  TESSIQ_UNREACHABLE("unhandled ErrorCode in error_code_name");
}

std::string Error::to_string() const {
  return std::string(error_code_name(code_)) + ": " + message_;
}

Error Error::with_context(std::string_view context) const {
  return {code_, std::string(context) + ": " + message_};
}

// --- Logging ----------------------------------------------------------------------------------

std::string_view log_level_name(LogLevel level) {
  switch (level) {
  case LogLevel::kError:
    return "error";
  case LogLevel::kWarn:
    return "warn";
  case LogLevel::kInfo:
    return "info";
  case LogLevel::kDebug:
    return "debug";
  case LogLevel::kTrace:
    return "trace";
  }
  TESSIQ_UNREACHABLE("unhandled LogLevel in log_level_name");
}

bool parse_log_level(std::string_view name, LogLevel* out) {
  TESSIQ_DCHECK(out != nullptr, "parse_log_level requires a non-null out parameter");
  const std::string lowered = to_lower(name);
  for (const LogLevel level :
       {LogLevel::kError, LogLevel::kWarn, LogLevel::kInfo, LogLevel::kDebug, LogLevel::kTrace}) {
    if (lowered == log_level_name(level)) {
      *out = level;
      return true;
    }
  }
  return false;
}

bool log_enabled(LogLevel level, std::string_view module) {
  const LogConfig& config = LogConfig::instance();
  return static_cast<int>(level) <= static_cast<int>(config.level()) &&
         config.module_enabled(module);
}

void set_log_level(LogLevel level) { LogConfig::instance().set_level(level); }

LogLevel get_log_level() { return LogConfig::instance().level(); }

namespace detail {

LogMessage::LogMessage(LogLevel level, std::string_view module, const char* file, int line)
    : level_(level), module_(module), file_(file), line_(line) {}

LogMessage::~LogMessage() {
  std::ostringstream line;
  line << "[tessiq:" << log_level_name(level_) << ":" << module_ << "] " << stream_.str() << " ("
       << short_file_name(file_) << ":" << line_ << ")\n";

  const std::scoped_lock guard(LogConfig::instance().output_mutex());
  std::cerr << line.str() << std::flush;
}

} // namespace detail

} // namespace tessiq
