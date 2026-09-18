// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace tessiq {

/// Classification of a recoverable failure. Each code must mean something different to a caller
/// deciding what to do next.
enum class ErrorCode {
  kInvalidArgument,
  /// Well-formed, but this target cannot serve it. Triggers fallback rather than failure.
  kUnsupported,
  /// A lookup found nothing. Unlike kUnsupported, the thing may exist elsewhere.
  kNotFound,
  kResourceExhausted,
  /// An external component (NVRTC, a driver API, the filesystem) reported failure.
  kExternal,
  /// A bug in tessiq reached a path that could not assert.
  kInternal,
};

/// Returns a lowercase name for `code`, stable across releases: these appear in user-facing
/// messages and in test expectations.
std::string_view error_code_name(ErrorCode code);

/// A recoverable failure. Messages state what failed, the offending value, and what was
/// expected -- see PLAN.md section 7.2.
class Error {
 public:
  Error(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }

  [[nodiscard]] const std::string& message() const noexcept { return message_; }

  /// Formats as "code_name: message".
  [[nodiscard]] std::string to_string() const;

  /// Returns a copy with `context` prefixed, preserving the code. Nesting reads outermost-first:
  /// "lowering node 'add_3': unsupported dtype f64 ...".
  [[nodiscard]] Error with_context(std::string_view context) const;

 private:
  ErrorCode code_;
  std::string message_;
};

inline Error invalid_argument(std::string message) {
  return {ErrorCode::kInvalidArgument, std::move(message)};
}

inline Error unsupported(std::string message) {
  return {ErrorCode::kUnsupported, std::move(message)};
}

inline Error not_found(std::string message) { return {ErrorCode::kNotFound, std::move(message)}; }

inline Error resource_exhausted(std::string message) {
  return {ErrorCode::kResourceExhausted, std::move(message)};
}

inline Error external_error(std::string message) {
  return {ErrorCode::kExternal, std::move(message)};
}

inline Error internal_error(std::string message) {
  return {ErrorCode::kInternal, std::move(message)};
}

} // namespace tessiq
