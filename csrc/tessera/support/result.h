// Copyright 2026 The Tessera Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#include "tessera/support/error.h"
#include "tessera/support/macros.h"

namespace tessera {

/// Holds either a value of type `T` or an Error explaining why there is none.
///
/// The return type for recoverable failures; see ADR 0001 for why not exceptions. [[nodiscard]]
/// so that ignoring a fallible call is a compile error.
///
///   auto kernel = find_kernel("add", DType::kF32);
///   if (!kernel) {
///     return kernel.error().with_context("lowering node 'add_3'");
///   }
///
/// Not thread-safe: a Result is a value, moved between threads at most once.
template <typename T>
class [[nodiscard]] Result {
  static_assert(!std::is_same_v<std::decay_t<T>, Error>,
                "Result<Error> is ambiguous; use Result<void> or wrap the Error in a struct");
  static_assert(!std::is_reference_v<T>,
                "Result<T&> is not supported; return a pointer or a value");

 public:
  using value_type = T;

  // Implicit by design: `return value;` and `return unsupported("...");` should both just work.
  Result(T value) : storage_(std::move(value)) {} // NOLINT(google-explicit-constructor)

  Result(Error error) : storage_(std::move(error)) {} // NOLINT(google-explicit-constructor)

  [[nodiscard]] bool has_value() const noexcept { return std::holds_alternative<T>(storage_); }

  explicit operator bool() const noexcept { return has_value(); }

  /// Precondition: has_value(). Violating it is a programmer error, not a recoverable one -- the
  /// caller was obliged to check first.
  [[nodiscard]] const T& value() const& {
    TESSERA_DCHECK(has_value(),
                   "Result::value() called on an error Result: " + error().to_string());
    return std::get<T>(storage_);
  }

  T& value() & {
    TESSERA_DCHECK(has_value(),
                   "Result::value() called on an error Result: " + error().to_string());
    return std::get<T>(storage_);
  }

  T&& value() && {
    TESSERA_DCHECK(has_value(),
                   "Result::value() called on an error Result: " + error().to_string());
    return std::get<T>(std::move(storage_));
  }

  const T& operator*() const& { return value(); }

  T& operator*() & { return value(); }

  T&& operator*() && { return std::move(*this).value(); }

  const T* operator->() const { return &value(); }

  T* operator->() { return &value(); }

  /// Precondition: !has_value().
  [[nodiscard]] const Error& error() const& {
    TESSERA_DCHECK(!has_value(), "Result::error() called on a successful Result");
    return std::get<Error>(storage_);
  }

  Error&& error() && {
    TESSERA_DCHECK(!has_value(), "Result::error() called on a successful Result");
    return std::get<Error>(std::move(storage_));
  }

  /// For a genuine default. If the error carries information the caller needs, check has_value().
  [[nodiscard]] T value_or(T fallback) const& {
    return has_value() ? value() : std::move(fallback);
  }

 private:
  std::variant<T, Error> storage_;
};

/// For operations that can fail but yield no value.
template <>
class [[nodiscard]] Result<void> {
 public:
  using value_type = void;

  Result() = default;

  Result(Error error) : error_(std::move(error)) {} // NOLINT(google-explicit-constructor)

  [[nodiscard]] bool has_value() const noexcept { return !error_.has_value(); }

  explicit operator bool() const noexcept { return has_value(); }

  /// Precondition: !has_value().
  [[nodiscard]] const Error& error() const& {
    TESSERA_DCHECK(!has_value(), "Result::error() called on a successful Result");
    // Guarded by the precondition above; re-checking in release would turn a programmer error
    // into a silently different result.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return *error_;
  }

  Error&& error() && {
    TESSERA_DCHECK(!has_value(), "Result::error() called on a successful Result");
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return std::move(*error_);
  }

 private:
  std::optional<Error> error_;
};

inline Result<void> ok() { return {}; }

/// Propagates an error unchanged.
///
/// A statement macro rather than an expression, so propagation is visible when skimming rather
/// than buried inside a larger expression.
#define TESSERA_RETURN_IF_ERROR(expr)                                                              \
  do {                                                                                             \
    auto tessera_result_ = (expr);                                                                 \
    if (TESSERA_UNLIKELY(!tessera_result_.has_value())) {                                          \
      return std::move(tessera_result_).error();                                                   \
    }                                                                                              \
  } while (false)

// `a##__LINE__` pastes the literal token __LINE__; routing through a second macro expands it
// to the line number first.
#define TESSERA_CONCAT_IMPL(a, b) a##b
#define TESSERA_CONCAT(a, b) TESSERA_CONCAT_IMPL(a, b)

/// Assigns the value of a successful Result to `lhs`, or returns its error.
///
/// `lhs` may include a declaration: TESSERA_ASSIGN_OR_RETURN(auto kernel, find_kernel(...));
// NOLINTBEGIN(bugprone-macro-parentheses)
#define TESSERA_ASSIGN_OR_RETURN(lhs, expr)                                                        \
  auto TESSERA_CONCAT(tessera_assign_, __LINE__) = (expr);                                         \
  if (TESSERA_UNLIKELY(!TESSERA_CONCAT(tessera_assign_, __LINE__).has_value())) {                  \
    return std::move(TESSERA_CONCAT(tessera_assign_, __LINE__)).error();                           \
  }                                                                                                \
  lhs = std::move(TESSERA_CONCAT(tessera_assign_, __LINE__)).value()
// NOLINTEND(bugprone-macro-parentheses)

} // namespace tessera
