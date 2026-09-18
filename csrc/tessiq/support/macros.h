// Copyright 2026 The Tessiq Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tessiq::detail {

[[noreturn]] void
check_failed(const char* file, int line, const char* condition, const std::string& message);

} // namespace tessiq::detail

#if defined(__GNUC__) || defined(__clang__)
#define TESSIQ_LIKELY(x) __builtin_expect(!!(x), 1)
#define TESSIQ_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define TESSIQ_LIKELY(x) (x)
#define TESSIQ_UNLIKELY(x) (x)
#endif

/// Aborts unless `condition` holds. Retained in release builds.
///
/// For internal invariants only. User input is validated with Result<T> (support/result.h) --
/// a crash is never an acceptable response to bad input. See PLAN.md section 7.2.
#define TESSIQ_CHECK(condition, message)                                                           \
  do {                                                                                             \
    if (TESSIQ_UNLIKELY(!(condition))) {                                                           \
      ::tessiq::detail::check_failed(__FILE__, __LINE__, #condition, (message));                   \
    }                                                                                              \
  } while (false)

/// TESSIQ_CHECK, compiled out when NDEBUG is defined.
///
/// The condition is still parsed in release builds, so a stale DCHECK fails to compile rather
/// than rotting unnoticed.
#ifdef NDEBUG
#define TESSIQ_DCHECK(condition, message)                                                          \
  do {                                                                                             \
    if (false) {                                                                                   \
      static_cast<void>(condition);                                                                \
    }                                                                                              \
  } while (false)
#else
#define TESSIQ_DCHECK(condition, message) TESSIQ_CHECK(condition, message)
#endif

#define TESSIQ_UNREACHABLE(message)                                                                \
  ::tessiq::detail::check_failed(__FILE__, __LINE__, "unreachable", (message))

// NOLINTBEGIN(bugprone-macro-parentheses): ClassName names a type in a constructor declaration,
// where parenthesising it is not valid syntax.

/// Place in the public section of a class body.
#define TESSIQ_DISALLOW_COPY(ClassName)                                                            \
  ClassName(const ClassName&) = delete;                                                            \
  ClassName& operator=(const ClassName&) = delete

/// For types pinned by identity, such as registries.
#define TESSIQ_DISALLOW_COPY_AND_MOVE(ClassName)                                                   \
  TESSIQ_DISALLOW_COPY(ClassName);                                                                 \
  ClassName(ClassName&&) = delete;                                                                 \
  ClassName& operator=(ClassName&&) = delete

// NOLINTEND(bugprone-macro-parentheses)

#define TESSIQ_UNUSED(x) static_cast<void>(x)
