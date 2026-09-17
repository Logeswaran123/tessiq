// Copyright 2026 The Tessera Authors.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tessera::detail {

[[noreturn]] void
check_failed(const char* file, int line, const char* condition, const std::string& message);

} // namespace tessera::detail

#if defined(__GNUC__) || defined(__clang__)
#define TESSERA_LIKELY(x) __builtin_expect(!!(x), 1)
#define TESSERA_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define TESSERA_LIKELY(x) (x)
#define TESSERA_UNLIKELY(x) (x)
#endif

/// Aborts unless `condition` holds. Retained in release builds.
///
/// For internal invariants only. User input is validated with Result<T> (support/result.h) --
/// a crash is never an acceptable response to bad input. See PLAN.md section 7.2.
#define TESSERA_CHECK(condition, message)                                                          \
  do {                                                                                             \
    if (TESSERA_UNLIKELY(!(condition))) {                                                          \
      ::tessera::detail::check_failed(__FILE__, __LINE__, #condition, (message));                  \
    }                                                                                              \
  } while (false)

/// TESSERA_CHECK, compiled out when NDEBUG is defined.
///
/// The condition is still parsed in release builds, so a stale DCHECK fails to compile rather
/// than rotting unnoticed.
#ifdef NDEBUG
#define TESSERA_DCHECK(condition, message)                                                         \
  do {                                                                                             \
    if (false) {                                                                                   \
      static_cast<void>(condition);                                                                \
    }                                                                                              \
  } while (false)
#else
#define TESSERA_DCHECK(condition, message) TESSERA_CHECK(condition, message)
#endif

#define TESSERA_UNREACHABLE(message)                                                               \
  ::tessera::detail::check_failed(__FILE__, __LINE__, "unreachable", (message))

// NOLINTBEGIN(bugprone-macro-parentheses): ClassName names a type in a constructor declaration,
// where parenthesising it is not valid syntax.

/// Place in the public section of a class body.
#define TESSERA_DISALLOW_COPY(ClassName)                                                           \
  ClassName(const ClassName&) = delete;                                                            \
  ClassName& operator=(const ClassName&) = delete

/// For types pinned by identity, such as registries.
#define TESSERA_DISALLOW_COPY_AND_MOVE(ClassName)                                                  \
  TESSERA_DISALLOW_COPY(ClassName);                                                                \
  ClassName(ClassName&&) = delete;                                                                 \
  ClassName& operator=(ClassName&&) = delete

// NOLINTEND(bugprone-macro-parentheses)

#define TESSERA_UNUSED(x) static_cast<void>(x)
