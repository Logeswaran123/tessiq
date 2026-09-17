# ADR 0001: `Result<T>` for recoverable failures, exceptions only at the boundary

- **Status:** accepted
- **Date:** 2026-09-17
- **Phase:** 0

## Context

The compiler needs a failure mechanism for paths that fail routinely: kernel lookup that finds no
implementation for a dtype, a pattern match that does not apply, a cache probe that misses, an op
the backend cannot lower. These are not exceptional conditions. They are ordinary answers that
drive a decision — usually "fall back to CPU" or "try the next pattern".

Three options were considered.

## Options

**1. Exceptions throughout.** Idiomatic C++, no return-type ceremony, and errors propagate without
each layer handling them.

Rejected because it makes the common case invisible. A function signature says nothing about
whether it can fail, so a caller cannot tell from the interface that a fallback decision is
required. It also costs on hot paths — pattern matching runs thousands of times per compilation,
and "no match" is the expected outcome for most patterns tried.

**2. Sentinel returns** (`nullptr`, `-1`, `std::optional`). Cheap and simple.

Rejected because it discards the reason. "No kernel" and "kernel exists but this dtype is not
supported" lead to different decisions, and a bare `std::nullopt` cannot tell them apart. It is
also silently ignorable.

**3. `Result<T>` carrying a value or an `Error`.** Chosen.

## Decision

Recoverable failures return `Result<T>`. `Result` is `[[nodiscard]]`, so ignoring a fallible call
is a compile error. `Error` carries an `ErrorCode` for programmatic decisions and a message for
humans, and `with_context()` adds provenance while propagating.

Exceptions are confined to one place: `csrc/bindings/module.cpp`, where `Error` is translated into
the Python exception type a caller would expect. No C++ exception crosses the C ABI boundary.

Broken internal invariants are a third, separate category. They are not `Result` — they are
`TESSERA_DCHECK`, because a caller cannot do anything useful with "the compiler has a bug".

## Consequences

**Good.** Fallibility is visible in every signature. The compiler enforces that results are
checked. Error classification survives propagation, so the fallback logic can branch on `code()`
rather than on message text. No exception-handling cost on hot paths.

**Bad.** More ceremony than exceptions: each fallible call needs a check or a propagation macro.
`TESSERA_RETURN_IF_ERROR` and `TESSERA_ASSIGN_OR_RETURN` reduce it but do not remove it. The two
mechanisms must not be mixed by accident, which is a discipline question, not a compiler-enforced
one.

**Revisit if** C++23's `std::expected` becomes available across the supported toolchains, at which
point `Result<T>` should become a thin alias rather than an independent implementation.

## References

- `csrc/tessera/support/result.h`, `csrc/tessera/support/error.h`
- `PLAN.md` §7.2 — the three-category error model
