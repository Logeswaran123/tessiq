# ADR 0001: Result<T> for recoverable failures

- **Status:** accepted
- **Date:** 2026-09-17

## Context

Plenty of paths in a compiler fail as a matter of routine. Kernel lookup finds nothing for a dtype.
A pattern doesn't match. A cache probe misses. An op can't be lowered. None of that is exceptional;
it's the normal answer, and it usually drives a fallback decision.

## Options

**Exceptions.** Idiomatic, no ceremony at the call site, propagate on their own.

The problem is that the signature stops telling you anything. A caller can't see that a function
might need a fallback, so the decision gets made by whoever remembers. Pattern matching also runs
thousands of times per compilation with "no match" as the expected result, and paying for a throw
on the expected path is backwards.

**Sentinels** — `nullptr`, `-1`, `std::optional`. Cheap, but they throw away the reason. "No kernel
exists" and "a kernel exists but not for f64" lead somewhere different, and `nullopt` can't tell
them apart. Easy to ignore, too.

**Result<T>.** Chosen.

## Decision

Recoverable failures return `Result<T>`. It's `[[nodiscard]]`, so dropping one is a compile error.
`Error` carries an `ErrorCode` to branch on and a message for whoever reads the log;
`with_context()` adds provenance on the way up.

Exceptions live in exactly one file, `csrc/bindings/module.cpp`, where an `Error` becomes the
Python exception a caller would expect. Nothing throws across the C ABI.

Broken invariants are a separate thing again. Those are `TESSIQ_DCHECK`, because there is nothing
useful a caller can do with "the compiler has a bug".

## Consequences

Fallibility shows up in every signature, and the compiler enforces that results get checked.
Classification survives propagation, so fallback logic branches on `code()` instead of grepping
message text.

The cost is ceremony: every fallible call needs a check or a macro. `TESSIQ_RETURN_IF_ERROR` and
`TESSIQ_ASSIGN_OR_RETURN` take the edge off but don't remove it. And nothing stops someone mixing
the two mechanisms by accident — that's a review question, not a compiler-enforced one.

Worth revisiting once `std::expected` is available across the toolchains we support, at which point
`Result<T>` should become an alias rather than its own implementation.

## See also

`csrc/tessiq/support/result.h`, `csrc/tessiq/support/error.h`, [PLAN.md](../../PLAN.md) §7.2.
