# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CMake build, scikit-build-core packaging, pybind11 extension module.
- Support layer: `TESSERA_CHECK`/`TESSERA_DCHECK`, `Error`/`ErrorCode`, `Result<T>` with
  `TESSERA_RETURN_IF_ERROR` and `TESSERA_ASSIGN_OR_RETURN`, and module-scoped logging that skips
  argument evaluation when the level is off.
- `ErrorCode` to Python exception mapping in the bindings, so callers can use `except` rather than
  matching on message text.
- Quality gates: `-Werror`, clang-format, clang-tidy, ruff, mypy, ASan/UBSan, and CI running all
  of them.
- `requirements-dev.txt` pinning the tooling. CI and the pre-commit hooks install from it, which
  stops a hook disagreeing with the gate it mirrors.
- Device runtime: a `Device` interface with a CPU reference backend, and a caching allocator with
  size classes, reuse, statistics and `empty_cache()`.
- Registration with PyTorch as the `tessera` device. `torch.randn(n, device="tessera")`,
  `.to("tessera")` and `.cpu()` work, including for non-contiguous tensors and across dtypes.
- A CPU fallback for ops without a native kernel, so the whole ATen surface is usable now and
  kernels can replace it one at a time.
- `tessera.memory_stats()`, `memory_allocated()`, `memory_reserved()` and `empty_cache()`.
