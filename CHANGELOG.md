# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffolding: CMake build, scikit-build-core packaging, pybind11 extension module.
- Support layer: `TESSERA_CHECK`/`TESSERA_DCHECK` assertions, `Error`/`ErrorCode` values,
  `Result<T>` with `TESSERA_RETURN_IF_ERROR` and `TESSERA_ASSIGN_OR_RETURN`, and leveled
  module-scoped logging that costs nothing when disabled.
- Error translation boundary in the Python bindings, mapping each `ErrorCode` to the Python
  exception type a caller would expect.
- Engineering-standards automation: `-Werror` warning set, clang-format, clang-tidy, ruff, mypy,
  pre-commit hooks, ASan/UBSan build configuration, and CI enforcing all of them.
- `requirements-dev.txt` pinning the tooling, installed by both CI and the pre-commit hooks so
  that a hook cannot disagree with the gate it mirrors.
