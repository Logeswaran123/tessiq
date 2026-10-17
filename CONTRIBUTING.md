# Contributing

CI enforces everything below. A change that fails a check doesn't merge.

## Setup

```bash
pip install -r requirements-dev.txt        # exact pins, same as CI
pip install -e . --no-build-isolation
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
```

Python 3.10+ and CMake 3.26+. The C++ core builds without PyTorch:
`cmake -S . -B build -G Ninja -DTESSIQ_BUILD_PYTHON=OFF`.

## Checks

CI runs on GPU-less GitHub runners. To run the same locally:

```bash
ruff check . && ruff format --check . && mypy

cmake --preset dev && cmake --build --preset dev
ctest --preset dev
pytest -m "not cuda"          # drop -m to include GPU tests

python tools/lint_cpp.py --build-dir build/dev   # clang-format and clang-tidy
```

`CMakePresets.json` defines `dev` (Debug), `release` (RelWithDebInfo) and `sanitizers`, each built in
`build/<preset>`. Local variations go in an untracked `CMakeUserPresets.json`.

Sanitizers, on the CPU backend:

```bash
cmake --preset sanitizers && cmake --build --preset sanitizers
ASAN_OPTIONS=detect_leaks=1 ctest --preset sanitizers
```

On a machine with a GPU, use `detect_leaks=0`; the CUDA driver's own allocations show up as leaks.

- CI builds with GCC 13, which warns about things older compilers don't. With conda:
  `conda install -c conda-forge gxx_linux-64=13`, then configure with
  `-DCMAKE_CXX_COMPILER=$(which x86_64-conda-linux-gnu-g++)`.
- If the sanitizer build hangs at startup, run it under `setarch $(uname -m) -R`.

## Before sending a change

- [ ] Failure paths are handled and tested, including at least one case that must be rejected.
- [ ] Public API contracts are documented.
- [ ] No new warnings, lint or type errors.
- [ ] Hot-path changes are benchmarked, with numbers in the PR.
- [ ] CUDA changes are checked against the CPU reference.
- [ ] `CHANGELOG.md` is updated for user-visible changes.

## C++

The core (`support`, `runtime`, `ir`, `passes`, `codegen`) is C++17. `csrc/tessiq/torch/` and the
bindings are C++20 because recent PyTorch headers need it. Keep the core on C++17.

- RAII for every allocation, stream, handle and compiled program. No naked `new`/`delete`.
- `unique_ptr` for ownership, raw pointers and references for non-owning access. A `shared_ptr`
  in a signature needs a reason.
- Analysis passes take `const Graph&`.
- No global mutable state; registries are function-local statics.
- Keep headers minimal.

Errors:

| | Mechanism |
|---|---|
| Broken invariant | `TESSIQ_DCHECK` (debug builds) |
| Recoverable failure, e.g. no kernel for a dtype | `Result<T>`, `[[nodiscard]]` |
| Bad input from Python | `Result<T>` error, raised as a Python exception by the bindings |

Never assert on user input. Error messages say what was wrong and what would work:

```
no custom operation has a rewrite called 'layer_norm'; these do: fast_gelu, gemm_bias_gelu, gemm_bias_gelu_tanh, rms_norm
```

A pass either transforms a node completely or leaves it alone. The core doesn't throw; the
PyTorch integration uses `TORCH_CHECK` where PyTorch expects it. See
[design note 0001](docs/design_notes/0001-result-type-over-exceptions.md).

Log with `TESSIQ_LOG(level, module)`, which skips argument evaluation when disabled. Passes log what
they changed at `kDebug`. At run time, `TESSIQ_LOG_LEVEL` and `TESSIQ_LOG_MODULES` control output.

Document every class's thread safety in its header: not thread-safe, thread-safe, or
thread-compatible. Undocumented means not thread-safe. Device memory is stream-ordered: don't reuse
a block until the work that used it has finished.

## Python

`ruff` and `mypy --strict`. Public functions have type annotations and docstrings covering what
they raise. `python/tessiq/_C.pyi` is hand-written; update it with `csrc/bindings/module.cpp`.

## Tests

- Every pass has golden tests in `tests/cpp/golden/`: a `// RUN:` pipeline, input IR, and the
  expected output. The C++ suite picks up every file. Include a case the pass must leave alone.
  Rewrite passes need the Python extension, so theirs are in `tests/python/golden/`.
- GPU results are compared against the CPU reference device or eager PyTorch on the CPU.
- GPU tests are marked `@pytest.mark.cuda`; CI skips them.

## Comments

Most code needs none. Comment the non-obvious *why*, contracts the signature can't express
(ownership, lifetime, thread safety, preconditions), a pass's legality argument, and code that is
correct but looks wrong. Don't restate the code or narrate the change.

## Commits and pull requests

[Conventional Commits](https://www.conventionalcommits.org/): `feat`, `fix`, `perf`, `refactor`,
`docs`, `test`, `build`, `ci`, `chore`, `revert`. Imperative subject under 50 characters. Routine
changes need little or no body. Features and non-trivial fixes get up to about 15 lines on what the
diff doesn't show: the design choice and what was rejected, or the failure mode and why it wasn't
caught. Measurements go in the PR; longer reasoning goes in `docs/design_notes/`.

Keep PRs reviewable. Split larger work into PRs that each land working and tested. Branches are
`dev-<major>.<stage>`, with `dev-0.13.1`, `dev-0.13.2` and so on for multi-PR stages. The
description says what changed, why where it isn't obvious, what needs a careful look, and what was
left out on purpose.

## Performance and dependencies

Every performance claim needs a script in `bench/`. Report a median or a distribution, without
warmup. A faster kernel that fails its accuracy test is a bug, and tolerances are fixed before
looking at results.

Tools are pinned exactly in `requirements-dev.txt`; bump them in their own commit. PyTorch is
pinned to 2.11.0, the last release with CUDA 12.8 wheels, so moving past it means moving the CUDA
Toolkit too.
