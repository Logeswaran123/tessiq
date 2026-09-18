# Contributing

The rules here are enforced by CI. Nothing in this file is a preference you can talk your way past
in review; if a gate rejects your change, the change is what moves.

## Setting up

```bash
pip install -r requirements-dev.txt        # pinned; CI installs the same versions
pip install -e . --no-build-isolation
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
```

Python 3.10+, CMake 3.26+. The extension links libtorch, so torch is required to import tessiq.
The C++ core does not need it:

```bash
cmake -S . -B build -G Ninja -DTESSIQ_BUILD_PYTHON=OFF
```

## Running what CI runs

```bash
pytest                                     # Python tests
cmake -S . -B build -G Ninja && cmake --build build -j8 && ctest --test-dir build
ruff check . && ruff format --check . && mypy
clang-format --dry-run --Werror $(git ls-files 'csrc/*.cpp' 'csrc/*.h' 'tests/cpp/*.cpp')
for f in $(git ls-files 'csrc/*.cpp' 'tests/cpp/*.cpp'); do clang-tidy -p build "$f"; done
```

Sanitizers, which CI runs on the CPU backend:

```bash
cmake -S . -B build-asan -G Ninja -DTESSIQ_BUILD_PYTHON=OFF -DTESSIQ_SANITIZER=address,undefined
cmake --build build-asan -j8
ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build-asan
```

Two things that will waste an afternoon if nobody tells you:

- **Build with the compiler CI uses, not only yours.** CI is on g++ 13 and warns about things
  older compilers do not. `conda install -c conda-forge gxx_linux-64=13`, then configure with
  `-DCMAKE_CXX_COMPILER=$(which x86_64-conda-linux-gnu-g++)`. Every CI-only failure so far has been
  a diagnostic the newer compiler emits.
- **On some kernels the ASan build hangs at startup** on ASLR entropy. Run it under
  `setarch $(uname -m) -R` and it passes.

## Definition of done

Not "it works on my machine". All of these:

- [ ] Error handling on every failure path
- [ ] Tested, including the failure paths and at least one negative case
- [ ] Public API contracts documented; no comment that restates the code
- [ ] No new warnings, lint errors or type errors
- [ ] Logging at the right level
- [ ] Benchmarked if it touches a hot path, with the number in the PR
- [ ] Checked against the CPU reference if it touches the CUDA path
- [ ] CHANGELOG updated if user-visible

## C++

C++17 in the core. `csrc/tessiq/torch/` is C++20 because recent torch headers use
`std::strong_ordering` and `requires` clauses and will not parse otherwise. The split is per
target; do not widen it.

- **RAII everywhere.** No raw `new`/`delete` outside a resource wrapper. Every device allocation,
  stream and compiled program is owned by a type whose destructor releases it.
- **Ownership is in the type.** `unique_ptr` for sole ownership, raw pointer or reference for
  non-owning views. A `shared_ptr` in a signature needs justifying in review.
- **`const`-correct.** Analysis passes take `const Graph&`. A pass that mutates says so in its
  signature, which is how a reader knows what it does before reading it.
- **No global mutable state.** Registries are function-local statics; everything else is passed.
- **Headers stay minimal.** Implementation details go in `detail::` namespaces in `.cpp` files.

### Error handling

Three categories, one mechanism each. Mixing them is how error handling rots.

| Category | Mechanism | Example |
|---|---|---|
| Programmer error, broken invariant | `TESSIQ_DCHECK`, debug only | An IR node with a null operand after a pass |
| Recoverable failure | `Result<T>`, `[[nodiscard]]` | No kernel for this dtype, so fall back |
| User error at the API boundary | Exception, translated in the bindings | An unsupported dtype from Python |

- **Never assert on user input.** Validate at the boundary and reject with a message naming the
  offending value and what was expected.
- **No silent failures.** A pass either transforms a node or leaves it alone. Never drops it,
  never half-transforms it.
- **Errors name the fix.** `unsupported dtype f64 for kernel 'add' on cuda backend; supported:
  f32, f16, bf16`, not `unsupported dtype`.
- **Exceptions never cross the C ABI boundary.** They are caught and translated in
  `csrc/bindings/`. See `docs/design_notes/0001-result-type-over-exceptions.md` for why.

### Logging and concurrency

Log through `TESSIQ_LOG`, which does not evaluate its arguments when the level is off. Every pass
logs at `kDebug` what it changed and how many nodes it touched; a compiler you cannot see inside is
a compiler you cannot debug.

Document thread-safety on every class, in the header, as one of: not thread-safe, thread-safe, or
thread-compatible. Undocumented means not thread-safe. Device memory is stream-ordered: a buffer
freed on a stream is not reusable until prior work on that stream completes. That is the classic
source of corruption that reproduces once a week and never under a debugger.

## Python

`ruff` and `mypy --strict` both gate merges. Public functions are annotated and have docstrings
saying what they do, what they raise, and anything a caller cannot infer from the signature.

The native module is typed by a hand-written stub, `python/tessiq/_C.pyi`. It does not update
itself. If you add to `csrc/bindings/module.cpp`, update the stub in the same commit. mypy will
catch you, but only for code that calls the new function.

## Tests

Every pass gets golden tests: textual IR in, textual IR out. A pass test should show both a case it
fires on and a case it correctly refuses. CUDA results are checked against the CPU reference, which
is never optimised. Its value is being obviously correct.

CUDA tests are marked `@pytest.mark.cuda` and deselected in CI, which has no GPU. Run them locally
with `pytest -m cuda`.

## Comments

**Comment sparingly. Most code should carry none.** Comments that restate the code go stale the
moment it changes, and a file annotated line by line teaches the reader to skip all of it, including
the three that mattered.

Write one only when a competent reader would otherwise be misled or stuck. If the answer is on the
line below, delete it. A clearer name or a smaller function beats a comment every time.

What earns one: a non-obvious *why*; a contract a signature cannot express (ownership, lifetime,
thread-safety, preconditions); a pass's legality argument; a trap where correct code looks wrong.

What does not: restating the signature; banner dividers; rationale that belongs in a design note;
narrating a change, which is what the commit message is for.

## Commit messages

Conventional commits: `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `build:`, `ci:`,
`chore:`, `revert:`. Imperative mood.

**Subject under 50 characters, always.** It says what changed, never how well you did it.

**Body under five lines for routine work**, and often absent.

**A new feature or a subtle bug fix earns up to about fifteen**, for what the next person cannot
reconstruct from the diff:

- A feature: the design decision, and what was rejected.
- A bug fix: the failure mode. What went wrong, why it was not caught, what stops it returning. A
  fix whose cause is not recorded is one somebody reverts later.
- Either: a constraint that forced your hand, if it is not visible in the code.

Not in the body: evidence, metrics, search results, a list of what you verified, or a tour of the
files you touched. That goes in the PR. If the reasoning needs paragraphs, it is a design decision
and belongs in `docs/design_notes/`.

## Pull requests

Keep a pull request small enough that someone can review it properly. A phase of work should
arrive as several pull requests rather than one.

Split it along the seams the work already has. Build the data structure first. Then add the code
that fills it. Then add the code that consumes it. Each pull request should land working and tested
on its own.

Branches are named `dev-<major>.<phase>`. When a phase takes several pull requests, number them
`dev-0.4.1`, `dev-0.4.2` and so on.

The description should carry what a reviewer needs in order to judge the change. Say what it does.
Explain the decisions that the diff does not make obvious. Point out anything that deserves close
attention. State what you left out on purpose.

If the description runs to pages, most of it is background rather than review material. Keep that
somewhere else and link to it.

## Performance

**No performance claim without a script in `bench/` that reproduces it.** Report a distribution,
not one timing, with warmup excluded.

**Correctness gates performance.** A faster kernel that fails the accuracy test against the CPU
reference is not an optimisation, it is a bug. No tolerances chosen after seeing the failure.

## Dependencies

Every pin is exact, in `requirements-dev.txt`, and CI installs from that file so a hook cannot
disagree with the gate it mirrors. Bumping a pin is its own commit, with the reformatting or fixes
it causes.

torch is pinned to the last release carrying `cu128` wheels. Moving past it means moving the CUDA
toolkit too, so that is a deliberate change, not a routine upgrade.
