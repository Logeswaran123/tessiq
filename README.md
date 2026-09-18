# tessera

A PyTorch accelerator backend built from scratch: a custom device, a graph compiler, and a kernel
library. It exists to answer, in working code, what a hardware vendor actually has to build to put
a chip behind `model.to(...)` and `torch.compile`.

Phase 0 of 9. The build, the test harness and the support layer are in. There is no device, no IR
and no passes yet. [PLAN.md](PLAN.md) has the design and the phase breakdown.

## How it fits together

```
  torch.compile(model, backend="tessera")      model.to("tessera")
              │                                        │
              v                                        v
   Dynamo → FX → AOTAutograd                   aten op → dispatch
              │                                        │
              v                                        │
       Tessera IR ──> passes ──> lowering ──> recipe   │
              │        canonicalize                    │
              │        shape infer                     │
              │        layout                          │
              │        fuse                            │
              │        schedule                        │
              │        memory plan                     │
              v                                        v
        ┌─────────────── kernel registry ───────────────┐
        │  CPU reference              CUDA + NVRTC      │
        └───────────────────────────────────────────────┘
```

Two ways in. `torch.compile` hands over a whole graph, which is where the compiler earns its keep.
`.to("tessera")` dispatches one op at a time and catches whatever the compiler declines to take.

The CPU backend is there to be obviously right rather than fast; CUDA output is checked against it.
The passes don't know what hardware they're targeting, so CI runs the compiler on a GPU-less runner.

## Build

```bash
pip install -r requirements-dev.txt
pip install -e .
pytest

cmake -S . -B build -G Ninja
cmake --build build -j8
ctest --test-dir build
```

Python 3.10+, CMake 3.26+, C++17. PyTorch is only needed for the frontend, not to build or test the
compiler core.

If you're going to commit:

```bash
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
```

## Layout

| Path | |
|---|---|
| [`csrc/`](csrc/) | C++/CUDA core: runtime, IR, passes, lowering, codegen, kernels |
| [`python/tessera/`](python/tessera/) | Python API and the `torch.compile` backend |
| [`ops/`](ops/) | Op definitions in YAML, and the source of truth for codegen (phase 6) |
| [`tools/`](tools/) | `gen_ops.py`, `tessera-opt`, `tessera-explain` (phases 2, 6, 7) |
| [`tests/`](tests/) | gtest suites and golden pass tests; Python accuracy and end-to-end tests |
| [`bench/`](bench/) | Benchmarks, and the scripts behind any number published here |
| [`docs/`](docs/) | Architecture, IR spec, pass reference, design notes |

## Standards

`-Werror` with a wide warning set, clang-tidy, ruff, `mypy --strict` and ASan/UBSan all gate merges.
[PLAN.md §7](PLAN.md) has the rules. The parts that come up most: user input never reaches an
assert, every pass states why its transformation is legal, and no performance claim ships without
the script that produced it.

## Benchmarks

Nothing worth showing before phase 5. Numbers land here when they exist, next to the scripts that
produced them.

## Prior art

Built on publicly documented work: PyTorch's out-of-tree backend mechanism and `torch.compile`
internals, `pytorch/xla`, MLIR's pass infrastructure, IREE, TVM. Non-obvious decisions cite their
precedent where they're made.

## License

Apache-2.0.
