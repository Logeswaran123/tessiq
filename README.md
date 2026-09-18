# tessiq

A PyTorch accelerator backend built from scratch: a custom device, a graph compiler, and a kernel
library. It exists to answer, in working code, what a hardware vendor actually has to build to put
a chip behind `model.to(...)` and `torch.compile`.

Named after *tessera*, the Latin word for a mosaic tile, which is what the compiler makes of a
graph: pieces cut to fit, then set into one surface. Said `/ˈtɛs.ɪk/`.

Phase 1 of 10 is done. Tensors allocate, transfer and compute under `device="tessiq"`, backed by
a caching allocator, with a CPU fallback covering every op that has no kernel yet.

**Today that device runs on the CPU.** Its memory is host memory and its arithmetic is PyTorch's
own CPU kernels; no GPU is involved. What is real is the rest of it: PyTorch treats tessiq as a
separate device with its own dispatch key, allocator and memory accounting. The GPU backend, and
picking between the two per tensor, is phase 2. No IR and no compiler passes yet.
[PHASES.md](PHASES.md) has the full phase breakdown and what each one delivers.

```python
import torch, tessiq

x = torch.randn(1024, device="tessiq")
y = (x + x).cpu()
tessiq.memory_stats()
```

From phase 2 the backend is chosen per tensor rather than per process:

```python
reference = torch.randn(1024, device=tessiq.cpu)  # host memory
fast = reference.to(tessiq.gpu)  # GPU memory
```

## How it fits together

```
  torch.compile(model, backend="tessiq")      model.to("tessiq")
              │                                        │
              v                                        v
   Dynamo → FX → AOTAutograd                   aten op → dispatch
              │                                        │
              v                                        │
       Tessiq IR ──> passes ──> lowering ──> recipe   │
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
`.to("tessiq")` dispatches one op at a time and catches whatever the compiler declines to take.

The CPU backend is there to be obviously right rather than fast; CUDA output is checked against it.
The passes don't know what hardware they're targeting, so CI runs the compiler on a GPU-less runner.

## Build

```bash
pip install -r requirements-dev.txt
pip install -e . --no-build-isolation
pytest

cmake -S . -B build -G Ninja
cmake --build build -j8
ctest --test-dir build
```

Python 3.10+, CMake 3.26+, and a compiler with C++20 for the torch-facing code (the core is
C++17). The extension links libtorch, so PyTorch is needed to import
tessiq. The C++ core builds and tests without it, which is what keeps the compiler testable on a
machine with neither torch nor a GPU:

```bash
cmake -S . -B build -G Ninja -DTESSIQ_BUILD_PYTHON=OFF
```

If you're going to commit:

```bash
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
```

## Layout

| Path | |
|---|---|
| [`csrc/`](csrc/) | C++/CUDA core: runtime, IR, passes, lowering, codegen, kernels |
| [`python/tessiq/`](python/tessiq/) | Python API and the `torch.compile` backend |
| [`ops/`](ops/) | Op definitions in YAML, and the source of truth for codegen (phase 7) |
| [`tools/`](tools/) | `gen_ops.py`, `tessiq-opt`, `tessiq-explain` (phases 3, 7, 8) |
| [`tests/`](tests/) | gtest suites and golden pass tests; Python accuracy and end-to-end tests |
| [`bench/`](bench/) | Benchmarks, and the scripts behind any number published here |
| [`docs/`](docs/) | Architecture, IR spec, pass reference, design notes |

## Standards

`-Werror` with a wide warning set, clang-tidy, ruff, `mypy --strict` and ASan/UBSan all gate merges,
and the pre-commit hooks run the same versions CI does. The rules that come up most: user input
never reaches an assert, every pass states why its transformation is legal, and no performance
claim ships without the script that produced it.

[CONTRIBUTING.md](CONTRIBUTING.md) has the rest, and the commands to run every gate locally.

## Benchmarks

Nothing worth showing before phase 6. Numbers land here when they exist, next to the scripts that
produced them.

## Prior art

Built on publicly documented work: PyTorch's out-of-tree backend mechanism and `torch.compile`
internals, `pytorch/xla`, MLIR's pass infrastructure, IREE, TVM. Non-obvious decisions cite their
precedent where they're made.

## License

Apache-2.0.
