# tessiq

A PyTorch accelerator backend built from scratch: a custom device, a graph compiler, and a kernel
library. It exists to answer, in working code, what a hardware vendor actually has to build to put
a chip behind `model.to(...)` and `torch.compile`.

Named after *tessera*, the Latin word for a mosaic tile, which is what the compiler makes of a
graph: pieces cut to fit, then set into one surface. Said `/ˈtɛs.ɪk/`.

Phases 1 to 5 of 10 are done. Tensors allocate, transfer and compute on two backends, each with
its own caching allocator, and a CPU fallback covers every op that has no kernel yet.
`torch.compile(model, backend="tessiq")` captures a graph, imports it into the IR, partitions it
into the parts the backend claims, lowers that into a recipe, and runs it. Forward and backward
both go through it, and a graph compiled once is not compiled again.

```python
import torch

model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
compiled = torch.compile(model, backend="tessiq")  # the name resolves without importing tessiq
batch = torch.randn(8, 16)

torch.allclose(compiled(batch), model(batch))
```

The device is usable on its own, without the compiler:

```python
import torch, tessiq

reference = torch.randn(1024, device=tessiq.cpu)  # host memory, the correctness oracle
fast = reference.to(tessiq.gpu)  # real GPU memory

torch.allclose(fast.to(tessiq.cpu), reference)
tessiq.memory_stats(tessiq.gpu).allocated_bytes
```

Passes are exercised as text in, text out:

```console
$ tessiq-opt --passes=cse,dce graph.tsr
graph @demo(%a: f32[4,8], %b: f32[4,8]) -> f32[4,8] {
  %s0 = add(%a, %b) : f32[4,8]
  %out = mul(%s0, %s0) : f32[4,8]
  return %out
}
```

**Arithmetic still runs on the host.** Every op reaches PyTorch's CPU kernels through the fallback,
so a GPU tensor pays two transfers per operation and is slower than plain CPU. What exists is the
plumbing underneath: a real device, real device memory, per-backend accounting, two backends that
can be checked against each other, and a compiler that decides what to take and plans where it
goes. Kernels arrive next. [PHASES.md](PHASES.md) has the full breakdown.

The compiler's own numbers, on a 61-step elementwise graph, from
[`bench/compile_overhead.py`](bench/compile_overhead.py):

| | |
|---|---|
| Compiling a graph | 440 us |
| Answering from the recipe cache | 110 us |
| Intermediates, planned into one arena | 8 KB against 240 KB unplanned |

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
