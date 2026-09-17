# tessera

A from-scratch PyTorch accelerator backend: a custom device, an optimizing graph compiler, and a
kernel library — built end to end to show how a real hardware backend plugs into PyTorch.

> **Status: Phase 0 of 9 — scaffolding and engineering standards.**
> The build, test, lint and CI infrastructure are in place, along with the support layer everything
> else is built on. The device, the IR, the compiler passes and the kernels arrive in later phases.
> [`PLAN.md`](PLAN.md) is the full design document and phase plan.
>
> Nothing in this README claims a capability that is not yet implemented. Sections describing
> later phases are marked as such.

## What it does

Two execution paths, sharing one kernel layer:

```
  torch.compile(model, backend="tessera")        model.to("tessera")
              │                                          │
              v                                          v
   Dynamo → FX → AOTAutograd                    aten op → dispatch
              │                                          │
              v                                          │
       Tessera IR  ──> passes ──> lowering ──> recipe    │
              │         canonicalize                     │
              │         shape infer                      │
              │         layout                           │
              │         fuse                             │
              │         schedule                         │
              │         memory plan                      │
              v                                          v
        ┌──────────────── kernel registry ────────────────┐
        │  CPU reference (portable)   CUDA (cuBLAS/NVRTC) │
        └──────────────────────────────────────────────────┘
```

The CPU backend is the correctness oracle and runs anywhere; the CUDA backend is where fusion is
measured. Compiler passes are hardware-independent, so the full pass suite runs in CI without a GPU.

## Quickstart

```bash
git clone https://github.com/Logeswaran123/tessera.git && cd tessera
pip install -r requirements-dev.txt   # pinned tooling, matching CI exactly
pip install -e .                      # builds the native module
pre-commit install --install-hooks && pre-commit install --hook-type commit-msg
pytest                                # Python tests
cmake -S . -B build -G Ninja && cmake --build build -j8 && ctest --test-dir build
```

Requires Python ≥ 3.10, CMake ≥ 3.26, and a C++17 compiler. PyTorch is **not** required to build or
test the compiler core — only the frontend needs it.

## Project layout

| Path | Contents |
|---|---|
| [`csrc/`](csrc/) | C++/CUDA core: runtime, IR, passes, lowering, codegen, kernels |
| [`python/tessera/`](python/tessera/) | Python API and the `torch.compile` backend |
| [`ops/`](ops/) | Op definitions in YAML — the single source of truth for code generation *(phase 6)* |
| [`tools/`](tools/) | `gen_ops.py`, `tessera-opt`, `tessera-explain` *(phases 2, 6, 7)* |
| [`tests/`](tests/) | `cpp/` gtest suites and golden pass tests; `python/` accuracy and end-to-end tests |
| [`bench/`](bench/) | Reproducible benchmarks — every published number maps to a script here |
| [`docs/`](docs/) | Architecture, IR spec, pass reference, and design notes |

## Engineering standards

This repository holds itself to production standards, described in full in
[`PLAN.md` §7](PLAN.md). In short: `-Werror` with an extensive warning set; `clang-tidy`, `ruff` and
`mypy --strict` as merge gates; ASan/UBSan in CI; a three-category error model where user input is
never asserted on; documented thread-safety; and no performance claim without a reproducible
measurement.

## Benchmarks

*Phase 5 onwards.* This section will hold measured fusion speedups, kernel-launch counts, and peak
memory against naive allocation — each reproducible via a script in [`bench/`](bench/). It is
deliberately empty until there are real numbers to put in it.

## Documentation

- [`PLAN.md`](PLAN.md) — design document, architecture, and the nine-phase plan
- [`docs/design_notes/`](docs/design_notes/) — decision records for choices with a real alternative
- `docs/architecture.md`, `docs/ir.md`, `docs/passes.md` — *phases 2–7*

## Acknowledgements and prior art

The design draws on publicly documented work: PyTorch's out-of-tree backend mechanism and
`torch.compile` internals, `pytorch/xla`, MLIR's pass infrastructure, IREE, and TVM. Design
decisions with a public precedent cite it at the point of use.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
