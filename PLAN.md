# tessera — Design & Implementation Plan

> A from-scratch PyTorch accelerator backend: custom device, graph compiler, and kernel library.

**Status:** planning · **Owner:** Logeswaran123 · **Created:** 2026-09-17

---

## 1. What this project is

`tessera` is a complete, working PyTorch hardware-backend integration built from scratch. It registers
a real device with PyTorch, captures model graphs, runs them through a genuine optimizing compiler
pass pipeline, lowers the result to a kernel library, and executes it — with two execution targets:

- a **CPU reference backend** (portable, correctness oracle, runs in CI on any machine), and
- a **CUDA backend** (real kernels + JIT-generated fused kernels, where the optimizations pay off).

The name is Latin for a mosaic tile: the compiler's central job is tiling a model graph into fused
clusters that each become one kernel launch.

### Why this project

Backend integration is the least-documented, highest-leverage part of the ML systems stack. Most
public projects stop at "custom op" or "Triton kernel." Very few show the *whole* path — dispatcher
registration, device memory management, graph capture, an SSA IR, a pass manager, fusion, memory
planning, recipe caching, and kernel dispatch — as one coherent, runnable system. That end-to-end
span is the thing worth showing.

### Success criteria

The project succeeds if a reader can:

1. `pip install -e .` on a CPU-only machine, run the test suite, and see it pass.
2. Run `torch.compile(model, backend="tessera")` on a real model and get correct results.
3. Run `tessera-explain model.py` and watch the graph transform pass by pass, ending in a kernel schedule.
4. See a **measured** speedup from fusion on the CUDA backend, with the benchmark reproducible.
5. Read `docs/architecture.md` and understand how a PyTorch backend actually works.
6. Open any file at random and find production-quality code — error handling, tests, docs — not
   prototype code with the rough edges still showing (§7).
7. See exactly where a Triton emitter or an MLIR target would plug in, because the interfaces are
   already there (§9).

---

## 2. Prior art and design influences

tessera is designed from publicly documented sources. Backend integration is well described in the
open, and every structural decision here traces to something a reader can go and read.

| Source | What it informs |
|---|---|
| PyTorch's out-of-tree backend mechanism and `torch/csrc` | Device registration, dispatch, the allocator interface (§4.4) |
| `pytorch/xla` | Graph capture and lowering for a non-native device |
| MLIR | Pass infrastructure, the textual-IR-plus-golden-test method (§5.1), and the phase 9 export target (§9.3) |
| IREE, TVM | Fusion strategy, scheduling, memory planning |
| `torch.compile` internals (Dynamo, AOTAutograd, FX) | The graph path (§4.1) |
| Triton | Kernel-level codegen, and the phase 8 emitter (§9.2) |

Where a design decision has a public precedent, the code comment cites that precedent at the point
of use, so a reader can follow the reasoning to its source.

### The problems a backend must solve

These are the industry-general problems any accelerator backend faces. Naming them explicitly is
what makes the phase plan coherent rather than a list of features:

| Problem | How tessera addresses it |
|---|---|
| Ops arrive one at a time; per-op dispatch is too slow | Graph capture via `torch.compile`, recipe caching (§5.6) |
| Graphs need normalizing before optimizing | Canonicalization pass set (§5.3) |
| Some ops aren't supported by the backend | Capability-based partitioning + CPU fallback (§5.4) |
| Unfused elementwise chains are memory-bound | Fusion + NVRTC codegen (§5.5, §5.7) |
| Accelerators prefer different tensor layouts | Layout assignment with transpose elimination (§5.3) |
| Compilation is expensive and must be amortized | Recipe cache keyed on graph + shape signature (§5.6) |
| Shapes change between iterations | Symbolic shapes + bucketing (§5.8, Phase 7) |
| Hundreds of ops means hundreds of registrations | YAML op definitions + code generator (§6) |
| Peak memory dominates on-device feasibility | Liveness-based memory planner (§5.3) |

### Design decisions that are tessera's own

Three places where this project makes a choice the prior art above does not, and which are
therefore worth defending on their merits:

1. **Its own SSA IR with a textual format.** Most in-tree backends optimize the framework's
   existing graph structure in place. tessera defines its own IR with a parseable/printable text
   form, which makes every pass testable in isolation via golden files and enables a standalone
   `tessera-opt` tool. This follows MLIR/LLVM practice rather than framework practice.
2. **Runtime kernel generation.** Fused elementwise clusters are compiled to CUDA source at
   runtime via NVRTC and cached on disk, rather than dispatched to a fixed library of pre-built
   fused kernels.
3. **No lazy-tensor mode.** Graph capture is `torch.compile`-only (§4). The lazy path is legacy
   complexity; skipping it is a design decision, documented as such.
## 3. Target environment

Verified on this machine as of 2026-09-17:

| Component | Value | Notes |
|---|---|---|
| GPU | NVIDIA RTX 5080, 16 GB, **sm_120** (Blackwell) | Set `CMAKE_CUDA_ARCHITECTURES=120` |
| CUDA toolkit | 12.8 (`/usr/local/cuda-12.8`) | Supports sm_120; NVRTC ships with it |
| Host compiler | g++ 11.4.0 (Ubuntu 22.04, WSL2) | C++17; check nvcc host-compiler compatibility |
| Python | 3.13.11 (miniconda) | |
| CPU / RAM | 32 cores / 15 GB | RAM is the build constraint — cap `-j` (§12) |
| PyTorch | **not yet installed** | Must be installed before Phase 1 |
| CMake | **not yet installed** | Must be installed before Phase 0 |

### Setup prerequisites (before Phase 0)

```bash
conda create -n tessera python=3.12   # 3.12, not 3.13 — see note
conda activate tessera
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install cmake ninja pybind11 pytest
```

Two notes:

- **Pin the torch version exactly** in `requirements.txt` once chosen, and record it in the README.
  The device-backend registration APIs and the `torch.compile` backend contract both shift between minor
  releases; an unpinned torch will break the build without warning. Check the current stable release
  at setup time rather than assuming.
- **Python 3.12 over 3.13** unless the chosen torch release has confirmed 3.13 wheels with cu128.
  Verify before committing to a version; falling back to 3.12 costs nothing here.
- Confirm the installed torch was built with a CUDA new enough for sm_120
  (`torch.cuda.get_arch_list()` should include `sm_120`). If not, the CUDA backend will compile but
  fail at launch — a confusing failure mode worth ruling out on day one.

---

## 4. Architecture

### 4.1 The two paths

```
                          ┌─────────────────────────────────────┐
   torch.Tensor.to()      │        EAGER PATH (Phase 1)         │
   single aten op    ────>│  aten op → tessera dispatch key      │
                          │  → kernel registry → device runtime  │
                          └─────────────────────────────────────┘
                              correctness floor + fallback target

                          ┌─────────────────────────────────────┐
   torch.compile(         │        GRAPH PATH (Phase 3+)        │
     model,               │                                      │
     backend="tessera")   │  Dynamo  ──> FX GraphModule          │
                     ────>│     │                                │
                          │     v                                │
                          │  AOTAutograd ──> fwd graph + bwd graph
                          │     │            (functionalized,    │
                          │     │             decomposed to ATen)│
                          │     v                                │
                          │  Importer ──> Tessera IR             │
                          │     │                                │
                          │     v                                │
                          │  PassManager                         │
                          │     canonicalize → shape infer →     │
                          │     layout → algebraic → fuse →      │
                          │     partition → schedule → memplan   │
                          │     │                                │
                          │     v                                │
                          │  Lowering ──> Recipe                 │
                          │     (ordered KernelSpecs +           │
                          │      memory plan + IO bindings)      │
                          │     │                                │
                          │     v                                │
                          │  RecipeCache ──(hit)──> skip all above
                          │     │                                │
                          │     v                                │
                          │  Runtime ──> execute on Device       │
                          └─────────────────────────────────────┘
                                          │
                              ┌───────────┴───────────┐
                              v                       v
                        CPU reference            CUDA backend
                        (loops, portable)   (cuBLAS/cuDNN + NVRTC fused)
```

The eager path is not a throwaway. It is the **fallback target** for ops the compiler doesn't
support, and the **correctness oracle** the graph path is tested against.

### 4.2 Repository layout

```
tessera/
├── CMakeLists.txt
├── pyproject.toml                 # scikit-build-core → drives CMake
├── README.md
├── PLAN.md                        # this document
├── LICENSE                        # Apache-2.0 or MIT
│
├── csrc/                          # C++/CUDA core
│   ├── runtime/                   # Phase 1
│   │   ├── device.h/.cpp          # Device interface; CpuDevice, CudaDevice
│   │   ├── allocator.h/.cpp       # caching block allocator, stream-aware
│   │   ├── stream.h/.cpp          # stream + event abstraction
│   │   ├── host_memory.h/.cpp     # pinned staging buffers
│   │   └── register.cpp           # device backend registration, DeviceGuardImpl
│   │
│   ├── ir/                        # Phase 2
│   │   ├── types.h                # TensorType: shape, dtype, layout, strides
│   │   ├── graph.h/.cpp           # Graph, Node, Value; SSA, use-def
│   │   ├── builder.h/.cpp
│   │   ├── verifier.cpp           # structural invariants
│   │   ├── printer.cpp            # IR → text
│   │   └── parser.cpp             # text → IR  (roundtrip-tested)
│   │
│   ├── passes/                    # Phases 2,4,5,6
│   │   ├── pass.h                 # Pass interface, PassManager, registry
│   │   ├── canonicalize/          # dce, cse, const_fold, inplace_to_functional
│   │   ├── shape/                 # shape inference, symbolic dims
│   │   ├── layout/                # layout assignment, transpose elimination
│   │   ├── algebraic/             # simplification, conv+bn folding
│   │   ├── fusion/                # elementwise, reduction, pattern fusion
│   │   ├── partition/             # capability clustering, fallback marking
│   │   ├── schedule/              # liveness-aware topological ordering
│   │   └── memory/                # liveness analysis, buffer reuse, arena
│   │
│   ├── lowering/                  # Phase 4
│   │   ├── kernel_spec.h          # kernel id + params + buffer bindings
│   │   ├── recipe.h/.cpp          # executable plan
│   │   ├── recipe_cache.h/.cpp    # hash → recipe, LRU, on-disk persistence
│   │   └── lower.cpp              # IR → Recipe
│   │
│   ├── codegen/                   # Phase 5
│   │   ├── fused_emitter.cpp      # fused cluster → CUDA source
│   │   ├── nvrtc_compiler.cpp     # NVRTC compile + cubin disk cache
│   │   └── templates/             # kernel source templates
│   │
│   ├── kernels/                   # Phases 1,5,6
│   │   ├── registry.h/.cpp        # kernel_id → impl, per backend/dtype
│   │   ├── cpu/                   # reference implementations
│   │   └── cuda/                  # elementwise, reduce, gemm, conv, norm, softmax
│   │
│   ├── bindings/                  # pybind11 surface
│   └── generated/                 # gitignored; output of tools/gen_ops.py
│
├── ops/                           # Phase 6 — op definition YAML (source of truth)
│   ├── elementwise.yaml
│   ├── reduction.yaml
│   ├── contraction.yaml
│   └── movement.yaml
│
├── python/tessera/
│   ├── __init__.py
│   ├── _C.pyi                     # type stubs for the extension
│   ├── backend.py                 # torch.compile backend registration
│   ├── device.py                  # device API, memory stats, streams
│   ├── debug.py                   # IR dumps, pass timeline, graphviz export
│   ├── profiler.py                # kernel timeline → chrome trace
│   └── testing.py                 # accuracy harness vs CPU eager
│
├── tools/
│   ├── gen_ops.py                 # YAML → C++ registrations, shape fns, docs
│   ├── tessera_opt.cpp            # CLI: read IR, run pass pipeline, print
│   └── tessera_explain.py         # CLI: visualize a model's compilation
│
├── tests/
│   ├── cpp/                       # gtest: IR, passes (golden), memplan, allocator
│   │   └── passes/golden/         # *.tsr input + expected output pairs
│   ├── python/
│   │   ├── test_device.py         # eager path, copies, memory
│   │   ├── test_ops.py            # per-op accuracy vs CPU eager
│   │   ├── test_compile.py        # end-to-end torch.compile
│   │   ├── test_fallback.py       # unsupported ops fall back correctly
│   │   └── test_cache.py          # recipe cache hit/miss behavior
│   └── models/                    # MLP, ResNet block, transformer block
│
├── bench/
│   ├── micro/                     # per-pattern: fused vs unfused
│   └── models/                    # model-level, fusion on/off
│
├── docs/
│   ├── architecture.md            # the main explainer
│   ├── ir.md                      # IR spec + textual format grammar
│   ├── passes.md                  # each pass: what, why, example
│   ├── adding_an_op.md            # contributor guide
│   ├── op_coverage.md             # generated by gen_ops.py
│   └── design_notes/              # ADR-style decision records
│
└── .github/workflows/ci.yml       # CPU-only: build, cpp tests, py tests, lint
```

### 4.3 Core data structures

**`TensorType`** — shape (with symbolic dims), dtype, layout, strides, device. Every `Value` carries one.

**`Graph` / `Node` / `Value`** — SSA. A `Node` has an op name, ordered inputs, ordered outputs, and an
attribute dict. A `Value` has exactly one defining node and many uses. Use-def chains are maintained
incrementally so pattern matching is cheap.

**`Recipe`** — the compiled artifact:
```
Recipe {
  vector<KernelSpec>  kernels;      // ordered launch sequence
  MemoryPlan          plan;         // arena size + per-buffer offsets
  vector<IOBinding>   inputs;       // graph input index → buffer id
  vector<IOBinding>   outputs;
  GraphSignature      signature;    // cache key provenance
}
```

**Cache key** — hash over: graph structure (op sequence + topology), input shapes (or bucket ids for
dynamic shapes), dtypes, layouts, device capability, and compiler flags. Getting this right matters:
too coarse silently returns a wrong recipe, too fine never hits. Unit-test collisions explicitly.

### 4.4 Device backend registration and naming

PyTorch reserves one dispatch key for out-of-tree backends. That key's internal identifier is fixed
in PyTorch's own `DispatchKey` enum and **cannot be renamed by us** — it is their API surface, not a
name this project chose. What *is* ours is every name a user or a reader ever sees:

```python
torch.utils.rename_privateuse1_backend("tessera")  # device name: .to("tessera")
torch.utils.generate_methods_for_privateuse1_backend()  # enables tensor.tessera()
```

After the rename, the user-facing surface is `torch.tessera.*`, `.to("tessera")`, `tensor.tessera()`,
and `device="tessera"`. PyTorch's reserved identifier never appears in user-facing code or docs.

**The rule for this codebase:** the reserved identifier appears **only** inside
`csrc/runtime/register.cpp` and the one Python call above — the exact API boundary where PyTorch
requires it — and nowhere else. Everything else, in code, comments, docs and tests, says *tessera
device backend* or *device backend registration*. A grep for the reserved name should return a
single-digit number of hits, all of them in the registration layer.

This is not cosmetic. Confining a foreign vocabulary to one file is the same discipline as confining
a third-party API behind an adapter: when PyTorch changes how out-of-tree backends register — and it
will (§12) — exactly one file needs to change.

**Verify at Phase 1:** these helper names have moved between torch releases. Confirm them against
the installed, pinned version rather than trusting this document; if they've moved, fix them here
and in `register.cpp` together.

---

## 5. Compiler pass pipeline

The passes are the heart of the project. Each one is independently testable via `tessera-opt` on
textual IR with golden output files.

### 5.1 Pass infrastructure

`Pass` interface with `run(Graph&) -> bool changed`. `PassManager` handles ordering, fixpoint
iteration for canonicalization, per-pass enable/disable flags, before/after IR dumps, timing, and
verification after each pass in debug builds. Passes register themselves by name so `tessera-opt
--passes=dce,cse,fuse-elementwise` works from the CLI.

### 5.2 Import

FX graph → Tessera IR. Runs after AOTAutograd, so the input is already functionalized and decomposed
to core ATen — a much smaller and better-behaved op set than raw `aten`. Unknown ops become opaque
`fallback` nodes rather than errors.

### 5.3 Canonicalization and analysis

| Pass | What it does |
|---|---|
| `dce` | Remove nodes whose outputs are unused |
| `cse` | Deduplicate identical pure subexpressions |
| `const-fold` | Evaluate ops with all-constant inputs at compile time |
| `inplace-to-functional` | Normalize remaining in-place ops (mostly handled upstream by functionalization) |
| `shape-infer` | Propagate shapes/dtypes; symbolic dims where unknown; fails loudly on inconsistency |
| `layout-assign` | Choose per-op preferred layout (e.g. NHWC for conv on CUDA), insert conversions |
| `transpose-elim` | Cancel adjacent inverse layout conversions inserted above |
| `algebraic-simplify` | `t(t(x))→x`, `x+0→x`, `x*1→x`, `reshape(reshape)→reshape` |
| `fold-conv-bn` | Fold BatchNorm into preceding Conv weights (inference only — guard on training flag) |

`layout-assign` + `transpose-elim` as a pair is the interesting one: assignment inserts conversions
greedily and locally, elimination cleans up globally. That separation is why it works.

### 5.4 Partitioning

Capability-based clustering: walk the graph, mark each node supported/unsupported by the active
backend, then grow maximal connected clusters of supported nodes. Unsupported nodes stay on CPU;
cluster boundaries become explicit device↔host transfers. Clusters that are too small to amortize
launch overhead are rejected back to fallback — the threshold is a tunable, and measuring it is a
legitimate small experiment worth writing up.

### 5.5 Fusion

- **Elementwise fusion** — the primary win. Producer-consumer chains of elementwise ops become one
  kernel, turning N memory round-trips into 1. Fusion is legal when the producer has a single
  consumer, or when recomputation is cheaper than materializing (a cost-model decision).
- **Reduction fusion** — fuse elementwise producers into a reduction's read loop.
- **Pattern fusion** — specific high-value patterns matched structurally: `bias+activation` after
  GEMM, `BN+ReLU+residual add`, softmax, layernorm.

Each fusion pass must state its legality condition in a comment and have a golden test showing both
a case it fires on and a case it correctly refuses.

### 5.6 Scheduling, memory planning, lowering

- **Schedule** — topological order chosen to minimize peak liveness, not just any valid order.
- **Memory plan** — liveness intervals per buffer, then greedy interval-graph coloring into a single
  arena. Report peak memory vs naive allocation; that ratio is a headline number for the README.
- **Lower** — each node/cluster → `KernelSpec` (kernel id, params blob, buffer bindings, launch
  config). Result is a `Recipe`, cached by signature.

### 5.7 CUDA codegen (the differentiator)

Fused elementwise clusters are emitted as CUDA C++ source at compile time, compiled with NVRTC, and
cached as cubins on disk keyed by source hash. This is what makes fusion *actually* pay — no
combinatorial library of pre-built fused kernels needed.

Start narrow: same-shape, same-dtype, unary/binary elementwise chains only. Expand to broadcasting,
then mixed dtypes, then reduction epilogues. Every expansion needs a correctness test against the
CPU reference before it lands.

### 5.8 Dynamic shapes (Phase 7)

Symbolic dims through shape inference; recipes keyed on shape *buckets* rather than exact shapes, so
a varying sequence length doesn't recompile every step. Guard conditions recorded per recipe and
checked on lookup. This is genuinely hard — it is deliberately last.

---

## 6. Op definitions and code generation

Hundreds of ops × (registration + shape function + lowering + docs) is untenable by hand. `ops/*.yaml`
is the single source of truth:

```yaml
# ops/elementwise.yaml
add.Tensor:
  category: elementwise
  kernel: add
  dtypes: [f32, f16, bf16, i32, i64]
  backends: [cpu, cuda]
  shape: broadcast          # named shape rule
  fusable: true

sum.dim_IntList:
  category: reduction
  kernel: reduce_sum
  dtypes: [f32, f16, bf16]
  backends: [cpu, cuda]
  shape: reduce_dims
  fusable: producer_only
```

`tools/gen_ops.py` generates: dispatcher registrations for the eager path, importer entries mapping
ATen → IR ops, shape functions from named rules, lowering boilerplate, the kernel registry table, and
`docs/op_coverage.md`. Generated files land in `csrc/generated/` (gitignored, built by CMake) so
generated code is never reviewed as if it were hand-written.

Hand-written escapes matter: any op can override any generated piece by supplying a custom function
name in YAML. Without that escape hatch the generator becomes a straitjacket at op ~40.

---

## 7. Engineering standards

**Every line written for this project is production code.** Not prototype code that will be cleaned
up later — there is no later. A repo whose purpose is to demonstrate engineering judgment is judged
on exactly that, and a reviewer who opens a random file and finds a commented-out block, a `TODO:
fix`, or a function with no error handling has already formed their opinion. These are the rules,
and they are not negotiable per-file.

### 7.1 C++ standards

- **C++17**, no compiler extensions. `-Wall -Wextra -Wpedantic -Werror` in CI; warnings are build
  failures, not noise to scroll past.
- **RAII everywhere.** No raw `new`/`delete` outside a resource wrapper's implementation. Every
  device allocation, stream, event, and NVRTC program is owned by a type whose destructor releases it.
- **Ownership is explicit in the type.** `unique_ptr` for sole ownership, raw pointer/reference for
  non-owning views, `shared_ptr` only where ownership genuinely is shared (it rarely is — IR nodes
  are arena-owned by their `Graph`). A `shared_ptr` in a signature must be justified in review.
- **`const`-correct by default.** Analysis passes take `const Graph&`. If a pass needs mutation it
  says so in its signature — this is how a reader knows what a pass does before reading it.
- **Rule of zero.** Write no copy/move/destructor unless the type manages a resource; then write all
  of them and delete what doesn't apply.
- **No global mutable state.** Registries are function-local statics with defined initialization
  order; anything else is passed explicitly. Singletons make the compiler untestable in parallel.
- **Headers are minimal.** Public headers expose the smallest possible interface; implementation
  details use pimpl or live in `detail::` namespaces in `.cpp` files. Include what you use.

### 7.2 Error handling

Three distinct categories, each with one mechanism — mixing them is what makes error handling rot:

| Category | Mechanism | Example |
|---|---|---|
| **Programmer error** (broken invariant) | `TESSERA_DCHECK`, debug-only, crashes loudly | IR node with a null operand after a pass |
| **Recoverable failure** (expected, must be handled) | `Result<T>` return type, `[[nodiscard]]` | Kernel not available for this dtype → fall back |
| **User error** (bad input at the API boundary) | Exception with an actionable message | Unsupported dtype passed from Python |

Rules that follow from this:
- **Never `assert` on user input.** User input is validated at the boundary and rejected with a
  message naming the offending value and what was expected.
- **No silent failures. Ever.** A pass that can't handle a node either transforms it or leaves it
  untouched — never drops it, never half-transforms it.
- **Error messages name the fix, not just the problem.** `"unsupported dtype f64 for kernel 'add' on
  cuda backend; supported: f32, f16, bf16"` — not `"unsupported dtype"`.
- **Exceptions never cross the C ABI boundary.** Caught and translated at the pybind11 layer.

### 7.3 Logging, diagnostics, and observability

- Leveled logging (`ERROR`/`WARN`/`INFO`/`DEBUG`/`TRACE`), **zero cost when disabled** — the argument
  expressions must not evaluate. Controlled by `TESSERA_LOG_LEVEL` and per-module `TESSERA_LOG_MODULES`.
- Every pass logs, at `DEBUG`, what it changed and how many nodes it touched. A compiler you can't
  see inside is a compiler you can't debug.
- `TESSERA_DUMP_IR=before,after,all` writes IR snapshots per pass to a directory. This is the single
  most valuable debugging feature in the project — build it in Phase 2, not when you first need it.
- Compilation statistics (pass timings, node counts, cache hit/miss, kernels generated) are queryable
  from Python, not just printed.

### 7.4 Concurrency

- **Thread-safety is documented per class**, in the header, as one of: not thread-safe / thread-safe /
  thread-compatible. An undocumented class is assumed not thread-safe.
- The recipe cache and kernel registry are the shared-state hot spots — they get explicit locking and
  dedicated concurrent tests.
- **Stream-ordered semantics** for all device memory: a buffer freed on a stream is not reusable until
  prior work on that stream completes. This is the classic source of intermittent, unreproducible
  corruption; the allocator handles it and tests assert it.

### 7.5 Style, naming, and layout

- `clang-format` (LLVM-derived) and `clang-tidy` enforced in CI, never by hand. `ruff` + `mypy` on the
  Python side, `ruff format` for formatting.
- Naming: `PascalCase` types, `snake_case` functions/variables, `member_` trailing underscore,
  `kConstant`, `TESSERA_` macro prefix. One convention, applied everywhere, no exceptions.
- **File layout mirrors the architecture** (§4.2). A reader who understands the architecture can
  find any file without searching.
- **No dead code, no commented-out code.** Git remembers. Deleting is free.
- **No bare `TODO`.** Either fix it now or file an issue and write `TODO(#123): <what and why>`.

### 7.6 Documentation, and comment restraint

**Comment sparingly. Most code should carry no comment at all.**

Over-commenting is a real defect, not a harmless excess. Comments that restate the code go stale
the moment it changes, and a file where every line is annotated trains the reader to skip all of
it — including the three comments that mattered. Dense commentary also *reads* as uncertainty: it
suggests the author did not trust the code to speak for itself.

The bar: **write a comment only when a competent reader of this code would otherwise be misled or
stuck.** If the answer is on the line below, delete the comment. A clearer name, a smaller
function, or a better type beats a comment every time — reach for those first.

What earns a comment:

- **Why, when the why is not visible.** A non-obvious ordering constraint, a workaround with a
  ticket link, a deliberate deviation from the obvious approach.
- **Contracts a signature cannot express**: ownership and lifetime, thread-safety, failure modes,
  preconditions. One line each, on public headers only.
- **Pass legality arguments.** Every pass states its precondition, postcondition, and why its
  transformation is sound. A fusion pass without a written legality condition is a latent
  wrong-results bug, and this is the one place where more prose is warranted.
- **Traps.** Code that looks wrong but is correct, where the next reader would "fix" it and break
  something.

What does not:

- Restating the signature, the type, or the line below it.
- Section-divider banners in a file short enough to read in one screen.
- Rationale that belongs in an ADR or in this document. Link to it instead of repeating it — one
  reference, not a paragraph in every file that touches the decision.
- Narrating a change ("now uses X instead of Y"). That is what the commit message is for.
- Restating a `NOLINT` suppression's own explanation more than once.

**Rough calibration:** for implementation files, comment lines in the low single-digit percentages;
for public headers, where contracts genuinely need stating, up to roughly 15–20%. A file that is a
third comments is over-documented and should be cut, not defended. These are smells to check
against, not thresholds to enforce in CI — a legality argument or a hard-won trap is worth its
lines wherever it lands.

**ADRs** (`docs/design_notes/`) carry the reasoning instead: context, options considered, decision,
consequences. That is where a design argument belongs, written once and linked to, rather than
scattered across the files it touches. They are also the best portfolio artifact in the repo.

### 7.7 Commits, review, and CI gates

- **Conventional commits** (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`), imperative mood,
  body explaining *why*. The git log is part of what a reviewer reads.
- **Small, atomic, self-contained commits.** One logical change. Every commit builds and passes tests —
  `git bisect` must stay usable.
- **Feature branches + PRs even when solo.** The PR description is where the reasoning lives, and
  reviewing your own PR a day later catches real bugs.
- **Merge gates, all mandatory:** build clean with `-Werror`; all tests pass; new code has tests;
  clang-format/clang-tidy/ruff/mypy clean; no coverage regression; benchmarks show no unexplained
  perf regression.
- **Sanitizers in CI**: ASan + UBSan on the CPU backend build. `compute-sanitizer` on CUDA tests
  locally before any release tag.

### 7.8 Definition of done

A change is done when **all** of these are true — not when it works on your machine:

- [ ] Implemented with error handling for every failure path (§7.2)
- [ ] Unit tested, including the failure paths and at least one negative case
- [ ] Public API contracts documented, and no comment that restates the code (§7.6)
- [ ] No new warnings, lint errors, or type errors
- [ ] Logging added at the right level; IR dumps work if it's a pass
- [ ] Benchmarked if it touches a hot path; the number is in the PR description
- [ ] Correctness verified against the CPU reference if it touches the CUDA path
- [ ] CHANGELOG updated if user-visible

### 7.9 Performance discipline

- **No performance claim without a reproducible measurement.** Every number in the README maps to a
  script in `bench/` that anyone can run.
- Benchmarks report **distribution, not a single number** — median plus spread, with warmup excluded
  and variance reported. A single timing on a laptop is not a measurement.
- **Correctness gates performance, always.** A faster kernel that fails the accuracy test against the
  CPU reference is not an optimization; it's a bug. No exceptions, no "close enough" tolerances
  chosen after seeing the failure.

---

## 8. Testing strategy

| Layer | Tool | What it covers |
|---|---|---|
| IR | gtest | Build/verify/print/parse roundtrip; use-def integrity under mutation |
| Passes | gtest + golden `.tsr` files | Each pass: fires correctly, *refuses* correctly, is idempotent |
| Memory planner | gtest | Peak memory vs known-optimal on small graphs; no overlapping live buffers |
| Allocator | gtest | Fragmentation, reuse, stream-ordered free correctness |
| Ops | pytest | Every op vs CPU eager, all dtypes, edge shapes (empty, 0-d, broadcast) |
| End-to-end | pytest | MLP / ResNet block / transformer block: forward, then forward+backward |
| Fallback | pytest | Graphs with unsupported ops produce correct results |
| Cache | pytest | Hit on repeat, miss on shape/dtype change, no false hits |
| Perf | pytest-benchmark | Regression guard on fused-vs-unfused microbenchmarks |

**The golden-file approach for passes is the highest-value testing decision here.** A pass test is a
text file in, a text file out — readable in a diff, trivial to add, and it documents the pass's
behavior better than prose. This is why the IR needs a textual format (§4.2).

**CI** (`ubuntu-latest`, no GPU): build CPU-only, run all C++ tests, run all Python tests except those
marked `@pytest.mark.cuda`, run clang-format/clang-tidy/ruff. CUDA tests run locally via
`pytest -m cuda`. A green badge on a GPU-less runner is only honest if the CPU backend genuinely
covers the compiler — which it does, since passes are hardware-independent.

---

## 9. Extensibility: Triton and MLIR/LLVM

**Short answer: yes, the plan scales to both — but only if three seams are designed in from the
start.** Retrofitting them later means a rewrite. They cost almost nothing to build now and they are
the difference between a project that grows and one that calcifies.

The two technologies enter at *different levels* and are complementary, not alternatives: Triton
replaces how a single fused kernel is produced; MLIR replaces how the whole graph is represented and
lowered.

```
  Tessera IR ──> passes ──> fused clusters ──┬──> NVRTC emitter    ──> cubin   (Phase 5)
       │                                     ├──> Triton emitter   ──> cubin   (Phase 8)
       │                                     └──> CPU emitter      ──> loops
       │
       └──> MLIR export ──> linalg/tensor ──> MLIR pipeline ──> LLVM ──> CPU/GPU  (Phase 9)
              ^
              └── seam 2                      ^
                                              └── seam 1 (KernelEmitter)
```

### 9.1 The three seams to build now

**Seam 1 — `KernelEmitter` interface (build in Phase 5).** Fusion produces a *cluster description*;
something else turns that into a runnable kernel. Define the interface even while NVRTC is the only
implementation:

```cpp
class KernelEmitter {
 public:
  virtual ~KernelEmitter() = default;
  virtual bool  supports(const FusedCluster&) const = 0;
  virtual Result<CompiledKernel> emit(const FusedCluster&, const TargetInfo&) = 0;
  virtual std::string_view name() const = 0;
};
```

Registered by name, selected by config (`TESSERA_EMITTER=nvrtc|triton|cpu`). Adding Triton then means
adding one class and zero changes anywhere else — and, critically, it means NVRTC vs Triton can be
**benchmarked head-to-head on identical clusters**, which is a far more interesting result than either
one alone.

**Seam 2 — `LoweringTarget` abstraction (build in Phase 4).** `Recipe` must not be hardcoded as *the*
output of lowering. Lowering targets a `LoweringTarget` interface; `RecipeTarget` is the first
implementation, `MlirTarget` a later one. The trap to avoid: letting `lower.cpp` reach into
`Recipe`-specific structures. If lowering only ever speaks the target interface, MLIR export is an
additive change.

**Seam 3 — keep the IR op set small, orthogonal, and target-neutral (ongoing, from Phase 2).** This
is the one that's easy to violate by accident and expensive to repair. Rules:
- IR ops carry **no CUDA-specific assumptions** — no block sizes, no warp semantics, no launch config.
  Those belong in the emitter, below the IR.
- Prefer a **small orthogonal op set** (core-ATen-like) over a large ad-hoc one. Every ad-hoc fused op
  invented for convenience is an op with no `linalg` equivalent later.
- **Shape, dtype, and layout live on the type** (§4.3), not in side tables — this maps directly onto
  MLIR's ranked tensor types. Side-table metadata does not survive the conversion.

### 9.2 Adding Triton (Phase 8)

Triton slots in at seam 1 as an alternative emitter. A fused cluster becomes generated Triton kernel
source instead of CUDA C source; `triton.compile` replaces NVRTC; the cubin cache is unchanged.

**What it buys:** autotuning over block sizes and num_warps for free; far easier expression of
reductions, softmax, and eventually matmul than hand-written CUDA; block-pointer semantics that make
tiling explicit rather than index arithmetic.

**What it costs:** Python in the compile path — either in-process (simpler, but couples the C++
compiler to the interpreter) or via a subprocess with a cached artifact (cleaner separation, more
moving parts). Decide this when you get there; the emitter interface makes it a contained choice.
It also adds a heavyweight dependency, which is exactly why the NVRTC emitter stays the default and
Triton is opt-in.

**Why this order matters:** writing the NVRTC emitter *first* means you understand what the generated
code has to do before adopting a tool that does it for you. Reaching for Triton first would make the
project a Triton wrapper — which is the common, less interesting version of this project.

### 9.3 Adding MLIR/LLVM (Phase 9)

Three possible integration modes; the plan recommends the first:

| Mode | What it means | Verdict |
|---|---|---|
| **Export target** | Tessera IR → MLIR (`tensor`/`linalg`) → MLIR's pipeline → LLVM → CPU/GPU | **Recommended.** Additive, keeps the hand-written compiler as the entry point, and demonstrates interop skill on top of compiler skill |
| **Import source** | `tessera-opt` consumes MLIR and runs tessera passes on it | Useful later for interop demos; cheap once the converter exists in both directions |
| **Full replacement** | Define a tessera dialect and delete the custom IR | **Rejected.** It's a rewrite, and it deletes the part of the project that demonstrates the most — building the IR and pass infrastructure from scratch |

**What already fits, by design.** The current plan is unusually well-positioned here, and not by
accident — several choices were made with this in mind:

| Plan decision | Why it makes MLIR integration tractable |
|---|---|
| SSA IR, explicit types (§4.3) | Direct structural correspondence to MLIR's SSA + type system |
| Textual IR format (§4.2) | Conversion is testable as text-to-text golden tests, same harness as passes |
| `PassManager` with a named-pipeline CLI (§5.1) | Deliberately modeled on MLIR's; passes port with mostly-mechanical change |
| Ops defined as data in YAML (§6) | **`gen_ops.py` can emit MLIR ODS/TableGen from the same source of truth** — the dialect definition comes nearly free |
| Shape/layout on the type (§4.3) | Maps onto ranked tensor types without a side channel |

That fourth row is the strongest argument that this plan scales: the op definitions are already data,
so a second generator backend targeting TableGen is a new emitter template, not a redesign.

**What it costs, honestly.** MLIR is a large build dependency with its own version churn, and it
makes the project harder to build for a casual visitor. Mitigation: MLIR support is a **build-time
opt-in** (`-DTESSERA_ENABLE_MLIR=ON`, default `OFF`), never required for the core project or CI.

**Realistic scope.** Phase 9 as written means: export the IR to `linalg` on tensors, run a standard
MLIR bufferization + lowering pipeline, and execute the result — for a *subset* of ops, demonstrated
end-to-end on one model. That is an honest, achievable deliverable and a strong one. "Reimplement the
whole backend in MLIR" is not.

### 9.4 Other extension points the seams enable

The same three seams open doors beyond Triton and MLIR, at little extra cost:

- **A new hardware backend** (ROCm, Metal, a simulator): implement `Device` + a `KernelEmitter`; the
  entire compiler above is unchanged.
- **Quantization**: a dtype and a pass set — the type system already carries dtype on every value.
- **Distributed/collectives**: new IR ops + a partitioning pass; no runtime changes below.
- **Inference-only graph export**: a `LoweringTarget` that serializes a recipe instead of executing it.

**The discipline that keeps all of this true:** whenever a choice arises between a general seam and a
convenient shortcut, and the seam costs less than about a day — take the seam. Past that, write the
shortcut and file an issue naming the seam it displaced. That rule is what keeps a project extensible
without paying for abstraction that never gets used.

---

## 10. Phased delivery

Each phase ends in a **demonstrable milestone**. The repo should be presentable at every phase
boundary, not just at the end.

### Phase 0 — Scaffolding and engineering standards
Repo init, CMake + scikit-build-core skeleton, pybind11 hello-world extension, pytest + gtest wired
up, LICENSE, README skeleton. **All of §7's automation lands here, before the first real line of
code:** `-Werror`, clang-format + clang-tidy, ruff + mypy, pre-commit hooks, ASan/UBSan build config,
conventional-commit linting, and CI enforcing every one of them.
**Milestone:** `pip install -e .` works; `import tessera` succeeds; CI is green; a deliberately
badly-formatted, warning-producing commit is *rejected* by CI.
**Why here:** standards adopted after the code exists are standards applied to a backlog. Adopted
first, they cost nothing and are never negotiated again.

### Phase 1 — Device runtime and eager path
`Device` abstraction with CPU and CUDA impls; caching allocator; streams/events; device backend
registration with `DeviceGuardImplInterface` and the backend rename (§4.4); H2D/D2H copies; a
handful of hand-registered eager ops.
**Milestone:** `x = torch.randn(1024, device="tessera"); y = (x + x).cpu()` gives correct results on
both backends. Memory stats API reports allocations.
**Risk:** Device backend registration is the most version-fragile part of the project. Do it first, on a
pinned torch, so the fragility is discovered on day one rather than in month three.

### Phase 2 — IR and pass infrastructure
Graph/Node/Value, TensorType, builder, verifier, printer, parser. PassManager + registry. `dce`,
`cse`, `const-fold`. `tessera-opt` CLI. Golden test harness.
**Milestone:** `tessera-opt --passes=dce,cse in.tsr` produces expected output; parse/print roundtrip
is property-tested; no torch dependency in any of it.

### Phase 3 — Graph capture
FX importer, `torch.compile` backend registration, capability partitioner, CPU fallback for
unsupported ops, naive lowering (one kernel per node, no fusion yet).
**Milestone:** `torch.compile(mlp, backend="tessera")` runs an MLP end-to-end with correct results,
and a graph containing a deliberately unsupported op still produces correct results via fallback.

### Phase 4 — Shape inference, memory planning, recipe cache
Shape inference pass, canonicalization set, liveness analysis, memory planner, `Recipe`, recipe cache
with signature hashing.
**Milestone:** second call with same shapes hits the cache (assert via counter); measured peak memory
below naive allocation; measured reduction in per-call overhead.

### Phase 5 — Fusion and CUDA codegen
Elementwise fusion pass; fused kernel emitter; NVRTC compilation with on-disk cubin cache; benchmarks.
**Milestone:** a chain like `(a*b + c).relu().sigmoid()` runs as **one** kernel launch, with a
measured, reproducible speedup over the unfused path on the 5080, and bit-comparable results against
the CPU reference.

### Phase 6 — Op coverage and real models
YAML op definitions + `gen_ops.py`; broad elementwise/reduction/movement coverage; GEMM via cuBLAS;
conv via cuDNN; layernorm/softmax kernels; layout assignment.
**Milestone:** a ResNet-18 block and a transformer encoder block compile and run correctly on both
backends. `docs/op_coverage.md` is generated and non-trivial.

### Phase 7 — Training, dynamic shapes, tooling
AOTAutograd backward graphs; symbolic shapes + bucketing; profiler with chrome-trace export;
`tessera-explain`; `docs/architecture.md`; a write-up.
**Milestone:** a small model **trains** to convergence on the tessera backend; sequence-length
variation doesn't trigger recompilation every step.

### Phase 8 — Triton emitter (optional, §9.2)
Second `KernelEmitter` implementation generating Triton kernels for fused clusters; autotuning;
head-to-head benchmark against the NVRTC emitter on identical clusters.
**Milestone:** `TESSERA_EMITTER=triton` produces correct results on the full op test suite, and
`bench/emitters/` publishes a per-pattern comparison table of the two emitters.

### Phase 9 — MLIR export target (optional, §9.3)
`MlirTarget` implementing `LoweringTarget`; Tessera IR → `linalg`-on-tensors converter; `gen_ops.py`
extended to emit ODS from the same YAML; standard MLIR bufferization + lowering pipeline to LLVM.
Gated behind `-DTESSERA_ENABLE_MLIR=ON`, off by default.
**Milestone:** a subset of ops exports to valid MLIR, round-trips through `mlir-opt`, and executes
end-to-end on one model with results matching the CPU reference.

### Sequencing note

Phases 0–3 are the risky ones: they're where an unfamiliar API can cost a week. Phases 4–7 are
additive and mostly under your own control. If time gets tight, a repo that stops cleanly after
Phase 5 — with fusion working and measured — is a strong artifact. A repo that sprawls into Phase 7
with a broken Phase 3 is not.

**Phases 8 and 9 are explicitly optional**, and framing them that way is deliberate. Their value to
the project is mostly realized in Phases 4 and 5, when the seams get built (§9.1) — a plan that
*demonstrably could* take a Triton emitter or an MLIR target, with the interfaces in place to prove
it, already shows the architectural judgment. Building them is a bonus; designing for them is the
skill. Do not let either become the reason Phase 6 never ships.

---

## 11. Showcase surface

What a visitor sees in the first 60 seconds determines whether they read further.

**README must have, in this order:** one-sentence description; an architecture diagram; a
before/after IR snippet showing a fusion actually happening; a benchmark table with real numbers from
the 5080; a 5-line quickstart; then links to deeper docs.

**`tessera-explain`** is the demo that carries the project — point it at a model and it renders the
captured graph, each pass's diff, the final kernel schedule, and the measured timeline. This is the
same instinct as your `cuda-breakdown` tool, applied one level up the stack.

**Numbers to publish** (each reproducible via a script in `bench/`):
- kernel launches before vs after fusion
- wall-clock speedup per fused pattern
- peak memory: planned vs naive
- compile time, and cache-hit time on the second call

**Write-ups** (each one is also a portfolio piece):
`docs/design_notes/` covering device backend registration, why an SSA IR with a text format, the fusion
cost model, and the recipe cache key design.

---

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Device backend registration API churn between torch versions | Build breaks on upgrade | Pin torch exactly; isolate all registration in `csrc/runtime/register.cpp`; document the version contract |
| sm_120 support gaps in torch/cuDNN builds | CUDA path fails at runtime | Verify `torch.cuda.get_arch_list()` on day one (§3); CPU backend keeps the project viable regardless |
| 15 GB RAM with 32 cores | OOM during parallel C++ builds | Cap at `-j8`; use ninja; enable ccache; keep translation units small |
| Scope creep — "one more op" | Never reaching a demo | Phase milestones are the gate; op *breadth* is explicitly Phase 6, not earlier |
| Fusion correctness bugs | Silent wrong results — the worst failure mode | Every fused path tested against CPU reference; CPU reference is never optimized, only kept obviously correct |
| Cache key too coarse | Silent wrong results | Dedicated collision tests; in debug builds, verify cache hits by re-running the compile and comparing |
| Generated code becomes unreviewable | Maintenance wall | Generated files gitignored and clearly marked; hand-written escape hatches per op (§6) |
| Design drifts from its cited public sources | Decisions become unjustifiable in review | Cite the public precedent in the comment at the point of use (§2); an uncited non-obvious choice is a review finding |
| Standards slip under time pressure (§7) | The repo stops demonstrating what it exists to demonstrate | Merge gates are automated, not remembered; CI enforces what discipline shouldn't have to |
| Extension seams (§9.1) never get used | Abstraction cost with no payoff | Seams are cheap by construction; the day-cost rule in §9.4 caps the investment |

---

## 13. Immediate next steps

1. Create the conda env and install pinned torch; verify `sm_120` is in `get_arch_list()` (§3).
2. Install cmake + ninja; confirm `nvcc` works with g++ 11.4 on a trivial `.cu` file.
3. `git init`, push an empty `Logeswaran123/tessera` repo with README + LICENSE + this plan.
4. Begin Phase 0 — and note that Phase 0 *is* the engineering-standards phase (§7). The tooling that
   enforces quality goes in before the code it will judge.

Decide before Phase 0: license (Apache-2.0 recommended — the patent grant matters for a compiler
project, and it matches TVM, IREE and MLIR/LLVM, which keeps the phase 9 export target clean).
