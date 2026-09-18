# Phases

The delivery plan for tessiq, and where each piece of the system arrives. Each phase ends in
something demonstrable; the repository is meant to be presentable at every boundary, not only at
the end.

Status is tracked here rather than in a project board, so that it moves in the same commit as the
work it describes.

| | Phase | State |
|---|---|---|
| 0 | Scaffolding and engineering standards | done |
| 1 | Device runtime and eager path | done |
| 2 | The CUDA device, and choosing a backend per tensor | done |
| 3 | IR and pass infrastructure | done |
| 4 | Graph capture | next |
| 5 | Shape inference, memory planning, recipe cache | |
| 6 | Fusion and CUDA codegen | |
| 7 | Op coverage and real models | |
| 8 | Training, dynamic shapes, tooling | |
| 9 | Triton emitter | optional |
| 10 | MLIR export target | optional |

---

### Phase 0: Scaffolding and engineering standards
Repo init, CMake + scikit-build-core skeleton, pybind11 hello-world extension, pytest + gtest wired
up, LICENSE, README skeleton. **All of the quality automation lands here, before the first real line of
code:** `-Werror`, clang-format + clang-tidy, ruff + mypy, pre-commit hooks, ASan/UBSan build config,
conventional-commit linting, and CI enforcing every one of them.
**Milestone:** `pip install -e .` works; `import tessiq` succeeds; CI is green; a deliberately
badly-formatted, warning-producing commit is *rejected* by CI.
**Why here:** standards adopted after the code exists are standards applied to a backlog. Adopted
first, they cost nothing and are never negotiated again.

### Phase 1: Device runtime and eager path (done)
`Device` abstraction with the CPU reference backend; caching allocator; device backend registration
with `DeviceGuardImplInterface` and the backend rename; transfers in both directions; enough
native ops to make tensors usable, with a CPU fallback covering the rest of ATen.
**Milestone:** `x = torch.randn(1024, device="tessiq"); y = (x + x).cpu()` gives correct results,
and the memory API reports allocations, reuse and cache hits.
**Risk:** Device backend registration is the most version-fragile part of the project. Do it first, on a
pinned torch, so the fragility is discovered on day one rather than in month three.

### Phase 2: The CUDA device, and choosing a backend per tensor (done)

A real `CudaDevice` alongside the CPU reference one, and an explicit choice between them at the
point a tensor is created rather than a process-wide switch.

- `CudaDevice` implementing `Device`: `cudaMalloc`/`cudaFree`, async `cudaMemcpy` in all three
  directions, streams and events, and `device_count()` from the driver.
- One `CachingAllocator` per device, so memory statistics are reported per backend rather than
  pooled into a number that means nothing.
- Device indices select the backend: index 0 is the CPU reference, indices 1..N are the
  CUDA devices. `tessiq.cpu` and `tessiq.gpu` name them in Python.
- Cross-backend copies (`tessiq:0` ↔ `tessiq:1`) stage through the host.
- The guard implementation gains real device switching, and `deviceCount()` covers both.
- Every op-level test runs against both backends and compares results.

**Milestone:** `torch.randn(1024, device=tessiq.gpu)` allocates on the GPU and shows up in
`nvidia-smi`; `.to(tessiq.cpu)` moves it back; the two backends produce identical results on the
op suite; `tessiq.memory_stats(tessiq.gpu)` reports device bytes independently of the CPU pool.

**Expect it to be slow, and do not treat that as a defect.** Until the kernel library exists every
op still reaches the host through the CPU fallback, so a GPU tensor makes two transfers per
operation and will lose to plain CPU. What this phase buys is control and a correctness baseline:
tensors genuinely live in device memory, both backends are exercised side by side, and the CUDA
runtime is debugged while it is the only new thing in the picture. Kernels arrive in phases 6 and
7, and the transfers disappear with them.

**Why here rather than with the kernels.** Driver setup, stream semantics and async copies each
fail in their own confusing way. Landing them next to fusion and runtime codegen would mean
debugging three unfamiliar things at once, and every one of them can produce a wrong number
rather than an error.

### Phase 3: IR and pass infrastructure (done)
Graph/Node/Value, TensorType, builder, verifier, printer, parser. PassManager + registry. `dce`,
`cse`, `const-fold`. `tessiq-opt` CLI. Golden test harness.
**Milestone:** `tessiq-opt --passes=dce,cse in.tsr` produces expected output; parse/print roundtrip
is property-tested; no torch dependency in any of it.

### Phase 4: Graph capture
FX importer, capability partitioner, `torch.compile` backend registration, CPU fallback for
unsupported ops, naive lowering (one kernel per node, no fusion yet). Delivered as three pull
requests: the importer, then the partitioner, then the backend registration that joins them.
**Milestone:** `torch.compile(mlp, backend="tessiq")` runs an MLP end-to-end with correct results,
and a graph containing a deliberately unsupported op still produces correct results via fallback.

### Phase 5: Shape inference, memory planning, recipe cache
Shape inference pass, canonicalization set, liveness analysis, memory planner, `Recipe`, recipe cache
with signature hashing.
**Milestone:** second call with same shapes hits the cache (assert via counter); measured peak memory
below naive allocation; measured reduction in per-call overhead.

### Phase 6: Fusion and CUDA codegen
Elementwise fusion pass; fused kernel emitter; NVRTC compilation with on-disk cubin cache; benchmarks.
**Milestone:** a chain like `(a*b + c).relu().sigmoid()` runs as **one** kernel launch, with a
measured, reproducible speedup over the unfused path on the 5080, and bit-comparable results against
the CPU reference.

### Phase 7: Op coverage and real models
YAML op definitions + `gen_ops.py`; broad elementwise/reduction/movement coverage; GEMM via cuBLAS;
conv via cuDNN; layernorm/softmax kernels; layout assignment.
**Milestone:** a ResNet-18 block and a transformer encoder block compile and run correctly on both
backends. `docs/op_coverage.md` is generated and non-trivial.

### Phase 8: Training, dynamic shapes, tooling
AOTAutograd backward graphs; symbolic shapes + bucketing; profiler with chrome-trace export;
`tessiq-explain`; a write-up.
**Milestone:** a small model **trains** to convergence on the tessiq backend; sequence-length
variation doesn't trigger recompilation every step.

### Phase 9: Triton emitter (optional)
Second `KernelEmitter` implementation generating Triton kernels for fused clusters; autotuning;
head-to-head benchmark against the NVRTC emitter on identical clusters.
**Milestone:** `TESSIQ_EMITTER=triton` produces correct results on the full op test suite, and
`bench/emitters/` publishes a per-pattern comparison table of the two emitters.

### Phase 10: MLIR export target (optional)
`MlirTarget` implementing `LoweringTarget`; Tessiq IR → `linalg`-on-tensors converter; `gen_ops.py`
extended to emit ODS from the same YAML; standard MLIR bufferization + lowering pipeline to LLVM.
Gated behind `-DTESSIQ_ENABLE_MLIR=ON`, off by default.
**Milestone:** a subset of ops exports to valid MLIR, round-trips through `mlir-opt`, and executes
end-to-end on one model with results matching the CPU reference.

### Sequencing note

Phases 0–4 are the risky ones: they're where an unfamiliar API can cost a week. Phases 5–8 are
additive and mostly under your own control. If time gets tight, a repo that stops cleanly after
Phase 6, with fusion working and measured, is a strong artifact. A repo that sprawls into Phase 8
with a broken Phase 4 is not.

### The seams that phases 9 and 10 depend on

Both optional phases are additive only because three interfaces get built earlier. Retrofitting
them later would mean a rewrite.

| Seam | Built in | What it buys |
|---|---|---|
| `KernelEmitter` interface | Phase 6 | A Triton emitter becomes one new class and nothing else changes |
| `LoweringTarget` abstraction | Phase 5 | `Recipe` is one target; an MLIR module can be another |
| A small, orthogonal, target-neutral IR op set | Phase 3 onward | Ad-hoc fused ops have no `linalg` equivalent later |

**Phases 9 and 10 are explicitly optional**, and framing them that way is deliberate. Their value to
the project is mostly realized in Phases 5 and 6, when the seams get built. A plan that
*demonstrably could* take a Triton emitter or an MLIR target, with the interfaces in place to prove
it, already shows the architectural judgment. Building them is a bonus; designing for them is the
skill. Do not let either become the reason Phase 7 never ships.
