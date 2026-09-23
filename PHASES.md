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
| 4 | Graph capture | done |
| 5 | Shape inference, memory planning, recipe cache | done |
| 6 | Fusion and CUDA codegen | done |
| 7 | Op coverage and real models | done |
| 8 | Training, dynamic shapes, tooling | done |
| 9 | Every operation on the device | done |
| 10 | Where the time goes | done |
| 11 | Convolution | done |
| 12 | Triton emitter | optional |
| 13 | MLIR export target | optional |

---

### Phase 0: Scaffolding and engineering standards (done)
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

### Phase 4: Graph capture (done)
FX importer, capability partitioner, `torch.compile` backend registration, CPU fallback for
unsupported ops, naive lowering (one kernel per node, no fusion yet). Delivered as three pull
requests: the importer, then the partitioner, then the backend registration that joins them.
**Milestone:** `torch.compile(mlp, backend="tessiq")` runs an MLP end-to-end with correct results,
and a graph containing a deliberately unsupported op still produces correct results via fallback.

### Phase 5: Shape inference, memory planning, recipe cache (done)
Shape inference pass, canonicalization set, liveness analysis, memory planner, `Recipe`, recipe cache
with signature hashing. `Recipe` carries an output spec, because a backward graph returns `None` at
every position whose input takes no gradient and the IR records only the values a graph computes.
See `docs/design_notes/0002-output-structure-is-calling-convention.md`.
**Milestone:** second call with same shapes hits the cache (assert via counter); measured peak memory
below naive allocation; measured reduction in per-call overhead.

### Phase 6: Fusion and CUDA codegen (done)
Elementwise fusion pass; fused kernel emitter; NVRTC compilation with on-disk cubin cache; benchmarks.
**Milestone:** a chain like `(a*b + c).relu().sigmoid()` runs as **one** kernel launch, with a
measured, reproducible speedup over the unfused path on the 5080, and bit-comparable results against
the CPU reference.

### Phase 7: Op coverage and real models (done)
TOML op definitions + `gen_ops.py`; broad elementwise/reduction/movement coverage; GEMM via cuBLAS;
layernorm and softmax as decompositions rather than kernels; layout assignment.
**Milestone:** a ResNet-18 block and a transformer encoder block compile and run correctly on both
backends. `docs/op_coverage.md` is generated and non-trivial.
**Moved out:** conv via cuDNN, to phase 11. It is a second vendor library binding, and the phase's
own measurements say the backend is held back by transfers and launches rather than by the
operations it cannot run.

### Phase 8: Training, dynamic shapes, tooling (done)
AOTAutograd backward graphs; symbolic shapes; profiler with chrome-trace export;
`tessiq-explain`; a write-up.
**Milestone:** a small model **trains** to convergence on the tessiq backend; sequence-length
variation doesn't trigger recompilation every step. Twenty distinct sequence lengths through the
encoder block cause two backend compiles, and a twenty-first the compiler never saw still launches
all 28 kernels.
**Dropped:** bucketing. It was listed as the way to avoid recompiling per shape, and real symbolic
support removed the recompiling it was for. Padding to a few fixed sizes now buys only whatever
those sizes are worth to a kernel, needs masking first, and has no measurement asking for it.

### Phase 9: Every operation on the device (done)
A copy kernel, so that materialising a tensor does not mean sending it to the host and back.
`clone` is the last operation the encoder block still runs there, and `layout-assign` needs the
same kernel to materialise an operand a fused loop cannot read in place.
**Milestone:** `tools/model_coverage.py` reports the encoder block at 80 of 80 operations on the
device, and `tessiq-explain --sections schedule` shows nothing on the host.
**Why it is first.** It is four operations and the whole of what is left on that block. Phase 10
cannot say what the backend costs while part of the graph is still crossing the boundary for a
reason we already understand.

### Phase 10: Where the time goes (done)
Separating kernel time from the synchronisation the profiler adds to measure it, and a like for
like comparison against eager CUDA with the data resident on both sides.
**Milestone:** a measured breakdown of the gap between tessiq and eager CUDA on the encoder block
with the profiler's own cost accounted for rather than folded in, and a statement of which of
kernels, launches and transfers the remainder is.
**What it found.** None of the three. On the encoder block at batch 8, 128 steps:

| | |
|---|---|
| the call | 840 us |
| the device is busy | 310 us of it, across 28 kernels |
| so the device idles | about two thirds of the call |
| eager cuda, the same block | 230 us end to end |

Two separate problems, and the second was not on the list. The kernels cost 310 us, which is more
than eager CUDA spends on the entire block, so the generated code is not competitive on its own.
And the host takes 840 us to issue 310 us of work, so the device spends most of the call waiting.
Splitting that host time: about half is steps with no kernel going one at a time through PyTorch's
dispatcher, a third is issuing the launches, and a sixth is transfers. The executor is a Python
loop walking an FX graph, and that loop is the largest single cost in the call.
**Why it is a phase rather than an afternoon.** The current answer is 1.37 ms against 0.23 ms, and
the profiled split reads kernel 0.88, transfer 0.54, interpreted 0.40. That kernel figure drains
the device before stopping its clock, so it is an upper bound holding an unknown amount of
synchronisation. Every remaining decision in this project depends on which of those three numbers
is real, and none of them can be trusted until the instrument stops contributing to them.

### Phase 11: Convolution (done)
`aten.convolution` via cuDNN, and batch norm alongside it, which together take a ResNet-18 block
from a third of its operations on the device to all of them.
**Milestone:** the residual block in `tests/python/test_models.py` runs entirely on the device, and
`tools/model_coverage.py` reports it.
**Why it is optional and why it is here.** It is a binding rather than a compiler: cuDNN picks the
algorithm and tessiq passes descriptors. Phases 7 and 8 each raised coverage and found the time
going elsewhere, twice making the model slower, so a phase whose only product is more coverage
comes after the measurement that would show whether it helped.
**Read this before starting it.** Phase 10 was that measurement, and it says the backend is held
back by two things convolution does not touch: a Python executor that takes two thirds of the
call, and generated kernels that cost more than eager CUDA spends on the whole block. Adding
convolution would take the residual block from 43% of its operations on the device to all of them
and would not make it faster. It is worth doing for what a ResNet block demonstrates, and it is
not the next thing to do for speed.
**That note was wrong, and why is worth keeping.** It extrapolated from phase 10's measurement of
the encoder block, which was already entirely on the device, to the residual block, which was
not. There convolution sat on the host and forced a transfer each way around it, which is the
cost that goes away when it moves. Measured on the residual block at batch 8, 32 by 32:

| | per call |
|---|---|
| before, convolution and batch norm on the host | 1.45 ms |
| batch norm decomposed, convolution still on the host | 1.74 ms |
| both on the device | 0.59 ms |

Two and a half times faster, and only with both: batch norm on its own made it slower, for the
same reason every partial move in phases 7 and 8 did. The four operations left on the host are
float32 to float32 casts the batch norm decomposition inserts on its running statistics.

### Phase 12: Triton emitter (optional)
Second `KernelEmitter` implementation generating Triton kernels for fused clusters; autotuning;
head-to-head benchmark against the NVRTC emitter on identical clusters.
**Milestone:** `TESSIQ_EMITTER=triton` produces correct results on the full op test suite, and
`bench/emitters/` publishes a per-pattern comparison table of the two emitters.

### Phase 13: MLIR export target (optional)
`MlirTarget` implementing `LoweringTarget`; Tessiq IR → `linalg`-on-tensors converter; `gen_ops.py`
extended to emit ODS from the same TOML; standard MLIR bufferization + lowering pipeline to LLVM.
Gated behind `-DTESSIQ_ENABLE_MLIR=ON`, off by default.
**Milestone:** a subset of ops exports to valid MLIR, round-trips through `mlir-opt`, and executes
end-to-end on one model with results matching the CPU reference.

### Sequencing note

Phases 0–4 are the risky ones: they're where an unfamiliar API can cost a week. Phases 5–8 are
additive and mostly under your own control. If time gets tight, a repo that stops cleanly after
Phase 6, with fusion working and measured, is a strong artifact. A repo that sprawls into Phase 8
with a broken Phase 4 is not.

Phases 9 and 10 were added after 8 rather than planned with the rest. Phase 8 left the encoder
block with four operations on the host and a profiler whose own cost sits inside the figure it
reports, and both of those have to be settled before a phase that adds coverage can be judged on
anything but coverage.

### The seams that phases 12 and 13 depend on

Both optional phases are additive only because three interfaces get built earlier. Retrofitting
them later would mean a rewrite.

| Seam | Built in | What it buys |
|---|---|---|
| `KernelEmitter` interface | Phase 6 | A Triton emitter becomes one new class and nothing else changes |
| `LoweringTarget` abstraction | Phase 5 | `Recipe` is one target; an MLIR module can be another |
| A small, orthogonal, target-neutral IR op set | Phase 3 onward | Ad-hoc fused ops have no `linalg` equivalent later |

**Phases 9 to 11 are explicitly optional**, and framing them that way is deliberate. For the two
alternative backends the value is mostly realized in phases 5 and 6, when the seams get built: a
plan that *demonstrably could* take a Triton emitter or an MLIR target, with the interfaces in
place to prove it, already shows the architectural judgment. Building them is a bonus; designing
for them is the skill.

Convolution is optional for a different reason. It is a binding rather than a compiler, and phase 7
ended with the backend held back by transfers and launches rather than by the operations it cannot
run, so more coverage is not what the next measurement wants.
