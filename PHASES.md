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
| 12 | Custom operations and rewrite passes | done |
| 13 | Precompiled kernels | planned |
| 14 | Triton emitter | optional |
| 15 | MLIR export target | optional |

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
Graph/Node/Value, TensorType, builder, verifier, printer, parser. PassManager + registry. `dead-node-elimination`,
`cse`, `const-fold`. `tessiq-opt` CLI. Golden test harness.
**Milestone:** `tessiq-opt --passes=dead-node-elimination,cse in.tsr` produces expected output; parse/print roundtrip
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
cost that goes away when it moves. Measured on the residual block at batch 8, 32 by 32, with each
commit built and the three timed alternately for five rounds:

| | median | range |
|---|---|---|
| before, convolution and batch norm on the host | 0.90 ms | 0.84 to 1.02 |
| batch norm decomposed, convolution still on the host | 1.41 ms | 1.15 to 1.66 |
| both on the device | 0.57 ms | 0.56 to 0.58 |

About 1.6 times faster, and only with both: batch norm on its own made it slower, for the same
reason every partial move in phases 7 and 8 did. This table first said 1.45 ms before and two and a
half times faster. That was one run of the old code while the machine was busy, and the versions
that convolve on the host are the ones that vary. The four operations left on the host are
float32 to float32 casts the batch norm decomposition inserts on its running statistics.

### Phase 12: Custom operations and rewrite passes (done)
Adding an operation with its own device kernel as one directory under `csrc/custom_ops/`, holding
the kernel and the C++ that defines the operation, with nothing else in tessiq edited. Then a
pattern rewriter that finds the subgraph a custom operation replaces. Each rewrite is its own named
pass.
**What happens today.** An operation declared with `torch.library.custom_op` compiles and gives
the right answer, but the compiler has never heard of it, so it runs on the host between two device
kernels and the data crosses the bus both ways:

```
  mul         aten.mul.Tensor           kernel %mul (moves its inputs)
  fast_gelu   demo.fast_gelu.default    host (moves its inputs)
  add         aten.add.Tensor           kernel %relu (moves its inputs)
```

Giving it a GPU kernel means a row in `ops.toml`, a category the passes understand, an extraction,
a kernel struct, a `compile_*` function, a slot in `CompiledGraph`, a binding, an entry in the
recipe's kernel map, and a rebuild. The four kernel kinds that exist were each wired in that way.
**What it adds.**
- A kernel protocol. Fused loops, reductions, matmuls and convolutions already answer the same two
  questions for the executor, what they read and how to run. That becomes an interface, a custom
  kernel is a fifth implementation of it, and nothing downstream counts how many kinds exist. The
  interface is "launch these inputs into these outputs on this stream", with the stream passed
  explicitly, and not the argument list of `cuLaunchKernel`. A kernel tessiq did not compile has to
  fit it too, which is what phase 13 adds.
- A registry of operations added at run time, consulted after the generated table. A registered
  operation carries its schema, the dtypes and ranks it has a kernel for, since support depends on
  the types (phase 8), and a reference implementation that is both the CPU backend's path and the
  oracle its kernel is tested against.
- Two ways to supply the kernel. An elementwise operation gives an expression and joins fusion
  groups like a built-in one. Anything else gives a `.cu` file and a C++ function that lays out
  its launch. The build carries the file's text in the extension, and it compiles through the
  existing NVRTC path and cubin cache for whichever GPU it runs on. It takes its extents at the
  launch, so it works on symbolic shapes.
- The PyTorch side, done by the definition: a `TORCH_LIBRARY` operation whose ATen reference runs
  on the CPU and, on tensors with no data, gives the result's shape for tracing, and an
  implementation for the tessiq dispatch key so eager tensors on the device run the same kernel.
- Replacing a torch operation. A custom operation can declare that it stands in for one, such as
  a softmax kernel for `aten::_softmax`. It becomes that operation's implementation on tessiq's
  devices, the decomposition table keeps the operation whole where the custom one has the types,
  and a pass of its own, `override-<name>`, swaps the node before anything else sees it. The torch
  operation keeps its backward.
- A pattern rewriter. A pattern is a small IR graph, either written in the textual form or traced
  from a Python function through the backend's own decompositions, so that it matches what
  AOTAutograd emits rather than what the user typed. Each rewrite registers as its own pass,
  `rewrite-<name>`, which can be named in a pipeline, is listed by `tessiq-opt --list`, and is
  tested with golden files. A match needs every interior value of the pattern to have no user
  outside it, which is the rule fusion already uses.
- No new execution machinery. The node that replaces a match keeps the root's value name and
  records the nodes it absorbed, which is how a fused kernel already relates to the FX graph the
  executor walks.

**Rules it keeps.**
- A rewrite runs only when a compile names it or its author declares it exact. A kernel that
  approximates changes the numbers, and phase 5 settled that the compiler does not do that
  silently.
- The backward graph is not rewritten. AOTAutograd derived it from the original operations, and a
  forward value it saved is a graph output, so a pattern covering that value does not match and
  training stays correct. A model that calls a custom operation directly needs an autograd
  registration for it, which is PyTorch's mechanism rather than ours.
- A custom operation has no `linalg` form, which cuts against the op-set seam phase 15 relies on.
  It exports through its reference implementation's decomposition, or as a call to an external
  function.

**Milestone:**
1. `tessiq::fast_gelu`, given as an expression, and `tessiq::rms_norm`, given as a CUDA kernel that
   does the reduction and the elementwise pass in one launch, are each one directory in
   `csrc/custom_ops/` with no change anywhere else in tessiq. Each runs on a tessiq tensor in eager
   mode and under `torch.compile`, and matches its reference within its declared tolerance on both
   backends.
2. `rewrite-fast-gelu` and `rewrite-rms-norm` are separate passes with golden files, including
   inputs they must leave alone: an interior value used elsewhere, and a dtype the kernel does not
   support.
3. A model with RMSNorm and the GELU approximation written out in plain PyTorch compiles with both
   rewrites named, `tessiq-explain` shows each matched subgraph as one custom launch, the result
   matches eager, the model still trains, and the time with and without the rewrites is measured
   and reported whichever way it goes.

**What it found.** The three milestones hold. `RMSNormBlock` in `tests/python/test_models.py` is
RMSNorm, a linear layer and the sigmoid GELU in plain PyTorch. Compiled with
`options={"rewrites": ["rms_norm", "fast_gelu"]}`, the norm's four kernels become one `rms_norm`
launch and the GELU becomes a `fast_gelu` call inside the loop it was already fused into, which
`tools/tessiq_explain.py rmsnorm --rewrites rms_norm,fast_gelu` shows. It matches eager, trains to
the same parameters, and against the same block compiled as written:

| | kernels | device time | call |
|---|---|---|---|
| 8×512, rewritten | 3 | 34 us | 0.181 ms |
| 8×512, as written | 6 | 60 us | 0.219 ms |
| 256×4096, rewritten | 3 | 335 us | 0.801 ms |
| 256×4096, as written | 6 | 364 us | 0.827 ms |

All of the difference is the norm. The GELU was one fused loop before its rewrite too, so the rewrite
changes the expression in that loop and not the number of launches. At the larger size the 4096 by
4096 matmul is most of the device time, so the rewrites take 8% off it and 3% off the call. The
figures are medians of four runs.

What the plan did not expect:
- **A rewrite does not reach training.** A training step's forward graph returns the values its
  backward graph reads, and the rule that keeps those values intact means the patterns find
  nothing there. That is the rule doing its job, and it makes the gain inference only until a
  custom operation can carry its own backward.
- **Replacing a torch operation was the more useful half.** A custom operation that declares
  `.replaces("aten::_softmax")` needs no rewrite named, and the torch operation keeps its backward,
  so it holds in training too. The encoder block went from 28 kernels to 25 that way.
- **Some of the plan was not needed.** Patterns are written in the textual IR only, since nothing
  about an operation lives in Python, and every rewrite runs only when a compile names it, because
  neither pattern needed the other half of that rule.
- **Two bugs in existing code surfaced.** A claimed elementwise or reduction kernel that read a
  transposed or broadcast value never compiled, which 12.5 fixed. And a float64 matmul is claimed
  and then refused by cuBLAS at the call, which is still open.

**Why here, before Triton.** The Triton emitter needs the same kernel protocol, and the Triton
kernel people most often write is a hand-written custom operation rather than a generated fused
cluster. Building the seam first makes a Triton-sourced custom operation a new source language
rather than a sixth path wired in by hand.

### Phase 13: Precompiled kernels (in progress)
A custom operation whose kernel is compiled ahead of time with nvcc, so that it can use CUTLASS,
cuBLAS or anything else that needs host code, which phase 12's run-time compiled source cannot do.
**Why phase 12 is not enough.** NVRTC compiles device code only. A cuBLAS call is host code with a
handle. A CUTLASS GEMM does its setup on the host and takes long enough to compile that it is built
with nvcc ahead of time, for the architecture of the card it will run on.
**What it adds.**
- A precompiled operation is a directory in `csrc/custom_ops/`, like every other custom operation.
  Its `.cu` is compiled by nvcc when tessiq is built, and its `.cpp` defines the operation as
  phase 12's do. Nothing about it lives in Python, and adding one is a directory and a rebuild.
- A launch function that runs host code. Rather than returning a grid, a block and arguments for
  tessiq to launch, it is given the operation's tensors, the stream to launch on, and a way to ask
  tessiq's allocator for workspace, and it launches what it needs itself. It fits phase 12's kernel
  interface, so the executor, the registry and the rewriter do not change.
- Built for the card it runs on. The build compiles for the architectures CMake detects or is told,
  and an operation whose kernel was not built for the card it is asked to run on is refused with a
  message naming both, rather than failing at the first launch, which is how the `sm_80` cubin
  failed in phase 6.
- A build without nvcc still builds. A precompiled operation then has no kernel and runs its
  reference, as every custom operation does on a machine without a GPU.
- CUTLASS only for the operations that use it. It is header-only and fetched for those builds, and
  the rest of the backend does not depend on it.

**Rules it keeps.**
- A launch uses the stream it is handed and sets any library handle to that stream. Every tessiq
  launch and the cuDNN handle use stream 0 today, and a kernel queued on another stream can run
  before its inputs exist.
- Workspace comes from tessiq's allocator, so the memory statistics include it and running out goes
  through the allocator's release-and-retry.
- A precompiled operation registers a reference in ATen like any custom operation, which is the
  CPU backend's path and the oracle its kernel is tested against.

**What changed from the first plan.** It had the user build a shared library for tessiq to load at
run time, with tensors handed over as DLPack, the interface versioned, and the operation registered
from Python. Phase 12 settled that an operation lives in `csrc/custom_ops/` and that nothing about
it lives in Python. A kernel built with the extension needs no loader, no version check and no
DLPack: it is handed tensors the way phase 12's launch functions are. A library loaded at run time
is still worth having for a kernel that cannot be built alongside tessiq, and would be a phase of
its own.

**Progress.** The interface is in: `.precompiled()`, host code in `<name>.nvcc.cu` registered with
`TESSIQ_PRECOMPILED`, workspace from the allocator, the stream it is handed, and the architecture
check. `layer_norm` shows it with two launches and workspace between them.

Milestones 1 and 2 are in. `gemm_bias_gelu` is a CUTLASS float32 GEMM with the bias and GELU in its
epilogue, splitting K through semaphores in workspace when the output has few tiles. With
`rewrite-gemm-bias-gelu`, `tessiq-explain` shows a linear layer and GELU as one custom kernel where
there were a GEMM and a fused GELU kernel. On an RTX 5080, `bench/gemm_bias_gelu.py` has it about
12% faster at 4096 x 1024 -> 4096, level at 64 x 4096 -> 4096, a few percent slower at
8192 x 768 -> 3072 and about 10% slower at 512 x 768 -> 3072. The pattern matches a 2-D input only: a batched one arrives with views
around the addmm. The cuBLAS operation is next.

**Milestone:**
1. A CUTLASS GEMM whose epilogue, the stage that runs on each output element before it is written,
   adds the bias and applies GELU, in `csrc/custom_ops/`, built with nvcc.
2. `rewrite-gemm-bias-gelu` maps `addmm` followed by GELU onto it. `tessiq-explain` shows one launch
   where there were two, the result matches eager within the operation's tolerance, and it is timed
   against the current path of a cuBLAS matmul followed by a fused GELU kernel, reported whichever
   way it goes.
3. A second operation calls cuBLAS directly, to show that a library call is defined the same way as
   a kernel.

**Why here.** It needs phase 12's kernel interface, registry and rewriter, and without the rewriter
a GEMM with a fused epilogue only helps models that call it by name. It comes before Triton because
a Triton kernel compiled ahead of time is one more precompiled operation.

### Phase 14: Triton emitter (optional)
Second `KernelEmitter` implementation generating Triton kernels for fused clusters; autotuning;
head-to-head benchmark against the NVRTC emitter on identical clusters.
**Milestone:** `TESSIQ_EMITTER=triton` produces correct results on the full op test suite, and
`bench/emitters/` publishes a per-pattern comparison table of the two emitters.

### Phase 15: MLIR export target (optional)
`MlirTarget` implementing `LoweringTarget`; Tessiq IR → `linalg`-on-tensors converter; `gen_ops.py`
extended to emit ODS from the same TOML; standard MLIR bufferization + lowering pipeline to LLVM.
Gated behind `-DTESSIQ_ENABLE_MLIR=ON`, off by default.
**Milestone:** a subset of ops exports to valid MLIR, round-trips through `mlir-opt`, and executes
end-to-end on one model with results matching the CPU reference. A custom operation from phase 12
or 13 exports through its reference decomposition.

### Sequencing note

Phases 0–4 are the risky ones: they're where an unfamiliar API can cost a week. Phases 5–8 are
additive and mostly under your own control. If time gets tight, a repo that stops cleanly after
Phase 6, with fusion working and measured, is a strong artifact. A repo that sprawls into Phase 8
with a broken Phase 4 is not.

Phases 9 and 10 were added after 8 rather than planned with the rest. Phase 8 left the encoder
block with four operations on the host and a profiler whose own cost sits inside the figure it
reports, and both of those have to be settled before a phase that adds coverage can be judged on
anything but coverage.

Phase 12 was added after 11. A custom operation compiled correctly and ran on the host, and adding
a kernel for one took edits in eight places and a rebuild, which is the wrong cost for the thing a
backend's users most often want to do. Phase 13 follows it for kernels that phase 12's run-time
compiler cannot build, such as CUTLASS templates and calls into cuBLAS.

### The seams that phases 14 and 15 depend on

Both optional phases are additive only because four interfaces get built earlier. Retrofitting
them later would mean a rewrite.

| Seam | Built in | What it buys |
|---|---|---|
| `KernelEmitter` interface | Phase 6 | A Triton emitter becomes one new class and nothing else changes |
| `LoweringTarget` abstraction | Phase 5 | `Recipe` is one target; an MLIR module can be another |
| A small, orthogonal, target-neutral IR op set | Phase 3 onward | Ad-hoc fused ops have no `linalg` equivalent later |
| The kernel protocol and the operation registry | Phase 12 | A Triton kernel is a new source language, not a new path through the executor |

**Phases 9 to 11 are explicitly optional**, and framing them that way is deliberate. For the two
alternative backends the value is mostly realized in phases 5 and 6, when the seams get built: a
plan that *demonstrably could* take a Triton emitter or an MLIR target, with the interfaces in
place to prove it, already shows the architectural judgment. Building them is a bonus; designing
for them is the skill.

Convolution is optional for a different reason. It is a binding rather than a compiler, and phase 7
ended with the backend held back by transfers and launches rather than by the operations it cannot
run, so more coverage is not what the next measurement wants.
