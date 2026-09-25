# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CMake build, scikit-build-core packaging, pybind11 extension module.
- Support layer: `TESSIQ_CHECK`/`TESSIQ_DCHECK`, `Error`/`ErrorCode`, `Result<T>` with
  `TESSIQ_RETURN_IF_ERROR` and `TESSIQ_ASSIGN_OR_RETURN`, and module-scoped logging that skips
  argument evaluation when the level is off.
- `ErrorCode` to Python exception mapping in the bindings, so callers can use `except` rather than
  matching on message text.
- Quality gates: `-Werror`, clang-format, clang-tidy, ruff, mypy, ASan/UBSan, and CI running all
  of them.
- `requirements-dev.txt` pinning the tooling. CI and the pre-commit hooks install from it, which
  stops a hook disagreeing with the gate it mirrors.
- Device runtime: a `Device` interface with a CPU reference backend, and a caching allocator with
  size classes, reuse, statistics and `empty_cache()`.
- Registration with PyTorch as the `tessiq` device. `torch.randn(n, device="tessiq")`,
  `.to("tessiq")` and `.cpu()` work, including for non-contiguous tensors and across dtypes.
- A CPU fallback for ops without a native kernel, so the whole ATen surface is usable now and
  kernels can replace it one at a time.
- `tessiq.memory_stats()`, `memory_allocated()`, `memory_reserved()` and `empty_cache()`.
- A CUDA backend, selected per tensor: `tessiq.cpu` is the reference, `tessiq.gpu` the first GPU,
  `tessiq.gpu_at(n)` the rest. Each device has its own allocator, so memory statistics are per
  backend rather than pooled.
- Copies between backends, staged through the host.
- An SSA IR with a textual form: `Graph`, `Node`, `Value`, `TensorType` with dynamic dimensions,
  use-def chains, a verifier, and a printer and parser that round-trip.
- Pass infrastructure: a `Pass` interface, a name-keyed registry, and a `PassManager` that
  verifies the graph after each pass in debug builds.
- `dce`, `cse` and `const-fold`.
- `tessiq-opt`, which runs a named pipeline over textual IR, and golden-file pass tests.
- `tessiq.import_graph(gm)`, which converts the FX graph a `torch.compile` backend receives into
  tessiq IR and returns its textual form.
- Partitioning, as two passes: `mark-supported` records whether the backend can run each node, and
  `cluster` groups neighbouring supported nodes into units, refusing any merge that would make the
  graph unexecutable.
- `tessiq.run_passes(ir, pipeline)` and `tessiq.pass_names()`.
- `shape-infer`, which derives result types from operand types and checks the ones already
  recorded. Resolves dimensions a graph left unknown, and reports a disagreement between two
  known sizes rather than overwriting one with the other.
- `algebraic-simplify`, which replaces operations whose result is already available: multiply
  or divide by one, add or subtract an integer zero, double negation, and repeated `relu` or
  `abs`. The additive identities are integer-only because `(-0.0) + 0.0` is `+0.0`.
- Liveness analysis, and a memory planner that packs every intermediate into one arena,
  reusing space between values that are never live at the same time. `tessiq-opt
  --memory-plan` prints the result, including what the same buffers would cost with no reuse.
- `tessiq.build_recipe(gm, device)`, which lowers a graph into a recipe: the steps to run,
  which device each runs on, which of them need their arguments moved first, the arena size,
  and the shape of the returned tuple. `tessiq.execute(recipe)` runs one.
- **Changed.** `tessiq.execute` takes a recipe rather than a graph and a cluster map.
- A recipe cache, keyed on the graph's structure, the type of every value in it, and the
  device it was lowered for. `tessiq.cache_stats()` reports hits, misses, entries and
  evictions; `tessiq.clear_cache()` empties it. Bounded and least-recently-used, because a
  recipe holds the graph it was built from.
- `fuse-elementwise`, which groups chains of elementwise operations that can run as one
  kernel. A value read twice, returned by the graph, or changing shape stays out, because
  each of those means it has to be written to memory anyway.
- A code generator: `tessiq.emit_cuda(ir)` and `tessiq-opt --emit-cuda` turn a fusion group
  into a CUDA kernel, with the intermediates in registers. One `Emitter` interface, so a
  second target is a second class rather than a second path.
- Runtime compilation through NVRTC, with the compiler's own log on failure, and an on-disk
  cubin cache keyed by the source, the architecture and the compiler version. Set
  `TESSIQ_CACHE_DIR` to move it; a directory it cannot write only makes compiling slower.
- Loading and launching generated kernels through the CUDA driver, with modules kept for the
  life of the process. `device_capability()` reports what a cubin has to be built for.
- Fused kernels run. `torch.compile(model, backend="tessiq", options={"device": "tessiq:1"})`
  compiles each fusion group into one kernel and launches it in place of the operations it covers.
  On a 4M-element `(a*b + c).relu().sigmoid()` that is one launch instead of four and 5.7x faster
  than the same graph without kernels. See `bench/fusion.py`.
- Operations are defined once, in `ops.toml`. `tools/gen_ops.py` generates the C++ table and
  `docs/op_coverage.md` from it, and a test fails if either is out of date. The two
  hand-written tables it replaces disagreed: `aten.add.Scalar` and its siblings were
  elementwise without being claimed, and are now claimed.
- `tessiq.partition(gm)`, which imports an FX graph, partitions it, and returns the IR along
  with the cluster each node landed in.
- `tessiq.execute(gm, clusters)`, which runs a partitioned graph: clustered nodes on a tessiq
  device, everything else on the host, results back on the host.
- `torch.compile(model, backend="tessiq")`, declared as a `torch_dynamo_backends` entry point
  so the name resolves without importing tessiq first. Forward and backward both go through
  the compiler. `options={"device": "tessiq:1"}` picks which device clusters run on.
- Broad operation coverage: elementwise, reductions, movement and matmul, defined in `ops.toml`
  and generated into the C++ table. Softmax and layer norm arrive as the operations they are
  made of, through decompositions, rather than as kernels of their own.
- Reduction kernels, described as `[outer, reduce, inner]` so one shape serves every rank. Which
  of the two kernel forms runs depends on whether a kept axis follows the reduced one, because
  that decides whether consecutive threads read consecutive addresses.
- Matrix multiplies through cuBLAS, with the transpose flags read off the layout so a weight
  stored the other way round costs a flag rather than a copy.
- Layouts are part of a tensor type. `layout-assign` materialises an operand a kernel cannot
  read in place, and a fused kernel works out where to read a broadcasting or permuted operand
  from the shapes rather than needing it packed first.
- Views on a tessiq tensor alias their storage instead of copying, so `x.view(n).fill_(v)` is
  visible through `x`.
- Inputs that have not changed stay on the device between calls, keyed on identity and version.
- `tessiq.profile()`, which records kernels, interpreted operations and transfers with exclusive
  time, and writes a Chrome trace. Kernels are drained before the clock stops, because a launch
  returns before the work does.
- Training. AOTAutograd's backward graph goes through the same importer, passes and kernels as
  the forward one, so a model trains on the backend without the backend knowing training exists.
- Shapes that are not known until the call. The IR records an unknown extent as `?`, and every
  kernel takes the numbers it needs as arguments: the element count and a broadcast's divisor
  for a fused loop, three extents for a reduction, and m, n and k for a matmul. Twenty distinct
  sequence lengths through a transformer encoder block cause two compiles rather than twenty,
  and get the same 28 kernels a fixed shape does.
- `tessiq.explain(model, example)` and `tools/tessiq_explain.py`, which report one model's whole
  compilation: the imported graph, what each pass changed, the kernels, and where every step
  runs. `tessiq.capture()` and `tessiq.default_pipeline()` alongside it, so nothing reporting on
  a compilation has to keep its own copy of how one is made.
- A kernel that packs a view, so a strided tessiq tensor becomes contiguous on the device rather
  than through the host. `clone` is claimed as its own `copy` category, which put the last four
  operations of the encoder block on the device.
- A clock on the device. The profiler no longer drains the device around every kernel, which had
  more than doubled the call it was measuring, and `tessiq.profile(device_time=True)` reports what
  each kernel cost the device. `tessiq.kernel_costs(recipe)` measures each kernel by replaying it,
  and `bench/where_the_gap_is.py` splits the gap to eager CUDA into the code tessiq generates and
  everything around it.
- Convolution through cuDNN, and batch norm in eval mode decomposed into arithmetic. The ResNet-18
  residual block runs entirely on the device and takes 0.57 ms, against 0.90 ms with convolution
  and batch norm on the host.
- One `Kernel` interface for every kind of compiled work, which the executor runs without knowing
  what kind it is, and a stream on every launch.
- A launch over any grid and block, with dynamic shared memory and typed arguments.
- Custom operations, one directory each in `csrc/custom_ops/`. A `.cpp` gives the schema, what
  the operation computes in ATen, and either a launch function for a `.cu` kernel beside it or an
  elementwise expression the compiler fuses. Each becomes `torch.ops.tessiq.<name>` and runs its
  kernel on a tessiq GPU, called directly and in a compiled graph. `tessiq.library` lists them,
  compiles them ahead of their first call, and counts the calls that ran the reference. `rms_norm`,
  `fast_gelu`, `swiglu` and `softmax` ship with it.
- A custom operation can replace a torch one, with `.replaces("aten::_softmax")`. It becomes the
  torch operation's implementation on tessiq's devices and gets a pass, `override-<name>`, that
  swaps the node in a graph. The torch operation keeps its backward. `softmax` replaces
  `aten::_softmax`.
- A custom operation can carry a pattern: the computation it replaces, written out in tessiq's IR.
  Its pass, `rewrite-<name>`, replaces each match with one call, when a compile names it in
  `options={"rewrites": [...]}`. `rms_norm` and `fast_gelu` rewrite what models write out.
- `docs/passes.md`, on the passes a compile runs, in what order and why, and how to add one.
- `ops.toml` can name the element types an operation takes. The matmuls and the convolution name
  float32, since cuBLAS and cuDNN are float32 only here.
- `tessiq.current_device()`, `set_device()` and `synchronize()`, which `torch.tessiq` uses too.
- torch's PrivateUse1 hooks. `pin_memory()` and a DataLoader with `pin_memory=True` work,
  page-locking when there is a GPU. `torch.Generator(device="tessiq")` gives a host generator,
  since random operations run on the host. A tessiq storage can be resized, and
  `torch.accelerator.synchronize()` waits for the device.

### Changed

- `dce` is called `dead-node-elimination`.
- A custom operation's launch function that declines a call with `TORCH_CHECK` makes the call run
  the reference, rather than fail.

### Fixed

- The profiler reported every input reused from an earlier call as a transfer of the whole tensor.
- `tessiq-explain` showed operations that run on the host as running on the device.
- A convolution on a tessiq tensor with no cuDNN kernel for it, such as one on the CPU reference
  device, raised instead of running on the host.
- A kernel reading a transposed or broadcast value did not compile, which also stopped training a
  module whose loss is computed inside it.
- The IR could print a string attribute holding a comma and not read it back.
- A float64 matmul or convolution compiled for the GPU was claimed and then refused by cuBLAS or
  cuDNN when it was called. It runs on the host now.
- A fusion group compiled for more than one shape sized its result from its first input. When that
  input broadcast, as in `a + b` with `a` of shape `[n, 1]` and `b` of shape `[1, 32]`, the result
  came back as `[n, 1]` with no error whenever `n` was a multiple of 32, and the call raised
  otherwise.
- `torch.tessiq.synchronize()` returned without waiting, and `torch.tessiq.set_device()` was
  ignored. A GPU device's `synchronize()` also waited only for transfers and not for kernels.
- `torch.arange` on a GPU, and any other operation whose output the fallback resizes, failed with
  `cudaMemcpyAsync: invalid argument`. `resize_` allocated on the thread's current device rather
  than the output's.
